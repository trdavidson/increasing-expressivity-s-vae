import math
import torch
from torch.distributions.kl import register_kl

from hyperspherical_vae.ops.ive import ive
from hyperspherical_vae.distributions.hyperspherical_uniform import HypersphericalUniform


# noinspection PyCallingNonCallable
class VonMisesFisher(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        # return self.loc * (ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale))
        # return self.loc * self.__ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)

        return self.loc * self.__ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, r=torch.Tensor([1.]), validate_args=None):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.r = r.to(loc.device)
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = (torch.Tensor([1.] + [0] * (loc.shape[-1] - 1))).to(self.device).type(self.dtype)
        
        super(VonMisesFisher, self).__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size(), eps=1e-20):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        
        w = self.__sample_w3(shape=shape, eps=eps) if self.__m == 3 else self.__sample_w_rej(shape=shape, eps=eps)

        v = (torch.distributions.Normal(0, 1).sample(
            shape + torch.Size(self.loc.shape)).to(self.device).transpose(0, -1)[1:]).transpose(0, -1).type(self.dtype)
        v = v / v.norm(dim=-1, keepdim=True)
        
        x = torch.cat((w, torch.sqrt((1 - (w ** 2)).clamp(eps)) * v), -1)
        z = self.__householder_rotation(x)
        z = z * self.r

        return z.type(self.dtype)
    
    def __sample_w3(self, shape, eps=1e-20):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0+eps, 1-eps).sample(shape).to(self.device)
        self.__w = 1 + torch.stack([torch.log(u), torch.log(1 - u) - 2. * self.scale], dim=0).logsumexp(0) / self.scale
        return self.__w

    def __sample_w_rej(self, shape, eps=1e-20):
        c = torch.sqrt(((4 * (self.scale ** 2)) + (self.__m - 1) ** 2).clamp(eps))
        b_true = (-2. * self.scale + c) / (self.__m - 1)
        
        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)
#         s = torch.min(torch.max(torch.Tensor([0.]), self.scale - 10), torch.Tensor([1.]))
        s = torch.min(torch.max(torch.tensor([0.], dtype=self.dtype, device=self.device), self.scale - 10.),
                      torch.tensor([1.], dtype=self.dtype, device=self.device))
        b = b_app * s + b_true * (1 - s)
        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, eps=eps)
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = (x > 0).int()
        idx = torch.where(mask.any(dim=dim), mask.argmax(dim=1).squeeze(),
                          torch.tensor(invalid_val, device=x.device))
        return idx

    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        #  matrix while loop: samples a matrix of [A, k] samples, to avoid looping all together
        b, a, d = [e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1) for e in (b, a, d)]
        w, e, bool_mask = torch.zeros_like(b).to(self.device), torch.zeros_like(
            b).to(self.device), (torch.ones_like(b) == 1).to(self.device)

        sample_shape = torch.Size([b.shape[0], k])
        shape = shape + torch.Size(self.scale.shape)

        while bool_mask.sum() != 0:
            con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            e_ = torch.distributions.Beta(con1, con2).sample(sample_shape).to(self.device).type(self.dtype)

            u = torch.distributions.Uniform(0 + eps, 1 - eps).sample(sample_shape).to(self.device).type(self.dtype)

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1.) * t.log() - t + d) > torch.log(u)
            accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
            accept_idx_clamped = accept_idx.clamp(0)
            # we use .abs(), in order to not get -1 index issues, the -1 is still used afterwards
            w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
            e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))

            reject = (accept_idx < 0)
            accept = ~reject #(1 - reject)

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e.reshape(shape), w.reshape(shape)

    def __householder_rotation(self, x):
        u = (self.__e1 - self.loc)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-20)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    @staticmethod
    def __ive_fraction_approx(v, kappa):
        # I_(m/2)(k) / I_(m/2 - 1)(k) >= z / (v-1 + ((v+1)^2 + z^2)^0.5
        return kappa / (v - 1 + torch.pow(torch.pow(v + 1, 2) + torch.pow(kappa, 2), 0.5))

    @staticmethod
    def __ive_fraction_approx2(v, kappa, eps=1e-20):

        def delta_a(a):
            lamb = v + (a - 1.) / 2.
            return (v - 0.5) + lamb / (2 * torch.sqrt((torch.pow(lamb, 2) + torch.pow(kappa, 2)).clamp(eps)))

        delta_0 = delta_a(0.)
        delta_2 = delta_a(2.)
        B_0 = kappa / (delta_0 + torch.sqrt((torch.pow(delta_0, 2) + torch.pow(kappa, 2))).clamp(eps))
        B_2 = kappa / (delta_2 + torch.sqrt((torch.pow(delta_2, 2) + torch.pow(kappa, 2))).clamp(eps))

        return (B_0 + B_2) / 2.

    def entropy(self):
        # output = - self.scale * ive(self.__m / 2, self.scale) / ive((self.__m / 2) - 1, self.scale)
        # output = - self.scale * self.__ive_fraction_approx(torch.tensor(self.__m/2), self.scale)
        output = - self.scale * self.__ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

        return output.view(*(output.shape[:-1])) + self._log_normalization() + (self.__m - 1) * torch.log(self.r)
        
    def log_prob(self, x):
        # g(z) = f(z') * 1/r^(m-1)
        # g(z) = f(z / r) * 1/r^(m-1)
        a = self._log_unnormalized_prob(x / self.r)
        b = self._log_normalization()
        c = ((self.__m - 1) * torch.log(self.r)).squeeze()
        d = a - b - c
        return d

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)

        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = - ((self.__m / 2 - 1) * torch.log(self.scale) - (self.__m / 2) * math.log(2 * math.pi) - (
                self.scale + torch.log(ive(self.__m / 2 - 1, self.scale))))

        return output.view(*(output.shape[:-1]))


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return - vmf.entropy() + hyu.entropy()
