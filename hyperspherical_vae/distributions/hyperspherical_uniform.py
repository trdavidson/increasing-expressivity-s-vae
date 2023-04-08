
import math
import torch


# noinspection PyCallingNonCallable
class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    @property
    def radius(self):
        return self._r

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if isinstance(val, torch.device) else torch.device(val)
    
    def __init__(self, dim=3, r=torch.tensor([1.]),  device="cpu", validate_args=None):
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), r, validate_args=validate_args)
        self._dim = dim
        self._r = r.to(device)
        self.device = device

    def sample(self, shape=torch.Size()):
        output = torch.distributions.Normal(0., 1.).sample(
            (shape if isinstance(shape, torch.Size) else torch.Size([shape])) +
            torch.Size([self._dim + 1])).to(self.device)

        return output / output.norm(dim=-1, keepdim=True) * self._r

    def entropy(self):
        return self.__log_surface_area()
    
    def log_prob(self, x):
        return - torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area().transpose(-1, 0)

    def __log_surface_area(self):
        # S(S^m) = r^m (2pi^(m-1/2) / gamma((m-1)/2)))
        # log(S) = m*log(r) + log(2) + (m-1/2)*log(pi) - log(gamma(m-1/2))
        return self._dim * torch.log(self._r) + math.log(2.) + ((self._dim + 1) / 2.) * \
               math.log(math.pi) - torch.lgamma(torch.tensor([(self._dim + 1) / 2.], device=self.device))
