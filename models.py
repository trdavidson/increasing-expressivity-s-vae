import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform
from utils.model_utils import create_encoder, create_decoder, lower_triangular_matrix_from_vector


class VAE(nn.Module):

    def __init__(self, h_dims, z_dim, input_size=[1, 28, 28], input_type='binary', distribution='normal',
                 encode_type='mlp', decode_type='mlp', device='cpu', flags=None):
        """
        ModelVAE initializer
        :param h_dims: dimension of the hidden layers, list
        :param z_dim: dimension of the latent representation
        :param input_size: dimensions of input
        :param input_type: (str) e.g binary
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        :param encode_type: (str) one of {mlp, cnn}
        :param decode_type: (str) one of {mlp, cnn}
        :param device: device to use
        :param flags: user-defined settings in Namespace object
        """
        super(VAE, self).__init__()

        self.flags = flags
        self.name = distribution
        self.epochs, self.num_restarts = 0, 0
        self.input_size, self.z_dim, self.distribution, self.device = input_size, z_dim, distribution, device
        self.encode_type, self.decode_type = encode_type, decode_type
        self.r = torch.tensor(1.)

        self.encoder, self.fc_mean, self.fc_var = create_encoder(input_size, input_type, [z_dim], h_dims,
                                                                 distribution, encode_type, flags)
        self.fc_mean, self.fc_var = self.fc_mean[0], self.fc_var[0]
        self.decoder = create_decoder(input_size, input_type, [z_dim], h_dims, decode_type)

    def encode(self, x):

        if self.encode_type == 'cnn':
            x = x.reshape(x.size(0), *self.input_size)

        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(h)
            z_var = self.fc_var(h)

        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean_unnormalized = self.fc_mean(h)
            z_mean = z_mean_unnormalized / z_mean_unnormalized.norm(dim=-1, keepdim=True)
            z_var = 1. + self.fc_var(h)  # the `+ 1` prevent collapsing behaviors
        else:
            raise NotImplemented

        return z_mean, z_var

    def decode(self, z):

        if self.decode_type == 'cnn':
            z = z.view(z.size(0), self.z_dim, 1, 1)

        x_recon = self.decoder(z)
        return x_recon.view(x_recon.size(0), -1)

    def reparameterize(self, z_mean, z_var):

        if self.distribution == 'normal':
            if self.flags.covariance_matrix == 'full':
                z_var = lower_triangular_matrix_from_vector(z_var, z_mean.size(1), z_mean.size(0))
                q_z = torch.distributions.MultivariateNormal(z_mean, scale_tril=z_var)
                p_z = torch.distributions.MultivariateNormal(
                    torch.zeros_like(z_mean, device=self.device),
                    torch.eye(z_mean.size(1), device=self.device).unsqueeze(0).repeat(z_mean.size(0), 1, 1))
            else:
                q_z = torch.distributions.normal.Normal(z_mean, z_var)
                p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))

        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var, self.r)
            p_z = HypersphericalUniform(self.z_dim - 1, self.r, device=self.device)
        else:
            raise NotImplemented

        return q_z, p_z

    def loss(self, q_z, p_z, x_mb, x_mb_recon):
        if self.flags.loss_function == 'bce':
            lf = nn.BCEWithLogitsLoss(reduction='none')
        elif self.flags.loss_function == 'mse':
            lf = nn.MSELoss(reduction='none')
        else:
            raise NotImplemented

        loss_recon = lf(x_mb_recon, x_mb.reshape(x_mb.size(0), -1)).sum(-1).mean()

        if self.distribution == 'normal':
            loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        elif self.distribution == 'vmf':
            loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        else:
            raise NotImplemented

        return loss_recon, loss_kl, None

    def log_likelihood(self, x, n=10):
        """
        :param x: e.g. MNIST data mini-batch
        :param n: number of MC samples
        :return: MC estimate of log-likelihood
        """

        (q_z, p_z), z, x_mb_recon = self.forward(x.reshape(x.size(0), -1), n=n)
        if self.flags.loss_function == 'bce':
            lf = nn.BCEWithLogitsLoss(reduction='none')
        elif self.flags.loss_function == 'mse':
            lf = nn.MSELoss(reduction='none')
        else:
            raise NotImplemented

        log_p_x_z = -lf(x_mb_recon, x.reshape(x.size(0), -1).repeat((n, 1, 1))).sum(-1)

        if self.distribution == 'normal':
            if self.flags.covariance_matrix == 'full':
                log_p_z = p_z.log_prob(z)
                log_q_z_x = q_z.log_prob(z)
            else:
                log_p_z = p_z.log_prob(z).sum(-1)
                log_q_z_x = q_z.log_prob(z).sum(-1)
        elif self.distribution == 'vmf':
            log_p_z = p_z.log_prob(z)
            log_q_z_x = q_z.log_prob(z)
        else:
            raise NotImplementedError

        return ((log_p_x_z + log_p_z.to(self.device) - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()

    def forward(self, x, n=None):

        z_mean, z_var = self.encode(x)

        if torch.isnan(z_mean).sum() > 0 or torch.isnan(z_var).sum() > 0:
            return (None, None), None, None

        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample(torch.Size() if n is None else torch.Size([n]))

        if n is None:
            x_recon = self.decode(z)
        else:
            x_recon = self.decode(z.reshape(n * x.size(0), -1))
            x_recon = x_recon.reshape(n, x.size(0), -1)

        return (q_z, p_z), z, x_recon


class ProductSpaceVAE(torch.nn.Module):

    def __init__(self, h_dims, z_dims, input_size=[1, 28, 28], input_type='binary', distribution='normal',
                 encode_type='mlp', decode_type='mlp', device='cpu', flags=None):
        """
        ModelVAE initializer
        :param h_dims: dimension of the hidden layers, list
        :param z_dims: dimensions of the latent representation, list
        :param input_size: dimensions of input
        :param input_type: (str) e.g binary
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        :param encode_type: (str) one of {mlp, cnn}
        :param decode_type: (str) one of {mlp, cnn}
        :param device: device to use
        :param flags: user-defined settings in Namespace object
        """
        super(ProductSpaceVAE, self).__init__()

        self.flags = flags
        self.name = 'productspace'
        self.epochs, self.num_restarts = 0, 0
        self.input_size, self.distribution, self.device = input_size, distribution, device
        self.encode_type, self.decode_type = encode_type, decode_type

        self.z_dims = np.sort(np.asarray(z_dims))
        self.z_unique, self.z_counts = np.unique(self.z_dims, return_counts=True)
        self.z_u_idx = [np.where(self.z_dims == u)[0] for u in self.z_unique]

        self.r = torch.ones(len(z_dims), device=device)

        self.encoder, self.fc_means, self.fc_vars = create_encoder(input_size, input_type, self.z_dims, h_dims,
                                                                   distribution, encode_type, flags)
        self.decoder = create_decoder(input_size, input_type, z_dims, h_dims, decode_type)

    def encode(self, x):
        if self.encode_type == 'cnn':
            x = x.reshape(x.size(0), *self.input_size)

        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        if self.distribution == 'normal':
            # compute means and stds of the normal distributions
            z_means = [f(h) for f in self.fc_means]
            z_vars = [f(h) for f in self.fc_vars]
        elif self.distribution == 'vmf':
            # compute means and concentrations of the von Mises-Fishers
            z_means_unnormalized = [f(h) for f in self.fc_means]
            z_means = [zmu / zmu.norm(dim=-1, keepdim=True) for zmu in z_means_unnormalized]
            z_vars = [(f(h) + 1.) for f in self.fc_vars]  # the `+ 1` prevents collapsing behaviors
        else:
            raise NotImplemented

        return z_means, z_vars

    def decode(self, z):
        if self.decode_type == 'cnn':
            z = z.view(z.size(0), sum(self.z_dims), 1, 1)

        x_recon = self.decoder(z)

        return x_recon.view(x_recon.size(0), -1)

    def reparameterize(self, z_means, z_vars):
        # since z is sorted, we take the min index, and the max index, to slice the list of z_means, z_vars
        # this is done to not have convert to numpy array
        gather_zvs = [(torch.cat(z_means[min(u_idx):max(u_idx) + 1], 0),
                       torch.cat(z_vars[min(u_idx):max(u_idx) + 1], 0))
                      for u_idx in self.z_u_idx]

        if self.distribution == 'normal':
            # for each pair of z_mean, z_var, we make a distribution (sampling) object
            q_zs_sample = [torch.distributions.normal.Normal(z_mean, z_var) for (z_mean, z_var) in gather_zvs]

            q_zs = [torch.distributions.normal.Normal(z_mean, z_var) for z_mean, z_var in zip(z_means, z_vars)]
            p_zs = [torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var)) for
                    z_mean, z_var in zip(z_means, z_vars)]

        elif self.distribution == 'vmf':
            # for each pair of z_mean, z_var, we make a distribution (sampling) object
            q_zs_sample = [VonMisesFisher(z_mean, z_var) for (z_mean, z_var) in gather_zvs]

            q_zs = [VonMisesFisher(z_mean, z_var) for z_mean, z_var in zip(z_means, z_vars)]
            p_zs = [HypersphericalUniform(z_dim - 1, device=self.device) for z_dim in self.z_dims]
        else:
            raise NotImplemented

        return q_zs, p_zs, q_zs_sample

    def loss(self, q_zs, p_zs, x_mb, x_mb_recon):

        if self.flags.loss_function == 'bce':
            lf = nn.BCEWithLogitsLoss(reduction='none')
        elif self.flags.loss_function == 'mse':
            lf = nn.MSELoss(reduction='none')
        else:
            raise NotImplemented
        loss_recon = lf(x_mb_recon, x_mb.reshape(x_mb.size(0), -1)).sum(-1).mean()

        if self.distribution == 'normal':
            loss_kl = torch.stack([torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1) for
                                   q_z, p_z in zip(q_zs, p_zs)], dim=-1).sum(-1).mean()
        elif self.distribution == 'vmf':
            loss_kl = torch.stack([torch.distributions.kl.kl_divergence(q_z, p_z) for
                                   q_z, p_z in zip(q_zs, p_zs)], dim=-1).sum(-1).mean()
        else:
            raise NotImplemented

        return loss_recon, loss_kl, None

    def log_likelihood(self, x, n=10):
        """
        :param x: e.g. MNIST data flattened
        :param n: number of MC samples
        :return: MC estimate of log-likelihood
        """

        z_means, z_vars = self.encode(x.reshape(x.size(0), -1))
        q_zs, p_zs, _, = self.reparameterize(z_means, z_vars)
        z_parts = [q_z.rsample(torch.Size([n])) for q_z in q_zs]
        z = torch.cat(z_parts, dim=-1).reshape(n*x.size(0), -1)

        x_mb_recon = self.decode(z)
        x_mb_recon = x_mb_recon.reshape(n, x.size(0), -1)

        if self.flags.loss_function == 'bce':
            lf = nn.BCEWithLogitsLoss(reduction='none')
        elif self.flags.loss_function == 'mse':
            lf = nn.MSELoss(reduction='none')
        else:
            raise NotImplemented
        log_p_x_z = -lf(x_mb_recon, x.reshape(x.size(0), -1).repeat((n, 1, 1))).sum(-1)

        if self.distribution == 'normal':
            log_p_z = torch.stack([p_z.log_prob(z__).sum(-1) for p_z, z__ in zip(p_zs, z_parts)], dim=-1).sum(-1)
            log_q_z_x = torch.stack([q_z.log_prob(z__).sum(-1) for q_z, z__ in zip(q_zs, z_parts)], dim=-1).sum(-1)
        elif self.distribution == 'vmf':
            log_p_z = torch.stack([p_z.log_prob(z__) for p_z, z__ in zip(p_zs, z_parts)], dim=-1).sum(-1)
            log_q_z_x = torch.stack([q_z.log_prob(z__) for q_z, z__ in zip(q_zs, z_parts)], dim=-1).sum(-1)
        else:
            raise NotImplementedError

        return ((log_p_x_z + log_p_z.to(self.device) - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()

    def forward(self, x):

        z_means, z_vars = self.encode(x)
        if torch.isnan(z_means[0]).sum() > 0 or torch.isnan(z_vars[0]).sum() > 0:
            return (None, None), None, None

        q_zs, p_zs, q_zs_sample = self.reparameterize(z_means, z_vars)

        # sample z1, z2, .., zk and concatenate
        # z = torch.cat([q_z.rsample(torch.Size() if n is None else torch.Size([n])) for q_z in q_zs], dim=-1)  # slow
        z = torch.cat([torch.cat(torch.chunk(q_z.rsample(), int(c), dim=0), dim=-1)
                       for q_z, c in zip(q_zs_sample, self.z_counts)], dim=-1)
        # z_parts = list(torch.split(z, tuple(self.z_unique.repeat(self.z_counts)), -1))
        x_recon = self.decode(z)

        return (q_zs, p_zs), z, x_recon
