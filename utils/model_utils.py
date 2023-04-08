import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


class MLP(nn.Module):

    def __init__(self, h_dims=None, activation=nn.ReLU, end_activation=False, batch_norm=False):
        """
        Helper module to create MLPs
        :param h_dims: number of layers with hidden units specified, list
        :param activation: activation function to be used on layer outputs
        :param end_activation: (optional) final activation function, for example a softmax or sigmoid
        """
        super(MLP, self).__init__()

        self.h_dims, self.activation = h_dims, activation

        modules = []
        for input_dim, output_dim in zip(h_dims, h_dims[1:-1]):
            modules.append(nn.Linear(input_dim, output_dim))
            modules.append(self.activation())
            if batch_norm:
                modules.append(nn.BatchNorm1d(output_dim))
        modules.append(nn.Linear(h_dims[-2], h_dims[-1]))

        if end_activation:
            modules.append(self.activation())
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


def lower_triangular_vector(z):
    """
    Get number of non-zero indices of lower-triangular matrix
    :param z: dimensionality of [z x z] matrix
    :return:
    """
    return np.arange(z).cumsum()[-1] + z


def lower_triangular_matrix_from_vector(v, z, batch, idx=None):
    """
    Create lower triangular matrix from vector
    :param v: batched vector with index content, [batch, tril_values]
    :param z: dimensionality of matrix, [z x z]
    :param batch: batch size
    :param idx: (optional) precomputed indices for lower triangular
    :return: [batch, z, z] lower triangular matrix
    """
    tril = torch.zeros((batch, z, z), device=v.device)
    if idx is None:
        idx = tuple((np.arange(batch)[:, np.newaxis], *np.tril_indices(z, 0)))
    tril[idx] = v

    return tril


def create_encoder(input_size, input_type, z_dims, h_dims, distribution='vmf', encode_type='mlp', flags=None):
    """
    Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
    the encoder expects data as input of shape (batch_size, num_channels, width, height).
    """

    last_kernel_size = 7
    if input_size == [1, 28, 28] or input_size == [3, 28, 28] or input_size == [1, 10, 10]:
        last_kernel_size = 7
    elif input_size == [1, 28, 20]:
        last_kernel_size = (7, 5)
    else:
        pass
        # raise ValueError('invalid input size!!')

    h_last = 256 if encode_type == 'cnn' else h_dims[-1]

    if input_type == 'binary':
        if encode_type == 'cnn':
            q_z_nn = nn.Sequential(
                GatedConv2d(input_size[0], 32, 5, 1, 2),
                GatedConv2d(32, 32, 5, 2, 2),
                GatedConv2d(32, 64, 5, 1, 2),
                GatedConv2d(64, 64, 5, 2, 2),
                GatedConv2d(64, 64, 5, 1, 2),
                GatedConv2d(64, 256, last_kernel_size, 1, 0),

            )
        elif encode_type == 'mlp':
            q_z_nn = MLP([input_size[-2]*input_size[-1]] + h_dims, end_activation=True)
        else:
            raise NotImplementedError

        q_z_means = nn.ModuleList([nn.Linear(h_last, z_dim) for z_dim in z_dims])
        q_z_vars = nn.ModuleList([nn.Sequential(nn.Linear(h_last, (
            1 if (distribution == 'vmf' or (flags.distribution == 'normal' and flags.covariance_matrix == 'single'))
            else (lower_triangular_vector(z_dim) if (flags.distribution == 'normal' and
                                                     flags.covariance_matrix == 'full') else z_dim))),
                                                nn.Softplus()) for z_dim in z_dims])

        return q_z_nn, q_z_means, q_z_vars

    elif input_type == 'multinomial':
        act = None

        if encode_type == 'cnn':
            q_z_nn = nn.Sequential(
                GatedConv2d(input_size[0], 32, 5, 1, 2, activation=act),
                GatedConv2d(32, 32, 5, 2, 2, activation=act),
                GatedConv2d(32, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 64, 5, 2, 2, activation=act),
                GatedConv2d(64, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 256, last_kernel_size, 1, 0, activation=act)
            )
        elif encode_type == 'mlp':
            q_z_nn = MLP([input_size[-2]*input_size[-1]] + h_dims, end_activation=True)
        else:
            raise NotImplementedError

        q_z_means = nn.ModuleList([nn.Linear(h_last, z_dim) for z_dim in z_dims])
        q_z_vars = nn.ModuleList([nn.Sequential(nn.Linear(h_last, (1 if distribution == 'vmf' else z_dim)),
                                                nn.Softplus(), nn.Hardtanh(min_val=0.01, max_val=7.))
                                  for z_dim in z_dims])

        return q_z_nn, q_z_means, q_z_vars


def create_decoder(input_size, input_type, z_dims, h_dims, decode_type='mlp'):
    """
    Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
    """

    last_kernel_size = 7
    if input_size == [1, 28, 28] or input_size == [3, 28, 28] or input_size == [1, 10, 10]:
        last_kernel_size = 7
    elif input_size == [1, 28, 20]:
        last_kernel_size = (7, 5)
    else:
        pass
        # raise ValueError('invalid input size!!')

    num_classes = 256

    if input_type == 'binary':
        if decode_type == 'cnn':
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(sum(z_dims), 64, last_kernel_size, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2)
            )

            p_x_mean = nn.Sequential(
                nn.Conv2d(32, input_size[0], 1, 1, 0),
                # nn.Sigmoid()
            )

            p_x_nn = nn.Sequential(p_x_nn, p_x_mean)

        elif decode_type == 'mlp':
            p_x_nn = MLP([sum(z_dims)] + h_dims[::-1] + [input_size[-2] * input_size[-1]])
        else:
            raise NotImplementedError

        return p_x_nn

    elif input_type == 'multinomial':
        act = None

        if decode_type == 'cnn':
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(sum(z_dims), 64, last_kernel_size, 1, 0, activation=act),
                GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act)
            )

            p_x_mean = nn.Sequential(
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, input_size[0] * num_classes, 1, 1, 0),
                # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
            )

            p_x_nn = nn.Sequential(p_x_nn, p_x_mean)
        elif decode_type == 'mlp':
            # only used now for synthetic data .. should be rewritten!
            p_x_nn = MLP([sum(z_dims)] + h_dims[::-1] + [input_size[-2] * input_size[-1]])
            # raise NotImplementedError
        else:
            raise NotImplementedError

        return p_x_nn

    else:
        raise ValueError('invalid input type!!')


class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


class GatedConvTranspose2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding=0, dilation=1,
                 activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g
