import torch
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from hyperspherical_vae.distributions import HypersphericalUniform


def swap_part_random(zs, p_zs, idx):
    """Swap out parts to study disentanglement"""
    zs = zs.copy()
    dims = [zs[i].shape[0] for i in idx]
    zks = [p_z.sample(dim) for (p_z, i, dim) in zip(p_zs, idx, dims)]
    zs = np.asarray(zs)
    zs[idx] = zks
    z_ = torch.cat(list(zs), dim=-1)

    return z_, zs


def swap_part_inter(z, num_steps, i, j):
    """ Interpolate one of the z_k
    :param z: the original sample
    :param num_steps: number of interpolation steps
    :param i: start index
    :param j: end index"""
    num_steps = num_steps - 2

    zk = z[i:j]
    zk = interpolate_circle(zk, zk, steps=num_steps)
    z = z.repeat(num_steps, 1)
    z[:, i:j] = zk
    return z, zk


# https://stackoverflow.com/a/25628397/3723434
def get_cmap(n, name='Paired'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_knobs(zs, cmap=None, cmap_name='Paired', figsize=(10, 10), title='', s=100, y=None):
    """plots z_1 .. z_k coordinates assuming circles of a RV z"""

    fig = plt.figure(figsize=figsize)
    if cmap is None:
        cmap = get_cmap(10, name=cmap_name)
        cmap = [cmap(y_) for y_ in y]

    for i, z in enumerate(zs):
        knob = z.detach().cpu().numpy()
        # batch = knob.shape[0]
        ax = fig.add_subplot(len(zs), len(zs), i + 1)
        ax.scatter(knob[:, 0], knob[:, 1], c=cmap, s=np.full_like(cmap, s), label=y, cmap=cmap_name)
        plt.axis('equal')
        plt.axis('off')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        circle = plt.Circle((0, 0), 1., color='gray', alpha=0.1)
        plt.gcf().gca().add_artist(circle)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if y is not None:
        labels = [str(i) for i in np.unique(y)]
        idx_u = [np.where(y == i)[0][0] for i in np.unique(y)]
        colors = np.asarray(cmap)[idx_u]
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("s", colors[i]) for i in range(len(labels))]

        # Put a legend to the right of the current axis
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.axis('off')
    plt.show()
    return cmap


def interpolate_circle(pa, pb=None, steps=1, shortest=True):
    """ Interpolate between two points on a circle
    :param pa: point a = (xa, ya)
    :param pb: point b = (xb, yb)
    :param steps: number of interpolation steps
    :param shortest: take the shortest path or not"""

    pb = pa if pb is None else pb

    (xa, _), (xb, _) = pa, pb
    phi_a, phi_b = torch.acos(xa), torch.acos(xb)

    if phi_a > phi_b:
        temp = phi_a
        phi_a = phi_b
        phi_b = temp

    # get distance between two points, pa --- pb
    theta = phi_b - phi_a
    from_0 = 0.

    if not shortest and (theta < np.pi):
        from_0 = 2 * np.pi - phi_b + 1e-5
        phi_b = torch.tensor(0.)
        phi_a += from_0

    # interpolate entire circle on self
    if torch.eq(pa, pb).all():
        phi_a = phi_a - 2 * np.pi

    i = torch.linspace(0., 1., steps=2 + steps, device=pa.device)
    phi = (phi_a * (1. - i) + phi_b * i) - from_0

    return torch.stack([torch.cos(phi), -torch.sin(phi)], -1)


def plot_canvas(x, rows=2, title=None, size=None):
    x_base = torch.cat(torch.chunk(x, x.size(0), dim=0), dim=1).squeeze()
    x_base = torch.cat(torch.chunk(x_base, x.size(0), dim=0), dim=1).squeeze()

    x_canvas = torch.cat(torch.chunk(x_base, rows, dim=1), dim=0).squeeze()
    x_canvas = expit(x_canvas.detach().cpu().numpy())

    f = plt.figure(figsize=((20, 6) if size is None else size))
    plt.imshow(x_canvas, interpolation='none', cmap='gray')
    plt.axis('off')
    if title is not None:
        plt.title(title)

    return f


def plot_2d_interpolation(model, num_steps=100, radius=1., point=None,
                          rows=1, title=None, size=None, save=None):
    if model.z_dim > 2:
        print('for 2D embedding')
        return
    if point is None:
        u = HypersphericalUniform(model.z_dim - 1)
        point = u.sample()

    num_steps = num_steps - 2
    inter = interpolate_circle(point, steps=num_steps) * radius

    x_ = model.decode(inter.to(model.device))
    x_ = x_.reshape(-1, *model.input_size).squeeze()

    f = plot_canvas(x_, rows=rows, title=title, size=size)
    if save is not None:
        f.savefig(save + "_interpolation.pdf", bbox_inches='tight')

    return point


def plot_square(model, points=10, buffer=1., fname=None, format='eps'):
    if model.z_dim > 2:
        print('for 2D embedding')
        return

    max_r = max(model.r_classes) + buffer
    x = np.linspace(-max_r, max_r, points).astype(np.int)
    y = np.linspace(-max_r, max_r, points).astype(np.int)
    X, Y = np.meshgrid(x, y)
    Z = np.hstack((X.reshape(-1, 1), np.flip(Y.reshape(-1, 1))))

    x_ = model.decode(torch.tensor(Z).type(torch.float).to(model.device))
    x_ = x_.reshape(-1, *model.input_size).squeeze()

    f = plot_canvas(x_, rows=points, title=None, size=(20, 20))

    if fname is not None:
        f.savefig(fname, bbox_inches='tight', format=format, dpi=500)


def hammer_projection(model, z, y):
    if model.z_dim != 3:
        print('for 3D embedding')
        return

    z_ = z.detach().cpu().numpy()
    z_k = np.linalg.norm(z_, axis=-1).round(1)
    ks, k_ = np.unique(z_k, return_counts=True)
    idx_k = [np.where(z_k == k)[0] for k in ks]

    # ys = [y[k] for k in idx_k]

    def plot_hammer(z, y):
        z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]

        R = np.linalg.norm(z[0], axis=-1).round(1)
        phi = np.arcsin(z3 / R)
        lamb = np.arctan2(z2, z1)

        z_ = np.sqrt(1 + np.cos(phi) * np.cos(lamb / 2))
        x_ = np.cos(phi) * np.sin(lamb / 2) / z_
        y_ = np.sin(phi) / z_

        cmap = get_cmap(10)
        cmap = [cmap(y_) for y_ in y]

        _ = plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        ax.scatter(x_, y_, c=cmap, s=2)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        if y is not None:
            labels = [str(i) for i in np.unique(y)]
            idx_u = [np.where(y == i)[0][0] for i in np.unique(y)]
            colors = np.asarray(cmap)[idx_u]
            f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
            handles = [f("s", colors[i]) for i in range(len(labels))]

            # Put a legend to the right of the current axis
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.axis('off')
        plt.title('Hammer Projection of radius: %.2f' % R)
        plt.show()

    [plot_hammer(z_[k], y[k]) for k in idx_k]


def plot_reconstructions(model, loader, num_recons=6, rows=1, x_mb=None, idx=None, show_title=True,
                         save_o=None, save_r=None):
    if x_mb is None:
        print('none')
        x_mb = next(iter(loader))[0]
        if model.flags.dynamic_binarization:
            x_mb = (x_mb.to(model.device) > torch.distributions.Uniform(0, 1).sample(x_mb.shape).to(
                model.device)).float()
        idx = np.random.randint(0, loader.batch_size, num_recons)

    (_, _), z, x_recon = model(x_mb.reshape(x_mb.size(0), -1))

    originals = x_mb[idx]
    recons = x_recon[idx].reshape(-1, *model.input_size).squeeze()

    f1 = plot_canvas(originals, rows=rows, title='Original' if show_title else None)
    f2 = plot_canvas(recons, rows=rows, title='Reconstructions' if show_title else None)

    if save_o is not None:
        f1.savefig(save_o + "_original_recon.png", bbox_inches='tight')
    if save_r is not None:
        f2.savefig(save_r + "_recon_recon.png", bbox_inches='tight')


def plot_samples(model, num_samples=6, rows=1, title='', save=None):
    if model.name == 'productspace':
        _p_zs = [HypersphericalUniform(z_dim - 1, device=model.device) for z_dim in model.z_dims]
        z = torch.cat([p_z.sample(num_samples) for p_z in _p_zs], dim=-1)
    else:
        pz = HypersphericalUniform(model.z - 1, device=model.device)
        z = pz.sample(num_samples)

    x_ = model.decode(z.to(model.device))

    f = plot_canvas(x_.reshape(-1, *model.input_size).squeeze(), rows=rows, title=title)

    plt.tight_layout()
    plt.show()
    if save is not None:
        f.savefig(save + "_samples.png", bbox_inches='tight')


def get_embeddings(model, dataloader):
    """
    Get latent embeddings of trained model
    :param dataloader: dataLoader object
    :param model: trained VAE model
    :return: z_, latent embeddings, y_, labels of embeddings
    """
    z_, y_ = None, None
    for x_mb, y_mb in dataloader:

        x_mb = x_mb.to(model.device)
        # dynamic binarization
        if model.flags.dynamic_binarization:
            x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape).to(model.device)).float()

        if model.name in ['productspace', 'vmf', 'normal']:
            (_, _), z, _ = model(x_mb.reshape(x_mb.size(0), -1).to(model.device))
        else:
            raise NotImplementedError

        z_ = z if z_ is None else torch.cat((z_, z), 0)
        y_ = y_mb if y_ is None else torch.cat((y_, y_mb), 0)

    return z_, y_
