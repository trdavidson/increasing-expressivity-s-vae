import torch
import numpy as np
import scipy
import scipy.special
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
from argparse import Namespace
from hyperspherical_vae.distributions import HypersphericalUniform


# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters, clamp=False, v=0):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    msg = ''
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.mean())
            max_grads.append(p.grad.max())
            if ('encoder' in n) or ('vars' in n) or ('mean' in n):
                msg += ('(grad) %s: min(%f), max(%f), mean(%f)\n' %
                        (n, p.grad.min(), p.grad.max(), p.grad.mean()))
                msg += ('(data) %s: min(%f), max(%f), mean(%f)\n' %
                        (n, p.data.min(), p.data.max(), p.data.mean()))
                if clamp:
                    p.grad.data.clamp_(-1., 1.)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.02, top=0.05)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'],
               loc='lower right')
    plt.show()
    if v > 0:
        print()
        print(msg)
        print()


def plot_sphere(z, y):
    r = np.linalg.norm(z[0], axis=-1).round(1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=z[:, 0], ys=z[:, 1], zs=z[:, 2], c=y)
    plt.title('Sphere of radius %.1f, with %d points' % (r, z.shape[0]))
    plt.show()


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
def plot_sample(x, batch=1, num_rows=1, num_cols=1, colors=None, save=None,
                title='', figsize=None, correct=True, in_dim_sqrt=28, r_title=False):
    fig = plt.figure(figsize=figsize if figsize is not None else (num_cols * 1.5, num_rows * 8))

    for i in range(0, batch):
        # equivalent but more general
        ax = fig.add_subplot(batch, num_cols, i + 1, frameon=False)
        ax.imshow(scipy.special.expit(x.detach().cpu().numpy()[i, :]).reshape(in_dim_sqrt, in_dim_sqrt),
                  interpolation='none', cmap='gray')

        if colors is not None:
            ax.text(-6., 4.5, '  ', bbox={'facecolor': colors[i], 'pad': 5})
        if r_title:
            ax.set_title('r: %.1f' % (np.linalg.norm(x.detach().cpu().numpy()[i])))
        ax.axis('off')

    fig.suptitle(title)
    if correct:
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    if save is not None:
        fig.savefig(fname, bbox_inches='tight', format='eps', dpi=500)


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


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
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


def get_best_results(folder=None, criteria=None, min_date=None, max_date=None, v=0, ignore_temp=True):
    """
    Get best result models and stats

    :param folder: path to folder with results
    :param criteria: filter criteria of subfolders
    :param min_date: minimal date of experiment, e.g. 2019-05-01
    :param max_date: maximum date of experiment, e.g. 2019-06-01
    :param v: verbosity level
    :param ignore_temp: ignore results in temp folder
    :return: best saved models and best model stats lists
    """
    assert folder is not None, print("folder can not be NONE")

    subdirs = [d for d in os.listdir(folder) if os.path.isdir(folder + d)]
    select_dirs = [d for d in subdirs if d in criteria] if criteria is not None else subdirs
    if ignore_temp:
        select_dirs = [d for d in select_dirs if d != 'temp']

    log_paths = [folder + sd + '/' for sd in select_dirs]

    exps = []
    for log_path in log_paths:
        e = os.listdir(log_path)
        for e_ in e:
            exps.append(log_path + e_)

    tars, stats, fnames = [], [], []
    for f in exps:
        # only take the filename
        f_ = os.path.basename(os.path.normpath(f))
        # strip all non-date information
        f_ = re.sub(r'_best*.*|.tar', r'', f_)
        # convert to datetime
        f_date = datetime.strptime(f_, '%Y-%m-%dT%H:%M:%S.%f')
        # check if file was produced within range
        in_range = (True if min_date is None else f_date > datetime.strptime(min_date, '%Y-%m-%d'))
        in_range = in_range and (True if max_date is None else f_date < datetime.strptime(max_date, '%Y-%m-%d'))
        if in_range:
            if '.tar' in f and 'best' in f:
                tars.append(f)
                fnames.append(f)
            if 'stats' in f and 'best' in f:
                stats.append(f)

    # don't use .index() because don't want to deal with not-found issues
    matched = []
    for s in stats:
        s_ = os.path.basename(os.path.normpath(s)).replace('_stats.npy', '')
        for i, t in enumerate(tars):
            t_ = os.path.basename(os.path.normpath(t)).replace('.tar', '')
            if s_ == t_:
                matched.append((t, s))

    if v > 0:
        print('total results: ', len(matched))
        print('unmatched tars: ', (max(len(stats), len(tars)) - len(matched)))

    return [x[0] for x in matched], [x[1] for x in matched]


def plot_canvas(x, rows=2, title=None, size=None):
    x_base = torch.cat(torch.chunk(x, x.size(0), dim=0), dim=1).squeeze()
    x_base = torch.cat(torch.chunk(x_base, x.size(0), dim=0), dim=1).squeeze()

    x_canvas = torch.cat(torch.chunk(x_base, rows, dim=1), dim=0).squeeze()
    x_canvas = scipy.special.expit(x_canvas.detach().cpu().numpy())

    f = plt.figure(figsize=((20, 6) if size is None else size))
    plt.imshow(x_canvas, interpolation='none', cmap='gray')
    plt.axis('off')
    if title is not None:
        plt.title(title)

    return f


def plot_2d_embeddings(model, z, y, show_legend=True, show_axis=True, show_circle=True, fname=None, format='eps'):
    if model.z_dim > 2:
        print('for 2D embedding')
        return

    fig = plt.figure(figsize=(10, 10))
    lim = model.r_classes.max() + 1.

    cmap = get_cmap(10)
    cmap = [cmap(y_) for y_ in y]

    plt.scatter(x=z[:, 0].detach(), y=z[:, 1].detach(), c=cmap, s=np.full_like(cmap, 4), cmap='Paired')
    plt.axis('equal')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    if show_circle:
        circle = plt.Circle((0, 0), lim - 0.5, color='gray', alpha=0.05)
        plt.gcf().gca().add_artist(circle)

    if not show_axis:
        plt.axis('off')

    if y is not None and show_legend:
        labels = [str(i) for i in np.unique(y)]
        idx_u = [np.where(y == i)[0][0] for i in np.unique(y)]
        colors = np.asarray(cmap)[idx_u]
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("s", colors[i]) for i in range(len(labels))]
        plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)
    plt.show()

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', format=format, dpi=500)


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


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
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


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
def plot_spheres(model, z, y):
    if model.z_dim != 3:
        print('for 3D embedding')
        return

    z_ = z.detach().cpu().numpy()
    z_k = np.linalg.norm(z_, axis=-1).round(1)
    ks, k_ = np.unique(z_k, return_counts=True)
    idx_k = [np.where(z_k == k)[0] for k in ks]

    # ys = [y[k] for k in idx_k]

    def plot_sphere(z, y):
        r = np.linalg.norm(z[0], axis=-1).round(1)
        cmap = get_cmap(10)
        cmap = [cmap(y_) for y_ in y]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs=z[:, 0], ys=z[:, 1], zs=z[:, 2], c=cmap, s=2)
        plt.title('Sphere of radius %.1f, with %d points' % (r, z.shape[0]))

        if y is not None:
            labels = [str(i) for i in np.unique(y)]
            idx_u = [np.where(y == i)[0][0] for i in np.unique(y)]
            colors = np.asarray(cmap)[idx_u]
            f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
            handles = [f("s", colors[i]) for i in range(len(labels))]
            plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)
        plt.show()

    [plot_sphere(z_[k], y[k]) for k in idx_k]


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
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


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
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


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
def plot_samples(model, num_samples=6, rows=1, title='', save=None):
    if model.name == 'productspace':
        _p_zs = [HypersphericalUniform(z_dim - 1, device=model.device) for z_dim in model.z_dims]
        z = torch.cat([p_z.sample(num_samples) for p_z in _p_zs], dim=-1)
    else:
        pz = HypersphericalUniform(z_d - 1, device=model.device)
        z = pz.sample(num_samples)

    x_ = model.decode(z.to(model.device))

    f = plot_canvas(x_.reshape(-1, *model.input_size).squeeze(), rows=rows, title=title)

    plt.tight_layout()
    plt.show()
    if save is not None:
        f.savefig(save + "_samples.png", bbox_inches='tight')


def plot_circle_embeddings(z, y, fs=(5, 5), lim=0, title='', cmap_name='Paired',
                           show_circle=True, save_f=None, show_axes=True):
    if z.shape[1] > 2:
        print('for 2D embedding')
        return

    fig = plt.figure(figsize=fs)
    lim = lim if lim > 0 else (np.linalg.norm(z, axis=-1).max() + 1.)

    plt.scatter(x=z[:, 0], y=z[:, 1], c=y, s=4, cmap=cmap_name)
    plt.axis('equal')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    if show_circle:
        circle = plt.Circle((0, 0), lim - 0.5, color='gray', alpha=0.05)
        plt.gcf().gca().add_artist(circle)
    if not show_axes:
        plt.axis('off')
    plt.title(title)
    plt.show()
    if save_f is not None:
        fig.savefig(save_f + ".eps", bbox_inches='tight', format='eps', dpi=1000)
    plt.close()
