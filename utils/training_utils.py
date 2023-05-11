import numpy as np
import torch
import torch.utils.data
from collections import defaultdict
from time import time
from argparse import Namespace

from models import VAE, ProductSpaceVAE


def eval_train(model, optimizer, e, train_loader, flags=None):
    """
    Evaluate PyTorch model on train data set
    :param model: PyTorch model
    :param optimizer: optimizer
    :param e: epoch number
    :param train_loader: PyTorch evaluation data set loadern
    :param flags: user-defined settings in Namespace object
    :return: loss, reconstruction loss, kl loss
    """
    temp_losses, temp_recon_losses, temp_kl_losses, temp_kl_cat, temp_qr = [], [], [], [], []
    msg_len = 0
    for i, (x_mb, _) in enumerate(train_loader):
        optimizer.zero_grad()

        x_mb, loss, loss_recon, loss_kl, optional = eval_batch(model, x_mb, e, flags=flags)
        if loss is None:
            return None, None, None

        loss.backward()
        optimizer.step()

        temp_losses.append(loss_recon.detach().cpu().numpy() + loss_kl.detach().cpu().numpy())
        temp_recon_losses.append(loss_recon.detach().cpu().numpy())
        temp_kl_losses.append(loss_kl.detach().cpu().numpy())

        msg = f'\r\t {i / (len(train_loader.dataset) // train_loader.batch_size): .2%} \t loss: {temp_losses[-1]: .3f}'
        print(' ' * msg_len, end='\r')
        print(msg, end='\r')
        msg_len = len(msg)

    return np.mean(temp_losses), np.mean(temp_recon_losses), np.mean(temp_kl_losses)


def eval_val(model, e, val_loader, n_sample=10, flags=None):
    """
    Evaluate PyTorch model on valuation data set
    :param model: PyTorch model
    :param e: epoch number
    :param val_loader: PyTorch evaluation data set loader
    :param n_sample: number of samples to use for evaluation
    :param flags: user-defined settings in Namespace object
    :return: loss, reconstruction loss, kl loss, (optional) log likelihood
    """
    with torch.no_grad():
        temp_losses, temp_recon_losses, temp_kl_losses, temp_ll, temp_kl_cat, temp_qr = [], [], [], [], [], []
        msg_len = 0
        for i, (x_mb, _) in enumerate(val_loader):

            x_mb, loss, loss_recon, loss_kl, optional = eval_batch(model, x_mb, e, flags=flags)
            if loss is None:
                return None, None, None, None

            temp_losses.append(loss_recon.detach().cpu().numpy() + loss_kl.detach().cpu().numpy())
            temp_recon_losses.append(loss_recon.detach().cpu().numpy())
            temp_kl_losses.append(loss_kl.detach().cpu().numpy())

            if flags.val_nll:
                ll = model.log_likelihood(x_mb, n_sample).data
                if not torch.isnan(ll).sum() > 0:
                    temp_ll.append(float(ll))

            msg = f'\t {i / (len(val_loader.dataset) // val_loader.batch_size): .2%} \t loss: {temp_losses[-1]: .3f}'
            print(' ' * msg_len, end='\r')
            print(msg, end='\r')
            msg_len = len(msg)

    return np.mean(temp_losses), np.mean(temp_recon_losses), np.mean(temp_kl_losses), (
        np.mean(temp_ll) if flags.val_nll else [])


def eval_batch(model, x_mb, e, flags=None):
    """
    Evaluate PyTorch model on a batch
    :param model: PyTorch model
    :param x_mb: batch of data points
    :param e: epoch number
    :param flags: user-defined settings in Namespace object
    :return: x_mb, loss, reconstruction loss, kl loss
    """
    x_mb = x_mb.to(model.device)
    # dynamic binarization
    if flags.dynamic_binarization:
        x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape).to(model.device)).float()

    kl_weighted = 0.

    (q_z, p_z), z, x_mb_recon = model(x_mb.reshape(x_mb.size(0), -1))
    if z is None:
        return None, None, None, None, None
    loss_recon, loss_kl, _ = model.loss(q_z, p_z, x_mb, x_mb_recon)

    kl_weighted += (0 if flags.kl_freeze > e else 1.) * torch.abs(
        (max(0, min(flags.kl_warmup, e - flags.kl_freeze)) / flags.kl_warmup) * flags.beta * loss_kl - flags.kl_target)
    loss = loss_recon + kl_weighted

    return x_mb, loss, loss_recon, loss_kl, None


def train(model, optimizer, train_loader=None, val_loader=None, flags=None, path=None):
    """
    Train model for number of epochs (training flags defined in the Namespace 'flags' object, see 'run_models.py')
    :param model: VAE model
    :param optimizer: optimizer
    :param train_loader: training data loader
    :param val_loader: validation data loader
    :param flags: flags used to initialize model
    :param path: path to save models
    :return: loss, statistics dictionary
    """
    assert train_loader is not None

    # saving lists
    train_losses, train_kl_losses, train_recon_losses = [], [], []
    val_losses, val_kl_losses, val_recon_losses, val_ll = [], [], [], []
    loss, val_loss = 0, 0
    last_save = -1
    best_loss, best_epoch = 1e10, 0
    continue_training = True
    num_restarts = 0

    e_offset = model.epochs
    for e in range(e_offset, e_offset + flags.epochs):
        if not continue_training:
            break

        model.epochs += 1  # keep track of the amount of epochs this model was trained for
        t = time()

        loss, recon_loss, kl_loss = eval_train(model, optimizer, e, train_loader, flags=flags)

        if loss is None:
            recovery = train_recover_nan_loss(model, flags.max_restarts, num_restarts, last_save, path)
            if recovery[0] is None:
                return recovery
            else:
                model, optimizer, epoch, loss = recovery
        else:
            train_losses.append(loss)
            train_recon_losses.append(recon_loss)
            train_kl_losses.append(kl_loss)

        if (e % flags.pf) == 0:
            msg = ('\r(train) e: %d (%.2f) \t loss: %.3f \t recon: %.3f \t kl: %.3f'
                   % (e, time() - t,
                      float(train_losses[-1]), float(train_recon_losses[-1]), float(train_kl_losses[-1])))
            print(msg + '\n', end='')

        if ((e + 1) % flags.val_f) == 0:
            val_loss, val_recon, val_kl, val_ll_ = eval_val(model, e, val_loader, flags=flags)
            if val_loss is None:
                recovery = train_recover_nan_loss(model, flags.max_restarts, num_restarts, last_save, path)
                if recovery[0] is None:
                    return recovery
                else:
                    model, optimizer, epoch, loss = recovery

            else:
                val_losses.append(val_loss)
                val_recon_losses.append(val_recon)
                val_kl_losses.append(val_kl)
                val_ll.append(val_ll_)

            msg = ('\r(val): \t loss: %.3f \t recon: %.3f \t kl: %.3f'
                   % (float(val_losses[-1]), float(val_recon_losses[-1]), float(val_kl_losses[-1])))
            if flags.val_nll:
                msg += ('\t nll: %.3f' % float(val_ll[-1]))
            print(msg + '\n\n', end='')

        # save model every X epochs to be saved
        if ((e + 1) % flags.save_f) == 0:
            last_save = e
            create_checkpoint(path, epoch=e, model=model, optimizer=optimizer, loss=loss, flags=flags)

        if (e + 1) > flags.burn_in:
            if best_loss > (val_loss + flags.min_improv):
                print('\nupdate best\n')
                best_loss, best_epoch = val_loss, e
                path_ = path + '_best'
                create_checkpoint(path_, epoch=e, model=model, optimizer=optimizer, loss=loss, flags=flags)

            if (e - best_epoch) > flags.look_ahead:
                print('\n finish training \n')
                continue_training = False

    stats = {'train_loss': train_losses, 'train_kl_loss': train_kl_losses, 'train_recon_loss': train_recon_losses,
             'val_loss': val_losses, 'val_kl_loss': val_kl_losses, 'val_recon_loss': val_recon_losses, 'val_ll': val_ll}

    return loss, stats


def train_recover_nan_loss(model, max_restarts, num_restarts, last_save, path):
    """
    safety mechanism to restart from checkpoint on NAN
    :param model: PyTorch model
    :param max_restarts: (int) max restarts
    :param num_restarts: (int) number of restarts passed
    :param last_save: (int) last saving epoch
    :param path: (str) path to load model from
    :return: model, optimizer, epoch number, loss
    """
    er_return = None, None
    if (max_restarts < num_restarts) or (last_save < 0):
        print('no saved versions' if last_save < 0 else 'max restarts reached, terminating training')
        return er_return
    else:
        num_restarts += 1
        model.num_restarts = max(model.num_restarts + 1, num_restarts)
        print('\n restarting model for %d time\n' % num_restarts)
        try:
            model, optimizer, epoch, loss, _ = load_from_checkpoint(path + '.tar')
        except FileNotFoundError as er:
            print(f'error: unable to load model from checkpoint {er}')
            return er_return

    return model, optimizer, epoch, loss


def test(model, n_sample, test_loader=None):
    """
    Evaluate model on test data set
    :param model: PyTorch model
    :param n_sample: number of samples to evaluate on
    :param test_loader: PyTorch test data set loader
    :return: statistics dictionary
    """
    assert test_loader is not None, print('must provide test data')
    print_ = defaultdict(list)
    with torch.no_grad():
        for x_mb, y_mb in test_loader:

            x_mb = x_mb.to(model.device)

            # dynamic binarization
            if model.flags.dynamic_binarization:
                x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape).to(model.device)).float()

            (q_z, p_z), z, x_mb_recon = model(x_mb.reshape(x_mb.size(0), -1))
            loss_recon, loss_kl, _ = model.loss(q_z, p_z, x_mb, x_mb_recon)

            print_['test_recon_loss'].append(float(loss_recon.data))
            print_['test_kl_loss'].append(float(loss_kl.data))
            print_['test_loss'].append(- print_['test_recon_loss'][-1] - print_['test_kl_loss'][-1])

            if model.name in ['vmf', 'normal', 'productspace']:
                ll = model.log_likelihood(x_mb, n_sample).data
                if not (torch.isnan(ll).sum() > 0):
                    print_['LL'].append(float(ll))

    return {k: np.mean(v) for k, v in print_.items()}


# https://pytorch.org/tutorials/beginner/saving_loading_models.html
def create_checkpoint(path, epoch, model, optimizer, loss, flags):
    """
    Create a checkpoint to load model from
    :param path: path to save checkpoint
    :param epoch: current epoch
    :param model: trained model
    :param optimizer: optimizer used to train model
    :param loss: current loss
    :param flags: Namespace object with experiment flags
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'exp_flags': vars(flags),
    }, path + '.tar')


def load_from_checkpoint(path):
    """
    Load saved model from checkpoint
    :param path: path to load model from
    :return: model, optimizer, epoch number, loss, flags
    """
    print('\n load checkpoint from path: %s \n' % path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    flags = Namespace(**checkpoint['exp_flags'])
    model = init_model(flags, device)

    optimizer = torch.optim.Adam(list(model.parameters()) + ([model.r]
                                                             if (flags.name == 'vmf' and flags.distribution == 'vmf')
                                                             else []), lr=flags.lr)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss, flags


def init_model(flags, device):
    """
    :param flags: Namespace with user defined experiment flags
    :param device: cuda or cpu
    :return: model
    """
    h_dims = [int(h) for h in flags.h_dims.split(",")]

    if flags.name == 'vmf' or flags.name == 'normal':
        z = flags.z if (flags.distribution == 'normal') else flags.z + 1
        model = VAE(input_size=flags.input_size, input_type=flags.input_type,
                    encode_type=flags.encode_type, decode_type=flags.decode_type,
                    h_dims=h_dims, z_dim=z, distribution=flags.distribution,
                    device=device, flags=flags).to(device)
    elif flags.name == 'productspace':
        z = [int(z) + (1 if flags.distribution == 'vmf' else 0) for z in flags.z_dims.split(",")]
        model = ProductSpaceVAE(input_size=flags.input_size, input_type=flags.input_type,
                                encode_type=flags.encode_type, decode_type=flags.decode_type,
                                h_dims=h_dims, z_dims=z, distribution=flags.distribution,
                                device=device, flags=flags).to(device)
    else:
        raise NotImplemented

    return model
