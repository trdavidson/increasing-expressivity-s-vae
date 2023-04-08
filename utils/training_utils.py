import numpy as np
import torch
import torch.utils.data
from collections import defaultdict
from time import time
from argparse import Namespace
import math

from models import VAE, ProductSpaceVAE


# https://gist.github.com/pimdh/eae45b6af5d75464fed38f398dca9967
class TensorLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.i = 0
        self.idxs = None

    def __iter__(self):
        self.i = 0
        self.idxs = self.indices()
        return self

    def indices(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        return idxs

    def __next__(self):
        if self.i >= len(self.dataset):
            raise StopIteration()
        batch = self.dataset[self.idxs[self.i: self.i + self.batch_size]]
        self.i += self.batch_size
        return batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def eval_train(model, optimizer, e, train_loader, flags=None):
    temp_losses, temp_recon_losses, temp_kl_losses, temp_kl_cat, temp_qr = [], [], [], [], []
    msg_len = 0
    for i, (x_mb, _) in enumerate(train_loader):
        optimizer.zero_grad()

        x_mb, loss, loss_recon, loss_kl, optional = eval_batch(model, x_mb, e, flags=flags)
        if loss is None:
            return None, None, None, None, None

        loss.backward()
        optimizer.step()

        temp_losses.append(loss_recon.detach().cpu().numpy() + loss_kl.detach().cpu().numpy())
        temp_recon_losses.append(loss_recon.detach().cpu().numpy())
        temp_kl_losses.append(loss_kl.detach().cpu().numpy())

        msg = ('\r\t %d \t loss: %.3f' % (len(train_loader.dataset) // train_loader.batch_size - i, temp_losses[-1]))
        if model.name == 'vmf':
            msg += ('\t r: %.3f' % model.r.data.detach().cpu().numpy())

        print(' ' * msg_len, end='\r')
        print(msg, end='\r')
        msg_len = len(msg)
        # print(msg, end='')

    return np.mean(temp_losses), np.mean(temp_recon_losses), np.mean(temp_kl_losses)


def eval_val(model, e, val_loader, n_sample=10, flags=None):
    with torch.no_grad():
        temp_losses, temp_recon_losses, temp_kl_losses, temp_ll, temp_kl_cat, temp_qr = [], [], [], [], [], []
        msg_len = 0
        for i, (x_mb, _) in enumerate(val_loader):

            x_mb, loss, loss_recon, loss_kl, optional = eval_batch(model, x_mb, e, flags=flags)
            if loss is None:
                return None, None, None, None, None, None

            temp_losses.append(loss_recon.detach().cpu().numpy() + loss_kl.detach().cpu().numpy())
            temp_recon_losses.append(loss_recon.detach().cpu().numpy())
            temp_kl_losses.append(loss_kl.detach().cpu().numpy())

            if flags.val_nll:
                ll = model.log_likelihood(x_mb, n_sample).data
                if not torch.isnan(ll).sum() > 0:
                    temp_ll.append(float(ll))

            msg = ('\r\t %d \t loss: %.3f' %
                   (len(val_loader.dataset) // val_loader.batch_size - i, temp_losses[-1]))
            if model.name == 'vmf':
                msg += ('\t r: %.3f' % model.r.data.detach().cpu().numpy())

            print(' ' * msg_len, end='\r')
            print(msg, end='\r')
            msg_len = len(msg)

    return np.mean(temp_losses), np.mean(temp_recon_losses), np.mean(temp_kl_losses), (
        np.mean(temp_ll) if flags.val_nll else [])


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
def eval_batch(model, x_mb, e, flags=None):
    x_mb = x_mb.to(model.device)
    # dynamic binarization
    if flags.dynamic_binarization:
        x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape).to(model.device)).float()

    kl_weighted = 0.

    (q_z, p_z), z, x_mb_recon = model(x_mb.reshape(x_mb.size(0), -1))
    if z is None:
        return None, None, None, None, None
    loss_recon, loss_kl, _ = model.loss(q_z, p_z, x_mb, x_mb_recon)

    kl_weighted += (0 if flags.kl_freeze2 > e else 1.) * torch.abs(
        (max(0, min(flags.kl_warmup, e - flags.kl_freeze)) / flags.kl_warmup) * flags.beta * loss_kl - flags.kl_target)
    loss = loss_recon + kl_weighted

    return x_mb, loss, loss_recon, loss_kl, None


def train(model, optimizer, train_loader=None, val_loader=None, flags=None, path=None):
    """
    Train model for number of epochs
    :param model: VAE model
    :param optimizer: optimizer
    :param train_loader: training data loader
    :param val_loader: validation data loader
    :param flags: flags used to initialize model
    :param path: path to save models

    # :param epochs: number of epochs to train
    # :param kl_warmup: linear warmup period for KL scaling
    # :param kl_target: KL target
    # :param beta: final KL scale
    # :param pf: print frequency in epochs
    # :param val_nll: evaluate negative log-likelihood on validation
    # :param val_f: validation evaluation frequency
    # :param save_f: save frequency in epochs
    # :param look_ahead: number of epochs allowed without improvement validation loss
    # :param min_improv: minimal improvement needed to update best model
    # :param max_restarts: maximum number of failure restarts allowed
    # :param burn_in: epochs needed before recording best model updates

    :return: None
    """
    assert train_loader is not None

    # saving lists
    train_losses, train_kl_losses, train_recon_losses, train_radii = [], [], [], []
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

        if loss is None:  # safety mechanism to restart from checkpoint on NAN
            if (flags.max_restarts < num_restarts) or (last_save < 0):
                if last_save < 0:
                    print('no saved versions')
                else:
                    print('max restarts reached, terminating training')
                return None, (None, None, None, None), (None, None, None, None)
            else:
                num_restarts += 1
                model.num_restarts = max(model.num_restarts + 1, num_restarts)
                print('\n restarting model for %d time\n' % num_restarts)
                model, optimizer, epoch, loss, _ = load_from_checkpoint(path + '.tar')
        else:
            train_losses.append(loss)
            train_recon_losses.append(recon_loss)
            train_kl_losses.append(kl_loss)

            if model.name == 'vmf':
                train_radii.append(model.r.data.detach().cpu().numpy())

        if (e % flags.pf) == 0:
            msg = ('\r(train) e: %d (%.2f) \t loss: %.3f \t recon: %.3f \t kl: %.3f'
                   % (e, time() - t,
                      float(train_losses[-1]), float(train_recon_losses[-1]), float(train_kl_losses[-1])))
            if model.name == 'vmf':
                msg += ('\t r: %.3f' % model.r.data.detach().cpu().numpy())
            print(msg + '\n', end='')

        if ((e + 1) % flags.val_f) == 0:
            val_loss, val_recon, val_kl, val_ll_ = eval_val(model, e, val_loader, flags=flags)
            if loss is None:  # safety mechanism to restart from checkpoint on NAN
                if (flags.max_restarts < num_restarts) or (last_save < 0):
                    print('max restarts reached, terminating training')
                    return None, (None, None, None, None), (None, None, None, None)
                else:
                    num_restarts += 1
                    model.num_restarts = max(model.num_restarts + 1, num_restarts)
                    print('\n restarting model for %d time\n' % num_restarts)
                    model, optimizer, epoch, loss, _ = load_from_checkpoint(path + '.tar')
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

        # save model every X epochs to be save
        if ((e + 1) % flags.save_f) == 0:
            last_save = e
            create_checkpoint(path, epoch=e, model=model, optimizer=optimizer, loss=loss, flags=flags)

        if (e + 1) > flags.burn_in:
            if best_loss > (val_loss + flags.min_improv):
                print('\n update best \n')
                best_loss, best_epoch = val_loss, e
                path_ = path + '_best'
                create_checkpoint(path_, epoch=e, model=model, optimizer=optimizer, loss=loss, flags=flags)

            if (e - best_epoch) > flags.look_ahead:
                print('\n finish training \n')
                continue_training = False

    return loss, (train_losses, train_kl_losses, train_recon_losses, train_radii), \
        (val_losses, val_kl_losses, val_recon_losses, val_ll)


def test(model, n_sample, test_loader=None):
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

            print_['recon loss'].append(float(loss_recon.data))
            print_['KL'].append(float(loss_kl.data))
            print_['ELBO'].append(- print_['recon loss'][-1] - print_['KL'][-1])

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
    :return:
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'exp_flags': vars(flags),
    }, path + '.tar')


# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
def load_from_checkpoint(path):
    """
    Load saved model from checkpoint
    :param path: path to load model from
    :return:
    """
    print('\n load checkpoint from path: %s \n' % path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    flags = Namespace(**checkpoint['exp_flags'])
    h_dims = [int(h) for h in flags.h.split(",")]

    if flags.name == 'vmf' or flags.name == 'normal':
        z = flags.z if (flags.distribution == 'normal') else flags.z + 1
        model = VAE(input_size=flags.input_size, input_type=flags.input_type,
                    encode_type=flags.encode_type, decode_type=flags.decode_type,
                    h_dims=h_dims, z_dim=z, distribution=flags.distribution,
                    r=torch.tensor(flags.r_start, requires_grad=flags.r_grad, device=device),
                    device=device, flags=flags).to(device)
    elif flags.name == 'productspace':
        z = [int(z) + (1 if flags.distribution == 'vmf' else 0) for z in flags.z_dims.split(",")]
        model = ProductSpaceVAE(input_size=flags.input_size, input_type=flags.input_type,
                                encode_type=flags.encode_type, decode_type=flags.decode_type,
                                h_dims=h_dims, z_dims=z, distribution=flags.distribution,
                                device=device, flags=flags).to(device)
    else:
        raise NotImplemented

    optimizer = torch.optim.Adam(list(model.parameters()) + ([model.r]
                                                             if (flags.name == 'vmf' and flags.distribution == 'vmf')
                                                             else []), lr=flags.lr)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss, flags


# noinspection PyUnresolvedReferences
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


def numpy_(x):
    """
    Convenience function when working in CUDA environment predominantly to save / plot results
    :param x:
    :return:
    """
    return x.detach().cpu().numpy()
