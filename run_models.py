# import numpy as np
# import torch
import torch.utils.data
import argparse
import datetime
# to enable saving plots
import matplotlib

matplotlib.use('agg')
from models import VAE, ProductSpaceVAE
from utils.training_utils import train, test, load_from_checkpoint, get_embeddings
from utils.load_data import *
from utils.classifying_utils import *

LOG_DIR = "logs/"
LOG_SUB_DIR = "temp"
LOG_VISUAL_SUB_DIR = "visual/"

# GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# noinspection PyCallingNonCallable
# noinspection PyUnresolvedReferences
def run_test(flags):
    """
    runs a test as specified in flags
    """

    torch.manual_seed(flags.rs)
    np.random.seed(flags.rs)

    # load dataset loaders
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    flags.device = device
    train_loader, val_loader, test_loader, flags = load_dataset(flags, **kwargs)

    # create log directory
    os.makedirs(LOG_DIR + flags.log_dir, exist_ok=True)

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

    print('training model on data(%s): %s(dis: %s)\n --> %s - %s - %s' %
          (flags.dataset, model.name, model.distribution, flags.encode_type, str(z), flags.decode_type))
    optimizer = torch.optim.Adam(list(model.parameters()) + ([model.r]
                                                             if (flags.name == 'vmf' and flags.distribution == 'vmf')
                                                             else []), lr=flags.lr)

    # train a model
    # with torch.autograd.detect_anomaly():  # super useful to debug
    path = LOG_DIR + flags.log_dir + "/" + flags.exp_name
    loss, \
    (train_losses, train_kl_losses, train_recon_losses, train_radii), \
    (val_losses, val_kl_losses, val_recon_losses, val_ll) = train(model, optimizer, train_loader=train_loader,
                                                                  val_loader=val_loader, flags=flags, path=path)

    if loss is None:
        print('training failed, too many restarts: %d' % model.num_restarts)
        test_eval, test_knn = None, None
    else:
        # load best model found in training
        best_model, _, epoch, _, _ = load_from_checkpoint(path + '_best' + '.tar')
        print('best model found in epoch: %d' % epoch)
        # test a model
        test_eval = test(best_model, flags.ll_sample, test_loader)
        print(test_eval)

        if flags.visualize_latent:
            # create log directory
            os.makedirs(LOG_DIR + LOG_VISUAL_SUB_DIR, exist_ok=True)

            # visualize the learned latent space
            z, y = get_embeddings(best_model, test_loader)
            f = path.replace(flags.log_dir + '/', LOG_VISUAL_SUB_DIR) + '_best'
            # (optional): insert some visualization code

    # save stats for plotting and analysis
    exp_dict = {'train_success': loss is not None,
                'loss': train_losses,
                'recon_loss': train_recon_losses,
                'kl_loss': train_kl_losses,
                'val_loss': val_losses,
                'val_recon_loss': val_recon_losses,
                'val_kl_loss': val_kl_losses,
                'val_ll': val_ll,
                'test_eval': test_eval,
                'radii': train_radii,
                'flags': vars(flags),
                'exp_name': flags.exp_name,
                'date': str(datetime.datetime.now().isoformat())
                }

    np.save(path + '_best' + '_stats', np.array([exp_dict], dtype=object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--z', type=int, default=5,
                        help='latent space dimensionality')
    parser.add_argument('-zd', '--z_dims', type=str, default='2,3',
                        help='(productspace) latent dimensions, comma separated list')
    parser.add_argument('--h', type=str, default='256,128',
                        help='hidden units, comma separated list')
    parser.add_argument('-et', '--encode_type', type=str, default='mlp',
                        help='encoder block structure, [mlp, cnn')
    parser.add_argument('-dt', '--decode_type', type=str, default='mlp',
                        help='decoder block structure, [mlp, cnn')
    parser.add_argument('-dis', '--distribution', type=str, default='vmf',
                        help='one of [vmf, normal, gumbel_softmax, atom]')
    parser.add_argument('-cvm', '--covariance_matrix', type=str, default='diagonal',
                        help='use a [single, diagonal, full] covariance matrix for Gaussian')
    parser.add_argument('-n', '--name', type=str, default='productspace',
                        help='one of [vmf, normal, productspace]')
    parser.add_argument('-r1', '--r_start', type=float, default=1.,
                        help='radius of hypersphere')
    parser.add_argument('-r2', '--r_end', type=float, default=15.,
                        help='radius of hypersphere')
    parser.add_argument('-rg', '--r_grad', action='store_true', default=False,
                        help='boolean indicating if radius is a (hyper)parameter')
    parser.add_argument('-lf', '--loss_function', type=str, default='bce',
                        help='loss function to use for reconstruction loss, [bce, mse]')
    parser.add_argument('-b', '--beta', type=float, default=1.,
                        help='scale kl divergence')
    parser.add_argument('-bi', '--burn_in', type=int, default=100,
                        help='number of epochs to train for before recording best model')
    parser.add_argument('-la', '--look_ahead', type=int, default=50,
                        help='number of epochs to train for until terminating without val score improvement')
    parser.add_argument('-mi', '--min_improv', type=float, default=1e-2,
                        help='minimal improvement needed to reset look_ahead counter')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='number of epochs to train for')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                        help='learning rate of model')
    parser.add_argument('-bs', '--batch_size', type=int, default=100,
                        help='batch size used for training')
    parser.add_argument('-klw', '--kl_warmup', type=int, default=100,
                        help='number of epochs to linearly scale KL')
    parser.add_argument('-klw2', '--kl_warmup2', type=int, default=100,
                        help='number of epochs to linearly scale KL')
    parser.add_argument('-klf', '--kl_freeze', type=int, default=0,
                        help='number of epochs to freeze kl contribution')
    parser.add_argument('-klf2', '--kl_freeze2', type=int, default=0,
                        help='number of epochs to freeze kl contribution')
    parser.add_argument('-klt', '--kl_target', type=float, default=0.,
                        help='KL target')
    parser.add_argument('-klt2', '--kl_target2', type=float, default=0.,
                        help='KL target 2')
    parser.add_argument('-mr', '--max_restarts', type=int, default=20,
                        help='maximum number of allowed restarts for unstable models')
    parser.add_argument('-lls', '--ll_sample', type=int, default=100,
                        help='number of samples for log likelihood')
    parser.add_argument('-bnz', '--binarize', action='store_true', default=False,
                        help='Only for pytorch mnist dataset')
    # printing fields
    parser.add_argument('-pf', '--pf', type=int, default=1,
                        help='print frequency during training in epochs')
    parser.add_argument('-vll', '--val_nll', action='store_true', default=False,
                        help='evaluate validation negative log likelihood during training')
    parser.add_argument('-vf', '--val_f', type=int, default=1,
                        help='validation frequency during training in epochs')
    parser.add_argument('-sf', '--save_f', type=int, default=5,
                        help='save frequency during training in epochs')
    parser.add_argument('-k', '--k', type=int, default=1,
                        help='k-nearest neighbors')
    parser.add_argument('--num_labels', type=int, default=100,
                        help='number of labels to use in k-nn')
    # synthetic data fields
    parser.add_argument('-sod', '--synthetic_outdim', type=int, default=100,
                        help='output dimension of synthetic data')
    parser.add_argument('-sne', '--synthetic_noise_eps', type=float, default=2.,
                        help='noise epsilon of synthetic data')
    parser.add_argument('-sds', '--synthetic_data_size', type=int, default=2000,
                        help='number synthetic data points per class')
    parser.add_argument('-sdz', '--synthetic_data_z', type=int, default=1,
                        help='latent dimensionality synthetic data')
    parser.add_argument('-sdnl', '--synthetic_data_nonlinearity', action='store_true', default=False,
                        help='non-linearity on synthetic data creation')
    # saving fields
    parser.add_argument('-visl', '--visualize_latent', action='store_true', default=False,
                        help='visualize latent space and save image')
    parser.add_argument('-en', '--exp_name', type=str, default=datetime.datetime.now().isoformat(),
                        help='name of experiment')
    parser.add_argument('-ld', '--log_dir', type=str, default=LOG_SUB_DIR,
                        help='directory to store experiment')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        help='dataset to run experiment, [f_mnist, static_mnist, mnist, freyfaces, caltech, omniglot]')
    parser.add_argument('-rs', '--rs', type=int, default=np.random.randint(1, 1000),
                        help='random seed to use for experiment')
    parser.add_argument('-v', '--verbosity', type=int, default=0,
                        help='verbosity level of some print statements')

    test_flags, _ = parser.parse_known_args()

    run_test(test_flags)
