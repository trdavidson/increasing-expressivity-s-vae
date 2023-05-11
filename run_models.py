import torch.utils.data
import numpy as np
import argparse
import datetime
import os
from utils.training_utils import train, test, load_from_checkpoint, init_model
from utils.load_data import load_dataset

LOG_DIR = "logs/"
LOG_SUB_DIR = "temp"

# GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_experiment(flags):
    """run experiment test as specified in user-defined flags"""
    torch.manual_seed(flags.rs)
    np.random.seed(flags.rs)

    # load dataset loaders
    flags.device = device
    train_loader, val_loader, test_loader, flags = load_dataset(flags)

    # create log directory
    os.makedirs(LOG_DIR + flags.log_dir, exist_ok=True)

    model = init_model(flags, device)
    h_dims = [int(h) for h in flags.h_dims.split(",")]

    z = flags.z if model.name != 'productspace' else [int(zd) for zd in flags.z_dims.split(",")]
    z = z if model.distribution != 'vmf' else (np.asarray(z) + 1)

    print(f'training {model.name}(dis: {model.distribution}) model on data({flags.dataset}):\n'
          f'architecture: input[{np.product(flags.input_size)}]: {flags.encode_type}{h_dims} - {z} - '
          f'{flags.decode_type}{h_dims[::-1]} : output[{np.product(flags.input_size)}]\n')
    optimizer = torch.optim.Adam(list(model.parameters()) + ([model.r]
                                                             if (flags.name == 'vmf' and flags.distribution == 'vmf')
                                                             else []), lr=flags.lr)
    # train a model
    path = LOG_DIR + flags.log_dir + "/" + flags.exp_name
    loss, stats = train(model, optimizer, train_loader=train_loader, val_loader=val_loader, flags=flags, path=path)

    if loss is None:
        print('training failed, too many restarts: %d' % model.num_restarts)
        test_stats = {}
    else:  # load best model found in training
        try:
            best_model, _, epoch, _, _ = load_from_checkpoint(path + '_best' + '.tar')
            print(f'best model found in epoch: {epoch} - evaluating test set using {flags.ll_sample} samples:')
        except FileNotFoundError as e:
            print(f'error: {e}\nusing latest model instead')
            best_model, epoch = model, -1

        # test (best/last) model
        test_stats = test(best_model, flags.ll_sample, test_loader)
        print([f'{k}: {v: .3f}' for k, v in test_stats.items()])

    # save stats for plotting and analysis
    exp_dict = {'train_success': loss is not None,
                'flags': vars(flags),
                'exp_name': flags.exp_name,
                'date': str(datetime.datetime.now().isoformat())
                }
    exp_dict.update(stats)
    exp_dict.update(test_stats)
    np.save(path + '_best_stats', np.array([exp_dict], dtype=object))


if __name__ == "__main__":
    """Pass in command line flags to specify experiment settings
    
    example usage:
    # single hypersphere (classic vMF VAE)
    python run_models.py -n vmf -zd 9 -e 5
    # two hyperspheres (product space vMF VAE)
    python run_models.py -n productspace -zd 4,4 -e 5
    """
    parser = argparse.ArgumentParser()

    # architecture
    parser.add_argument('-z', '--z', type=int, default=5,
                        help='latent space dimensionality')
    parser.add_argument('-zd', '--z_dims', type=str, default='2,3',
                        help='(productspace) latent dimensions, comma separated list')
    parser.add_argument('-hd', '--h_dims', type=str, default='256,128',
                        help='hidden units, comma separated list')
    parser.add_argument('-et', '--encode_type', type=str, default='mlp',
                        help='encoder block structure, [mlp, cnn')
    parser.add_argument('-dt', '--decode_type', type=str, default='mlp',
                        help='decoder block structure, [mlp, cnn')
    # model
    parser.add_argument('-dis', '--distribution', type=str, default='vmf',
                        help='one of [vmf, normal]')
    parser.add_argument('-cvm', '--covariance_matrix', type=str, default='diagonal',
                        help='use a [single, diagonal, full] covariance matrix for Gaussian')
    parser.add_argument('-n', '--name', type=str, default='productspace',
                        help='one of [vmf, normal, productspace]')

    # training
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
    parser.add_argument('-klf', '--kl_freeze', type=int, default=0,
                        help='number of epochs to freeze kl contribution')
    parser.add_argument('-klt', '--kl_target', type=float, default=0.,
                        help='KL target')
    parser.add_argument('-mr', '--max_restarts', type=int, default=20,
                        help='maximum number of allowed restarts for unstable models')
    parser.add_argument('-lls', '--ll_sample', type=int, default=100,
                        help='number of samples for log likelihood')

    # printing fields
    parser.add_argument('-pf', '--pf', type=int, default=1,
                        help='print frequency during training in epochs')
    parser.add_argument('-vll', '--val_nll', action='store_true', default=False,
                        help='evaluate validation negative log likelihood during training')
    parser.add_argument('-vf', '--val_f', type=int, default=1,
                        help='validation frequency during training in epochs')
    parser.add_argument('-sf', '--save_f', type=int, default=5,
                        help='save frequency during training in epochs')

    # saving/experiment fields
    parser.add_argument('-en', '--exp_name', type=str, default=datetime.datetime.now().isoformat(),
                        help='name of experiment')
    parser.add_argument('-ld', '--log_dir', type=str, default=LOG_SUB_DIR,
                        help='directory to store experiment')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        help='dataset to run experiment, [f_mnist, static_mnist, mnist, caltech, omniglot]')
    parser.add_argument('-bnz', '--binarize', action='store_true', default=False,
                        help='Only for pytorch mnist dataset')
    parser.add_argument('-rs', '--rs', type=int, default=np.random.randint(1, 1000),
                        help='random seed to use for experiment')
    parser.add_argument('-v', '--verbosity', type=int, default=0,
                        help='verbosity level of some print statements')

    test_flags, _ = parser.parse_known_args()
    if test_flags.burn_in > test_flags.epochs:
        print(f'warning: burn-in({test_flags.burn_in}) is more that #epochs({test_flags.epochs})! >Setting to #epochs')
        test_flags.burn_in = test_flags.epochs - 1

    run_experiment(test_flags)
