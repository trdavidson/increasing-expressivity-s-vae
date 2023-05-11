# largely copied from: source https://github.com/riannevdberg/sylvester-flows
import torch
import torch.utils.data as data_utils
from torchvision.datasets import MNIST
from torchvision import transforms
from scipy.io import loadmat
import numpy as np
import os
import math


def load_non_static_mnist(args):
    """
    Data loading function for dynamic mnist. Outputs image data in vectorized form: each image is a vector of size 784
    """
    args.dynamic_binarization = True
    args.input_type = 'binary'

    args.input_size = [1, 28, 28]

    def get_train_val_split(data, targets, num_val=1000):
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        train_idx, val_idx = idx[num_val:], idx[:num_val]
        train_data, train_labels = data[train_idx], targets[train_idx]
        val_data, val_labels = data[val_idx], targets[val_idx]

        train_ = data_utils.TensorDataset(train_data.type(torch.float32), train_labels)
        validation_ = data_utils.TensorDataset(val_data.type(torch.float32), val_labels)
        return train_, validation_

    # load MNIST dataset from PyTorch or local if previously downloaded
    dataset = MNIST('./data/MNIST', download=True, train=True, transform=transforms.ToTensor())
    test = MNIST('./data/MNIST', download=True, train=False, transform=transforms.ToTensor())
    if args.binarize:
        args.dynamic_binarization = False
        dataset.data = (dataset.data.float() > torch.distributions.Uniform(0, 1).sample(dataset.data.shape)).float()
        test.data = (test.data.float() > torch.distributions.Uniform(0, 1).sample(test.data.shape)).float()

    train, validation = get_train_val_split(dataset.data, dataset.targets, num_val=10000)
    test = data_utils.TensorDataset(test.data.type(torch.float32), test.targets)
    train_loader, val_loader, test_loader = _init_loaders(train, validation, test, args.batch_size)

    return train_loader, val_loader, test_loader, args


def load_static_mnist(args):
    """
    Data loading function for static mnist. Outputs image data in vectorized form: each image is a vector of size 784
    download link: https://github.com/riannevdberg/sylvester-flows/tree/master/data/MNIST_static
    """
    args.dynamic_binarization = False
    args.input_type = 'binary'

    args.input_size = [1, 28, 28]

    # start processing
    def lines_to_np_array(lines_):
        return np.array([[int(i) for i in line.split()] for line in lines_])

    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32').reshape((-1, 28, 28))
    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32').reshape((-1, 28, 28))
    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32').reshape((-1, 28, 28))

    # shuffle train data
    np.random.shuffle(x_train)

    # get idle y's
    y_train, y_val, y_test = get_idle_ys(x_train, x_val, x_test)

    # pytorch data loader
    train_loader, val_loader, test_loader = init_loaders(x_train, y_train, x_val, y_val, x_test, y_test,
                                                         args.batch_size)

    return train_loader, val_loader, test_loader, args


def load_omniglot(args):
    """
    Download link: https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat
    """
    n_validation = 1345

    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        # return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')
        return data.reshape((-1, 28, 28), order='F')

    omni_raw = loadmat(os.path.join('data', 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data
    x_train = train_data[:-n_validation]
    x_val = train_data[-n_validation:]

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # get idle y's
    y_train, y_val, y_test = get_idle_ys(x_train, x_val, x_test)

    # pytorch data loader
    train_loader, val_loader, test_loader = init_loaders(x_train, y_train, x_val, y_val, x_test, y_test,
                                                         args.batch_size)

    return train_loader, val_loader, test_loader, args


def load_caltech101silhouettes(args):
    """
    Download link: https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat
    """
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28), order='F')

    caltech_raw = loadmat(os.path.join('data', 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    train_loader, val_loader, test_loader = init_loaders(x_train, y_train, x_val, y_val, x_test, y_test,
                                                         args.batch_size)

    return train_loader, val_loader, test_loader, args


def init_data_sets(x_train, y_train, x_val, y_val, x_test, y_test):
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    return train, validation, test


def init_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    train, validation, test = init_data_sets(x_train, y_train, x_val, y_val, x_test, y_test)
    train_loader, val_loader, test_loader = _init_loaders(train, validation, test, batch_size)
    return train_loader, val_loader, test_loader


def _init_loaders(train, validation, test, batch_size):
    train_loader = TensorLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = TensorLoader(validation, batch_size=batch_size, shuffle=False)
    test_loader = TensorLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_idle_ys(x_train, x_val, x_test):
    """create dummy label tensors"""
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))
    return y_train, y_val, y_test


# https://gist.github.com/pimdh/eae45b6af5d75464fed38f398dca9967
class TensorLoader:
    """Speed up data loading for smaller data sets"""
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.i = 0
        self.idx = None

    def __iter__(self):
        self.i = 0
        self.idx = self.indices()
        return self

    def indices(self):
        idx = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idx)
        return idx

    def __next__(self):
        if self.i >= len(self.dataset):
            raise StopIteration()
        batch = self.dataset[self.idx[self.i: self.i + self.batch_size]]
        self.i += self.batch_size
        return batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def load_dataset(args):
    """check README or individual data loading routines for data set download links"""
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader, args = load_non_static_mnist(args)
    elif args.dataset == 'static_mnist':
        train_loader, val_loader, test_loader, args = load_static_mnist(args)
    elif args.dataset == 'caltech':
        train_loader, val_loader, test_loader, args = load_caltech101silhouettes(args)
    elif args.dataset == 'omniglot':
        train_loader, val_loader, test_loader, args = load_omniglot(args)
    else:
        raise Exception('Dataset not supported: [mnist, static_mnist, caltech, omniglot]')

    return train_loader, val_loader, test_loader, args
