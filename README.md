# Increasing the Expressivity of a Hyperspherical VAE
### A PyTorch implementation for the NeurIPS 2019, Workshop on Bayesian Deep Learning publication [[2]](#citation)

## Overview
This library contains a PyTorch implementation for a hyperspherical *product space* VAE [[2]](#citation) model based on the 
implementaion by [[1]](#citation):

## Dependencies

* **python>=3.8**
* **pytorch>=1.0**: https://pytorch.org
* **scipy**: https://scipy.org
* **numpy**: https://www.numpy.or

## Data
The experiments can be run on the following datasets:
* dynamic MNIST: dataset is downloaded through PyTorch `torchvision.datasets` routine;
* static MNIST: the dataset can be downloaded from [link](https://github.com/riannevdberg/sylvester-flows/tree/master/data/MNIST_static);
* FashionMNIST: dataset is downloaded through PyTorch `torchvision.datasets` routine;
* OMNIGLOT: the dataset can be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset can be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).
* Frey Faces: the dataset can be downloaded from [link](https://cs.nyu.edu/~roweis/data.html).

## Structure
* [hyperspherical_vae](https://github.com/trdavidson/increasing-expressivity-s-vae/tree/master/hyperspherical_vae)
    * [distributions](https://github.com/trdavidson/increasing-expressivity-s-vae/tree/master/hyperspherical_vae/distributions): Pytorch implementation of the r-von Mises-Fisher and hyperspherical Uniform distributions. Both inherit from `torch.distributions.Distribution`.
    * [ops](https://github.com/trdavidson/increasing-expressivity-s-vae/tree/master/hyperspherical_vae/ops): Low-level operations used for computing the exponentially scaled modified Bessel function of the first kind and its derivative
* [models](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/models.py): VAE and ProductSpaceVAE models
* [run_models](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/run_models.py): Run file to train / test model and save results
* [utils](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils.py): Utility functions 
    * [model_utils](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/model_utils.py): Utilities to support VAE model.
    * [classifying_utils](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/classifying_utils.py): Utilities for classification
    * [training_utils](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/training_utils.py): Utilities to support VAE model training.
    * [plotting_utils](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/plotting_utils.py): Utilities to investigate qualitative results of trained models.
    * [load_data](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/load_data.py): Data loading routines adopted from  [link](https://github.com/riannevdberg/sylvester-flows).
* [run_files](): Examples of `.sh` run files to run experiments


#### command line
The `run_model.py` file can be run from the command line using some of the following flags:

-    `-n, --name`, name of VAE model to run (vmf, normal, productspace).
-    `-dis, --distribution`, run either a vMF or Gaussian Normal.
-    `-z, --z`, dimension of latent space (regular VAE).
-    `-zd, --z_dims`, decomposition of latent space (ProductSpaceVAE).
-    `--h`, comma separated list outlining hidden units of encoder / decoder.
-    `-b, --beta`, scalar for the KL divergence.
-    `-e, --epochs`, number of epochs to train.
-    `-klw, --kl_warmup`, number of epochs to linearly scale-up the KL divergence.

Many more flags are defined when running the `help` command.

## License
MIT

## Citation
```
[1] Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T., and Tomczak, J. M. (2018). 
Hyperspherical Variational Auto-Encoders. (UAI-18).

[2] Davidson, T.R., Tomczak, J. M., and Gavves, E. (2019). 
Increasing the Expressivity of a Hyperspherical VAE. NeurIPS Workshop on Bayesian Deep Learning (NeurIPS-19).
```

BibTeX format:
```
@article{s-vae18,
  title={Hyperspherical Variational Auto-Encoders},
  author={Davidson, Tim R. and
          Falorsi, Luca and
          De Cao, Nicola and
          Kipf, Thomas and
          Tomczak, Jakub M.},
  journal={34th Conference on Uncertainty in Artificial Intelligence (UAI-18)},
  year={2018}
}

@article{davidson2019increasing,
  title={Increasing expressivity of a hyperspherical VAE},
  author={Davidson, Tim R and 
          Tomczak, Jakub M and 
          Gavves, Efstratios},
  journal={NeurIPS 2019, Workshop on Bayesian Deep Learning (NeurIPS-19)},
  year={2019}
}
```
