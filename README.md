# Increasing the Expressivity of a Hyperspherical VAE

## Overview
This repository contains a PyTorch implementation to train a hyperspherical *product space* VAE,
as presented in the NeurIPS 2019, Workshop on Bayesian Deep Learning publication 
"_Increasing the Expressivity of a Hyperspherical VAE_"[[2]](#citation) (https://arxiv.org/abs/1910.02912).

The core idea of the paper is best summarized as follows:
- Hyperspherical latent space (S^m) is sometimes preferred over 'flat' Euclidean space (R^m+1). This can be
accomplished by using a _von Mises-Fisher_ distribution (vMF) [[1]](#citation)
- Unfortunately, a vMF has limitations as dimensionality, _m_, increases:
    1. **flexiblity**: the concentration parameter, κ, is shared across all dimensions _m_
    2. **instability**: hyperspherical surface area converges to 0 for _m_ > 17

idea: break up S^m into multiple S^i, where the sum of all i <= _m_:
- (+) **increased flexibility**: κ_i concentration parameters are shared between fewer dimensions
- (+) **increased stability**: surface area of lower dimensional hyperspheres doesn't converge to 0
- (-) **fewer degrees of freedom**: when keeping total ambient space *m* fixed, each break loses a degree of freedom 

<p align="center">
<img src="https://github.com/trdavidson/increasing-expressivity-s-vae/blob/main/sxvae-figure.png" alt="sx9 example"/>
</p>

_Structural interpolation of S^9 ⊂ R^10, where each corner represents a separate dimension such that the dimensionality 
of the cross- product of (b), (c) can be smoothly embedded in R^10. Each separate hypersphere is equipped with an 
independent concentration parameter κ._

**Note**: since the publication of this work, De Cao and Aziz have proposed the 'Power Spherical' distribution as a 
more scalable and numerically stable alternative to the vMF [[3]](#citation). A similar hyperspherical break-up strategy 
could be pursued with this new distribution to increase concentration parameter flexiblity 
[(link to codebase)](https://github.com/nicola-decao/power_spherical).

## Dependencies

* **python>=3.8**
* **pytorch>=2.0**: https://pytorch.org
* **scipy**: https://scipy.org  

_Note: Older versions could work but are not tested_

Optional dependency for plotting:
* **matplotlib**: https://matplotlib.org

## Data
Standard public datasets used in experiments
* Dynamic MNIST: download using `torchvision.datasets` routine;
* Static MNIST: [download source link](https://github.com/riannevdberg/sylvester-flows/tree/master/data/MNIST_static);
* OMNIGLOT: [download source link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: [download source link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).

## Structure
* [models.py](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/models.py): VAE and ProductSpaceVAE models
* [run_models.py](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/run_models.py): run-file to perform experiments
* [utils](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils.py): 
    * [model_utils.py](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/model_utils.py): MLP, CNN, full-covariance matrix routines;
    * [training_utils.py](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/training_utils.py): training, evaluation, test routines;
    * [plotting_utils.py](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/plotting_utils.py): sample plots, latent interpolations, hammer projection etc.
    * [load_data.py](https://github.com/trdavidson/increasing-expressivity-s-vae/blob/master/utils/load_data.py): data loading routines
* [hyperspherical_vae](https://github.com/trdavidson/increasing-expressivity-s-vae/tree/master/hyperspherical_vae)
    * [see s-vae github for details](https://github.com/nicola-decao/s-vae-pytorch)


### command line examples
The `run_model.py` file can be run from the command line using some of the following flags:

-    `-n, --name`, name of VAE model to run (vmf, normal, productspace).
-    `-dis, --distribution`, run either a vMF or Gaussian Normal.
-    `-z, --z`, dimension of latent space (regular VAE).
-    `-zd, --z_dims`, decomposition of latent space (ProductSpaceVAE).
-    `-hd, --h_dims`, comma separated list outlining hidden units of encoder / decoder (symmetric encoding/decoding assumption).
-    `-e, --epochs`, number of epochs to train.

More flags are defined when running the `--help` command. Two basic examples to run:
```bash
# single hypersphere (classic vMF VAE)
python run_models.py -n vmf -zd 9 -e 5
# two hyperspheres (product space vMF VAE)
python run_models.py -n productspace -zd 4,4 -e 5
``` 

## License
MIT

## Citation
```
[1] Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T., and Tomczak, J. M. (2018). 
Hyperspherical Variational Auto-Encoders. (UAI-18).

[2] Davidson, T.R., Tomczak, J. M., and Gavves, E. (2019). 
Increasing the Expressivity of a Hyperspherical VAE. NeurIPS Workshop on Bayesian Deep Learning (NeurIPS-19).

[3] De Cao, N. and Aziz, W. (2020).
The Power Spherical Distribution. ICML Workshop on INNF+ (ICML-20).
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

@article{de2020power,
  title={The power spherical distribution},
  author={De Cao, Nicola and Aziz, Wilker},
  journal={ICML 2020, Workshop INNF+ (ICML-20)},
  year={2020}
}
```
