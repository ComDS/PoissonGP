# PoissonGP: Bayesian Non-parameteric Method for Sales Forecasting

Demo for the model in the research paper titled "Bayesian Non-parametric Method for Decision Support: Forecasting Online Product Sales." (2023) **_Decision Support Systems_**. and its further extensions.

**Update@20240522**: Considering some usage changes in the theano package (which is required to run pymc3), such as the removal of some functions and compatibility issues with pymc3, we now highly recommend that researchers use pyro (https://docs.pyro.ai/en/stable/) to expand PoissonGP now. The package is based on PyTorch framework, which makes probability programming easier to use.

**Upload@20230523**: Publish the demo and add the license.

## Computing Requirements

To ensure that the code runs successfully, the following dependencies need to be installed:

```
python 3.8.17
numpy 1.24.4
pymc3 3.11.4
```

Note: This code needs to be based on the pymc3 package (not pymc), which can be installed by `pip install pymc3`. Other base packages have no specific restrictions, and researchers can replicate models and continue developments. Our code is run under the GPU of GeForce RTX 3090. 


## File Structures

Here's our demo:

* **genpoi.py**: generate and analyze posterior, generate simulation results
* **gp_dp.py**: a simple demo for GP-Poission model.


## Data

We provide some simulation data and empirical demo. To comply with confidentiality regulations, complete raw (sales record) data is available on demand and remains for internal use for academic purposes. Due to the policies and regulations of the original data provider, we are unable to offer complete product attributes. The exact timestamp is recleaned, and the data reflects the order quantity but does not contain information about the products themselves.


## Citations

If you find this work interesting, please cite

```
@article{wu2023bayesian,
  title={Bayesian non-parametric method for decision support: Forecasting online product sales},
  author={Wu, Ziyue and Chen, Xi and Gao, Zhaoxing},
  journal={Decision support systems},
  volume={174},
  pages={114019},
  year={2023},
  publisher={Elsevier}
}
```