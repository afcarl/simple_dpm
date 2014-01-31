Simple Dirichlet Process Mixtures
==========================================

A small, straightforward Python implementation of Dirichlet process mixtures (DPMs). Inspired by Escobar and West [1], this example shows how to fit a mixture of three univariate Gaussians with a DPM and a normal inverse-gamma prior.

Example
-------

The results below are for the default settings in the `dp_mixture.py` file.

![Observed Data](https://github.com/tansey/simple_dpm/raw/master/points.png)

We start with 200 observed data points drawn from the true distribution.

![DPM Gibbs Results](https://github.com/tansey/simple_dpm/raw/master/results.png)

The results show that Bayes estimate is doing an okay, though not great, job at estimating the density.

![DPM Cluster Sizes](https://github.com/tansey/simple_dpm/raw/master/cluster_counts.png)

Since the true number of clusters is three, it's a good sign that we're getting a lot of draws from around there. Though it's overestimating a bit, that's to be expected [2].

Sensitivity
-----------

The model is __very__ sensitive to the prior in the above example. Specifically:

 - Changing alpha higher, say to 0.25, will shift the cluster distribution to the right and make the mode around 6.

 - Changing the parameters of the normal inverse-gamma prior can easily skew the resulting assignments and cluster parameters.

References
----------

[1] Escobar, Michael D., and Mike West. "Bayesian density estimation and inference using mixtures." Journal of the American Statistical Association. Vol. 90, No. 430 (1995): 577-588.

[2] Miller, Jeffrey W., and Matthew T. Harrison. "A simple example of Dirichlet process mixture inconsistency for the number of components." (NIPS 2013).