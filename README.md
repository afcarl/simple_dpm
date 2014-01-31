A Simple Dirichlet Process Mixture Example
==========================================

A small, straightforward Python implementation of Dirichlet process mixtures (DPMs). Inspired by Escobar and West (1995), this example shows how to fit a mixture of three univariate Gaussians with a DPM and a normal inverse-gamma prior.

Example
-------

We start with 200 observed data points drawn from the true distribution.

![Observed Data](https://github.com/tansey/simple_dpm/raw/master/points.png)

The results of running with the default parameters are below.

![DPM Gibbs Results](https://github.com/tansey/simple_dpm/raw/master/results.png)

Since the true number of clusters is three, it's a good sign that it looks like that's approximately the mode of the distribution.

![DPM Cluster Sizes](https://github.com/tansey/simple_dpm/raw/master/cluster_counts.png)

Sensitivity
-----------

The model is __very__ sensitive to the prior in the above example. Specifically:

    - Changing alpha higher, say to 0.25, will shift the cluster distribution to the right and make the mode around 6.

    - Changing the parameters of the normal inverse-gamma prior can easily skew the resulting assignments and cluster parameters.