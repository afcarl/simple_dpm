import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import *
from scipy.stats import norm, invgamma, t
import random
from copy import deepcopy
import pylab

# Mixture of 3 normals. Assume uniform weights.
TRUE_CLUSTERS = np.array([(5,1),(-6,1.25),(0,1.5)])

# Number of observed data points
NUM_POINTS = 200

# Number of MCMC iterations to run
ITERATIONS = 1100

# Number of initial MCMC runs to skip
BURN_IN = 100

# Number of MCMC iterations per saved sample
THIN = 2

# Concentration parameter for the DP prior
ALPHA = 0.1

# Normal Inverse-Gamma hyperparameters
MU_0 = 0
NU_0 = 0.1
A_0 = 1
B_0 = 1

def plot_noisy_means(graph_title, means, bands, series, xvals=None, xlabel=None, ylabel=None, subtitle=None, data=None, filename='results.pdf'):
    colors = ['blue','red','green', 'black', 'orange', 'purple', 'brown', 'yellow'] # max 8 lines
    assert(means.shape == bands.shape)
    assert(xvals is None or xvals.shape[0] == means.shape[1])
    assert(means.shape[0] <= len(colors))
    if xvals is None:
        xvals = np.arange(means.shape[0])
    ax = plt.axes([.1,.1,.8,.7])
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)
    for i,mean in enumerate(means):
        plt.plot(xvals, mean, label=series[i], color=colors[i])
        plt.fill_between(xvals, mean + bands[i], mean - bands[i], facecolor=colors[i], alpha=0.2)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if subtitle is not None:
        plt.figtext(.40,.9, graph_title, fontsize=18, ha='center')
        plt.figtext(.40,.85, subtitle, fontsize=10, ha='center')
    else:
        plt.title('{0}'.format(graph_title))
    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.savefig(filename)
    plt.clf()

def plot_cluster_distribution(clusters, fit_normal=False, filename='cluster_counts.pdf'):
    lengths = np.array(map(len, clusters))
    start = lengths.min()
    end = lengths.max()
    # the histogram of the data with histtype='step'
    n, bins, patches = plt.hist(lengths, end - start + 1)
    if fit_normal:
        # add a line showing the expected distribution
        norm_xvals = np.arange(start, end, 0.001)
        mean = lengths.mean()
        stdev = lengths.std()
        y = np.array([norm.cdf(norm_xvals[0], mean, stdev)] + [norm.cdf(norm_xvals[i], mean, stdev) - norm.cdf(norm_xvals[i-1], mean, stdev) for i in xrange(1, len(norm_xvals))])
        l = plt.plot(norm_xvals, y, 'k--', linewidth=4, color='red')
    plt.savefig(filename)
    plt.clf()

def plot_mcmc_samples(data, samples, true_clusters, x_min= -15, x_max=15, stepsize=0.001):
    # Plot the data
    plt.title('Observed Data Points\n{0} total'.format(len(data)))
    plt.hist(data, 20)
    plt.savefig('points.pdf')
    plt.clf()
    
    # Plot the mean and bands of size 2 stdevs for the samples.
    # Also plot the last sample from the MCMC chain.
    xvals = np.arange(x_min, x_max, stepsize)
    true_pdf = sum(norm.pdf(xvals, mean, stdev) for mean, stdev in true_clusters) / float(len(true_clusters))
    no_bands = np.zeros(len(xvals))
    sample_pdfs = np.array([sum([norm.pdf(xvals, mean, stdev) for mean, stdev in sample]) / float(len(sample)) for sample in samples])
    sample_means = sample_pdfs.mean(axis=0)
    sample_bands = sample_pdfs.std(axis=0)*2.
    last = sample_pdfs[-1]
    means = np.array([true_pdf, sample_means, last])
    bands = np.array([no_bands, sample_bands, no_bands])
    names = ['True PDF', 'Bayes\nEstimate\n(+/- 2 stdevs)', 'Last MCMC\nsample']
    mcmc_params = '{0} points, {1} iterations, {2} burn-in, {3} thin, {4} samples'.format(NUM_POINTS, ITERATIONS, BURN_IN, THIN, len(samples))
    dp_params = 'alpha={0}, mu0={1}, nu0={2}, a0={3}, b0={4}'.format(ALPHA, MU_0, NU_0, A_0, B_0)
    plot_noisy_means('Dirichlet Process Mixture Results', means, bands, names, xvals=xvals, xlabel='X', ylabel='Probability', subtitle='{0}\n{1}'.format(mcmc_params, dp_params))
    
    # Plot the distribution of clusters in the samples
    plot_cluster_distribution(samples)

def sample_uniform_mixture(clusters):
    mean,stdev = random.choice(clusters)
    return np.random.normal(mean, stdev)

def sample_mixture(clusters):
    u = random.random()
    cur = 0
    for mean,stdev,weight in clusters:
        cur += weight
        if u < cur:
            return np.random.normal(mean, stdev)

def weighted_sample(weights):
    probs = weights / weights.sum()
    u = random.random()
    cur = 0.
    for i,p in enumerate(probs):
        cur += p
        if u <= cur:
            return i
    raise Exception('Weights do not normalize properly! {0}'.format(weights) )

def crp_prior(draws, alpha):
    ''' Initialize the cluster assignments via draws from the CRP prior '''
    assignments = []
    table_counts = []
    for n in xrange(draws):
        new_table = alpha / float(n+alpha)
        # Sit at a new table with probability proportional to alpha
        if random.random() <= new_table:
            assignments.append(len(table_counts))
            table_counts.append(1)
        # Sit at an existing table
        else:
            val = random.randrange(n)
            cur = 0
            for table,count in enumerate(table_counts):
                cur += count
                if val < cur:
                    assignments.append(table)
                    table_counts[table] += 1
                    break
    return np.array(assignments)

class NormalInverseGamma(object):
    '''
    A conjugate prior for a simple univariate Gaussian with unknown
    mean and variance.
    '''
    def __init__(self, mu0, nu0, a0, b0):
        self.mu = mu0
        self.nu = nu0
        self.a = a0
        self.b = b0

    def sample(self):
        ''' Samples a mean and stdev from the Normal-InverseGamma distribution. '''
        variance = invgamma.rvs(self.a, scale=self.b)
        mean = np.random.normal(self.mu, np.sqrt(variance / self.nu))
        return (mean, np.sqrt(variance))

    def posterior(self, data):
        ''' Returns a posterior distribution with the hyperparameters updated given the data. '''
        mean = data.mean()
        n = len(data)
        mu1 = (self.nu * self.mu + n*mean) / (self.nu + n)
        nu1 = self.nu + n
        a1 = self.a + n/2.
        b1 = self.b + 0.5 * np.sum((data - mean)**2) + n * self.nu / (self.nu + n) * (mean - self.mu)**2 / 2.
        return NormalInverseGamma(mu1, nu1, a1, b1)

if __name__ == "__main__":
    # Generate samples from the actual mixture
    data = np.array([sample_uniform_mixture(TRUE_CLUSTERS) for _ in xrange(NUM_POINTS)])

    cluster_prior = NormalInverseGamma(MU_0, NU_0, A_0, B_0)

    # Initialize the assignments
    assignments = crp_prior(len(data), ALPHA)

    # Track the size of each table
    counts = [np.where(assignments == x)[0].shape[0] for x in xrange(assignments.max()+1)]
    
    # Initialize the clusters by sampling from the posterior given the assigned data points
    clusters = [cluster_prior.posterior(data[np.where(assignments == x)]).sample() for x in xrange(len(counts))]

    samples = []
    for iteration in xrange(ITERATIONS):
        # TODO: Should data points be randomized? (Murphy, K., 2012) says they should.
        # Draw assignments
        for i,y in enumerate(data):
            # Remove y from its current cluster
            k = assignments[i]
            counts[k] -= 1

            # If this was the last data point in this cluster, delete it
            if counts[k] == 0:
                del counts[k]
                del clusters[k]
                # Update the index of all the other assignments
                assignments[assignments > k] -= 1

            # Calculate the weight for a new cluster
            # See Escobar and West (1995) for details on why this is the weight.
            # See the Wikipedia page on conjugate priors for the form of the Student's t
            # distribution.
            new_cluster_posterior = cluster_prior.posterior(np.array([y]))
            t_scale = new_cluster_posterior.b * (new_cluster_posterior.nu + 1) / (new_cluster_posterior.a * new_cluster_posterior.nu)
            new_cluster_weight = ALPHA * t.pdf(y, 2. * new_cluster_posterior.a, loc=new_cluster_posterior.mu, scale=t_scale)

            # Calculate the weight for all the other clusters
            z = [counts[k] * norm.pdf(y, kmean, kstdev) for k,(kmean, kstdev) in enumerate(clusters)]
            z.append(new_cluster_weight)
            weights = np.array(z)

            # Draw a new assignment proportional to the cluster weights
            k = weighted_sample(weights)
            assignments[i] = k

            # If we sampled a new cluster
            if k == len(clusters):
                # We need to sample the parameters from the prior
                # TODO: should we instead sample from the posterior with the one sample?
                kmean, kstdev = cluster_prior.sample()
                clusters.append((kmean, kstdev))
                counts.append(1)
            # Otherwise we sampled an existing cluster
            else:
                counts[k] += 1

        # Draw cluster parameters
        next_clusters = []
        for k,cluster in enumerate(clusters):
            # Get the data points assigned to this cluster
            cluster_points = data[np.where(assignments == k)]

            # Figure out the posterior and sample a new mean and stdev from it
            kmean,kstdev = cluster_prior.posterior(cluster_points).sample()

            # Add the cluster to the mixture model
            next_clusters.append([kmean, kstdev])

        # Update the clusters to the newly sampled values
        clusters = next_clusters

        # We only keep samples after the chain has run long enough. Should
        # help overcome the bias of initial conditions and autocorrelation.
        if iteration >= BURN_IN and iteration % THIN == 0:
            samples.append(np.array(clusters))

        if iteration % 50 == 0:
            print 'Iteration #{0}'.format(iteration)

    plot_mcmc_samples(data, samples, TRUE_CLUSTERS)
