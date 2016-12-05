
"""Bayesian Mixture of Gaussian."""

import numpy as np
from scipy.special import psi, gammaln
from .model import Model


class Mixture(Model):
    """Bayesian mixture of Gaussian with a Dirichlet prior
    for the weights.

    """

    def __init__(self, components, prior_count):
        """Initialize the mixture.

        Parameters
        ----------
        components : list
            List of :class:`Gaussian` components.
        prior_count : numpy.ndarray
            Hyper-parameters of the Dirichlet prior.

        """
        super().__init__()
        self.prior_count = prior_count
        self.posterior_count = prior_count
        self.components = components

    def get_stats(self, data, weights):
        """Compute the sufficient statistics for the model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.
        weights : numpy.ndarray
            Weights (N x K) for each frame and each component.

        Returns
        -------
        stats : dict
            Nested dictionaries. Statistics for a specific model
            are accessible by the key (model.id) of the model.

        """
        stats_data = {}
        stats_data[self.uid] = {}
        stats_data[self.uid]['s0'] = weights.sum(axis=0)
        for i, component in enumerate(self.components):
            stats_data[component.uid] = {}
            stats_data[component.uid] = \
                component.get_stats(data, weights[:, i])

        return stats_data

    def expected_log_likelihood(self, data):
        """Expected value of the log likelihood.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.

        Returns
        -------
        E_llh : float
            Expected value of the log-likelihood.

        """
        ncomps = len(self.components)
        llh = np.zeros((data.shape[0], ncomps))
        log_weights = psi(self.posterior_count) - \
            psi(self.posterior_count.sum())
        for i, component in enumerate(self.components):
            llh[:, i] = log_weights[i] + \
                component.expected_log_likelihood(data)
        return llh

    def update(self, stats):
        """ Update the posterior parameters given the sufficient
        statistics.

        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.

        """
        self.posterior_count = self.prior_count + stats[self.uid]['s0']
        for component in self.components:
            component.update(stats)

    def kl_divergence(self):
        """Kullback-Leibler divergence between the posterior and
        the prior density.

        Returns
        -------
        ret : float
            KL(q(params) || p(params)).

        """
        kl_div = 0.
        kl_div = gammaln(self.posterior_count.sum())
        kl_div -= gammaln(self.prior_count.sum())
        kl_div -= gammaln(self.posterior_count).sum()
        kl_div += gammaln(self.prior_count).sum()
        log_weights = psi(self.posterior_count) - \
            psi(self.posterior_count.sum())
        kl_div += (self.posterior_count - self.prior_count).dot(log_weights)
        for component in self.components:
            kl_div += component.kl_divergence()
        return kl_div
