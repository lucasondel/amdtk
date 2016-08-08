
"""Mixture of distributions/densities."""

import numpy as np
from scipy.misc import logsumexp
from .discrete_latent_model import DiscreteLatentModel
from .dirichlet import Dirichlet


class MixtureStats(object):
    """Sufficient statistics for :class:BayesianMixture`.

    Methods
    -------
    __getitem__(key)
        Index operator.
    __add__(stats)
        Addition operator.
    __iadd__(stats)
        In-place addition operator.

    """

    def __init__(self, P_Z):
        self.__stats = P_Z.sum(axis=0)

    def __getitem__(self, key):
        if type(key) is not int:
            raise TypeError()
        if key < 0 or key > 2:
            raise IndexError
        return self.__stats

    def __add__(self, other):
        new_stats = MixtureStats(len(self.__stats))
        new_stats += self
        new_stats += other
        return new_stats

    def __iadd__(self, other):
        self.__stats += other.__stats
        return self


class BayesianMixture(DiscreteLatentModel):
    """Bayesian mixture of probability distributions (or densities).

     The prior is a Dirichlet density.

    Attributes
    ----------
    prior : :class:`Dirichlet`
        Prior density.
    posterior : :class:`Dirichlet`
        Posterior density.

    Methods
    -------
    expLogLikelihood(X)
        Expected value of the log-likelihood of the data given the
        model.
    KLPosteriorPrior()
        KL divergence between the posterior and the prior densities.
    updatePosterior(mixture_stats, pdf_stats)
        Update the parameters of the posterior distribution according to
        the accumulated statistics.
    """

    def __init__(self, alphas, components):
        super().__init__(components)
        self.prior = Dirichlet(alphas)
        self.posterior = Dirichlet(alphas.copy())

    def expLogLikelihood(self, X):
        """Expected value of the log-likelihood of the data given the
        model.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.

        Returns
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood for each frame.
        E_log_P_Z: numpy.ndarray
            Probability distribution of the latent states given the
            data.

        """
        E_log_weights = self.posterior.expLogPi()
        E_log_p_X = np.zeros((X.shape[0], self.k))
        for i, pdf in enumerate(self.components):
            E_log_p_X[:, i] += E_log_weights[i] + pdf.expLogLikelihood(X)
        log_norm = logsumexp(E_log_p_X, axis=1)
        E_log_P_Z = (E_log_p_X.T - log_norm).T
        return log_norm, E_log_P_Z

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns
        -------
        KL : float
            KL divergence.

        """
        KL = 0
        for component in self.components:
            KL += component.KLPosteriorPrior()
        return KL + self.posterior.KL(self.prior)

    def updatePosterior(self, mixture_stats):
        """Update the parameters of the posterior distribution.

        Parameters
        ----------
        mixture_stats : :class:MixtureStats
            Statistics of the mixture weights.

        """
        self.posterior = self.prior.newPosterior(mixture_stats)
