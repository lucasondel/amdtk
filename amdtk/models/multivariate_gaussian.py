
"""Multivariate Gaussian density."""

from .normal_gamma import NormalGamma


class GaussianDiagCovStats(object):
    """Sufficient statistics for :class:BayesianGaussianDiagCov`.

    Methods
    -------
    __getitem__(key)
        Index operator.
    __add__(stats)
        Addition operator.
    __iadd__(stats)
        In-place addition operator.

    """

    def __init__(self, X, weights):
        weighted_X = (weights*X.T).T
        self.__stats = [weights.sum(), weighted_X.sum(axis=0),
                        (weighted_X*X).sum(axis=0)]

    def __getitem__(self, key):
        if type(key) is not int:
            raise TypeError()
        if key < 0 or key > 2:
            raise IndexError()
        return self.__stats[key]

    def __add__(self, other):
        new_stats = GaussianDiagCovStats(len(self.__stats[1]))
        new_stats += self
        new_stats += other
        return new_stats

    def __iadd__(self, other):
        self.__stats[0] += other.__stats[0]
        self.__stats[1] += other.__stats[1]
        self.__stats[2] += other.__stats[2]
        return self


class BayesianGaussianDiagCov(object):
    """Bayesian multivariate Gaussian with a diagonal covariance matrix.

    The prior over the mean and variance for each dimension is a
    Normal-Gamma density.

    Attributes
    ----------
    prior : :class:`NormalGamma`
        Prior density.
    posterior : :class:`NormalGamma`
        Posterior density.

    Methods
    -------
    expLogLikelihood(X)
        Expected value of the log-likelihood of X.
    KLPosteriorPrior()
        KL divergence between the posterior and the prior densities.
    updatePosterior(stats)
        Update the parameters of the posterior density.

    """

    def __init__(self, mu_0, kappa_0, alpha_0, beta_0, mu_n, kappa_n, alpha_n,
                 beta_n):
        self.prior = NormalGamma(mu_0, kappa_0, alpha_0, beta_0)
        self.posterior = NormalGamma(mu_n, kappa_n, alpha_n, beta_n)

    def expLogLikelihood(self, X):
        """Expected value of the log-likelihood of X.

        Expected value of the log-likelihood of X given the
        hyper-parmeters of the multivariate Gaussian density
        distribution up to a constant.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.

        Returns
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood for each frame.

        """
        log_precision = self.posterior.expLogPrecision()
        precision = self.posterior.expPrecision()
        log_norm = (log_precision - 1/self.posterior.kappa)
        return .5*(log_norm - precision*(X - self.posterior.mu)**2).sum(axis=1)

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns
        -------
        KL : float
            KL divergence.

        """
        return self.posterior.KL(self.prior)

    def updatePosterior(self, stats):
        """Update the parameters of the posterior density.

        Parameters
        ----------
        stats : :class:MultivariateGaussianDiagCovStats
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:Dirichlet
            New Dirichlet density.

        """
        threshold = 0.1
        if stats[0] > threshold:
            self.posterior = self.prior.newPosterior(stats)
