
""" Normal-Gamma density."""

import numpy as np
from scipy.special import gammaln, psi


class NormalGamma(object):
    """Normal-Gamma density.

    Attributes
    ----------
    mu : numpy.ndarray
        Mean of the Gaussian density.
    kappa : float
        Factor of the precision matrix.
    alpha : float
        Shape parameter of the Gamma density.
    beta : numpy.ndarray
        Rate parameters of the Gamma density.

    Methods
    -------
    expLogPrecision()
        Expected value of the logarithm of the precision.
    expPrecision()
        Expected value of the precision.
    KL(self, pdf)
        KL divergence between the current and the given densities.
    newPosterior(self, stats)
        Create a new Normal-Gamma density.
    """

    def __init__(self, mu, kappa, alpha, beta):
        self.mu = mu
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

    def expLogPrecision(self):
        '''Expected value of the logarithm of the precision.

        Returns
        -------
        E_log_prec : numpy.ndarray
            Log precision.
        '''
        return psi(self.alpha) - np.log(self.beta)

    def expPrecision(self):
        """Expected value of the precision.

        Returns
        -------
        E_prec : numpy.ndarray
            Precision.
        """
        return self.alpha/self.beta

    def KL(self, q):
        """KL divergence between the current and the given densities.

        Returns
        -------
        KL : float
            KL divergence.

        """
        p = self

        exp_lambda = p.expPrecision()
        exp_log_lambda = p.expLogPrecision()

        return (.5 * (np.log(p.kappa) - np.log(q.kappa))
                - .5 * (1 - q.kappa * (1./p.kappa + exp_lambda * (p.mu - q.mu)**2))
                - (gammaln(p.alpha) - gammaln(q.alpha))
                + (p.alpha * np.log(p.beta) - q.alpha * np.log(q.beta))
                + exp_log_lambda * (p.alpha - q.alpha)
                - exp_lambda * (p.beta - q.beta)).sum()

    def newPosterior(self, stats):
        """Create a new Normal-Gamma density.

        Create a new Normal-Gamma density given the parameters of the
        current model and the statistics provided.

        Parameters
        ----------
        stats : :class:MultivariateGaussianDiagCovStats
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:NormalGamma
            New Dirichlet density.

        """
        # stats[0]: counts
        # stats[1]: sum(x)
        # stats[2]: sum(x**2)
        kappa_n = self.kappa + stats[0]
        mu_n = (self.kappa * self.mu + stats[1]) / kappa_n
        alpha_n = self.alpha + .5 * stats[0]
        v = (self.kappa * self.mu + stats[1])**2
        v /= (self.kappa + stats[0])
        beta_n = self.beta + 0.5*(-v + stats[2] + self.kappa * self.mu**2)

        return NormalGamma(mu_n, kappa_n, alpha_n, beta_n)
