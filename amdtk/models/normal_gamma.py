
""" Normal-Gamma prior for a Gaussian with diagonal covariance."""

import numpy as np
from scipy.special import gammaln, psi


class NormalGamma(object):
    """NormalGamma density."""


    def __init__(self, mean, kappa, alpha, beta):
        self.mean = mean
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

    def expected_log_precision(self):
        '''Expected value of the logarithm of the precision.

        Returns
        -------
        E_log_prec : numpy.ndarray
            Log precision.
        '''
        return psi(self.alpha) - np.log(self.beta)

    def expected_precision(self):
        """Expected value of the precision.

        Returns
        -------
        E_prec : numpy.ndarray
            Precision.
        """
        return self.alpha/self.beta

    def kl_divergence(self, prob):
        """KL divergence between the current and the given densities.

        Parameters
        ----------
        prob : :class:`NormalGamma`
            NormalGamma density to compute the divergence with.

        Returns
        -------
        KL : float
            KL divergence.

        """
        exp_lambda = self.expected_precision()
        exp_log_lambda = self.expected_log_precision()

        kl_div = .5 * (np.log(self.kappa) - np.log(prob.kappa))
        kl_div -= .5 * (1 - prob.kappa * \
            (1./self.kappa + exp_lambda * (self.mean - prob.mean)**2))
        kl_div += gammaln(prob.alpha) - gammaln(self.alpha)
        kl_div += self.alpha * np.log(self.beta) - prob.alpha * \
            np.log(prob.beta)
        kl_div += exp_log_lambda * (self.alpha - prob.alpha)
        kl_div -= exp_lambda * (self.beta - prob.beta)

        return kl_div.sum()


    def new_posterior(self, stats):
        """Create a new Normal-Gamma density given the parameters of the
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
        kappa_n = self.kappa + stats[0]
        mean_n = (self.kappa * self.mean + stats[1]) / kappa_n
        alpha_n = self.alpha + .5 * stats[0]
        value = (self.kappa * self.mean + stats[1])**2
        value /= (self.kappa + stats[0])
        beta_n = self.beta + 0.5*(-value + stats[2] + self.kappa * \
            self.mean**2)

        return NormalGamma(mean_n, kappa_n, alpha_n, beta_n)

