
"""Gaussian distribution with a GaussianWishart prior."""

import numpy as np
from scipy.special import psi, gammaln
from .model import Model


class GaussianDiagCov(Model):
    """Bayesian multivariate Gaussian with a diagonal covariance matrix.
    The prior over the mean and variance for each dimension is a
    Normal-Gamma density.

    """

    # pylint: disable=too-many-instance-attributes
    # The number of instance attributes could be reduced
    # by creating a specific class NormalGamma. However, this
    # class would be used only by "GaussianDiagCov". We
    # prefer to add more attributes to the current class
    # rather than creating a new class.

    def __init__(self, mean, mcount, prec, pcount):
        """Initializat the Gaussian.

        Parameters
        ----------
        mean : numpy.ndarray
            Prior mean.
        mcount : float
            Prior count for the mean.
        prec : numpy.ndarray
            Prior precision for each dimension.
        pcount : float
            Prior counts for the precision matrix.

        """
        super().__init__()
        self.prior_mean = mean
        self.prior_mcount = mcount
        self.posterior_mean = mean
        self.posterior_mcount = mcount
        self.prior_prec = prec
        self.prior_pcount = pcount
        self.posterior_prec = prec
        self.posterior_pcount = pcount

    @staticmethod
    def get_stats(data, weights):
        """Compute the sufficient statistics of the model
        given the data and the weights for each frame.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.
        weights : numpy.ndarray
            Weights for each frame.

        Returns
        -------
        stats : dict
            Dictionary where 's0', 's1' and 's2' are the keys for the
            zeroth, first and second order statistics respectively.

        """
        stats_0 = float(weights.sum())
        w_data = (weights * data.T).T
        stats_1 = w_data.sum(axis=0)
        stats_2 = (w_data * data).sum(axis=0)
        stats_data = {
            's0': stats_0,
            's1': stats_1,
            's2': stats_2
        }
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
        expected_log_prec = psi(self.posterior_pcount) \
            - np.log(self.posterior_prec)
        log_norm = (expected_log_prec - 1 / self.posterior_mcount)
        return .5 * (log_norm - expected_log_prec * \
                     (data - self.posterior_mean)**2).sum(axis=1)

    def update(self, stats):
        """ Update the posterior parameters given the sufficient
        statistics.

        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.

        """
        stats_0 = stats[self.id]['s0']
        stats_1 = stats[self.id]['s1']
        stats_2 = stats[self.id]['s2']
        self.posterior_mcount = self.prior_mcount + stats_0
        self.posterior_mean = (self.prior_mcount * self.prior_mean + stats_1)
        self.posterior_mean /= (self.prior_mcount + stats_0)
        self.posterior_pcount = self.prior_pcount + .5 * stats_0
        tmp = (self.prior_mcount * self.prior_mean + stats_1)**2
        tmp /= self.prior_mcount + stats_0
        self.posterior_prec = self.prior_prec
        self.posterior_prec += 0.5 * (-tmp + stats_2 + self.prior_mcount * \
                                      self.prior_mean**2)

    def kl_divergence(self):
        """Kullback-Leibler divergence between the posterior and
        the prior density.

        Returns
        -------
        kl_div : float
            KL(q(params) || p(params)).

        """
        expected_log_prec = psi(self.posterior_pcount) - \
            np.log(self.posterior_prec)
        expected_prec = self.posterior_pcount / self.posterior_prec
        kl_div = .5 * (np.log(self.posterior_mcount) - \
                       np.log(self.prior_mcount))
        kl_div -= .5 * (1 - self.prior_mcount * \
                        (1./self.posterior_mcount + \
                         expected_prec * \
                         (self.posterior_mean - self.prior_mean)**2))
        kl_div += gammaln(self.prior_pcount) - gammaln(self.posterior_pcount)
        kl_div += self.posterior_pcount * np.log(self.posterior_prec) - \
            self.prior_pcount * np.log(self.prior_prec)
        kl_div += expected_log_prec * (self.posterior_pcount - \
                                       self.prior_pcount)
        kl_div -= expected_prec * (self.posterior_prec - self.prior_prec)
        return  kl_div.sum()
