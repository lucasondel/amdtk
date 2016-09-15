
"""Multivariate Gaussian density."""

import numpy as np
from .model import Model
from .model import VBModel
from .model import InvalidModelParameterError
from .model import MissingModelParameterError
from .normal_gamma import NormalGamma


class BayesianGaussianDiagCov(Model, VBModel):
    """Bayesian multivariate Gaussian with a diagonal covariance matrix.

    The prior over the mean and variance for each dimension is a
    Normal-Gamma density.

    Attributes
    ----------
    prior : :class:`Prior`
        Prior density.
    posterior : :class:`Prior`
        Posterior density.

    """

    def __init__(self, params):
        """Initialize a NormalGamma Prior.

        Parameters
        ----------
        params : dict
            Dictionary containing:
              * prior: :class:`NormalGamma`
              * posterior: :class:`NormalGamma`

        """
        super().__init__(params)
        missing_param = None
        try:
            if not isinstance(self.prior, NormalGamma):
                raise InvalidModelParameterError(self, 'prior', self.prior)
            if not isinstance(self.posterior, NormalGamma):
                raise InvalidModelParameterError(self, 'posterior',
                                                 self.posterior)
        except KeyError as e:
            missing_param = e.__str__()

        if missing_param is not None:
            raise MissingModelParameterError(self, missing_param)

    @property
    def prior(self):
        return self.params['prior']

    @property
    def posterior(self):
        return self.params['posterior']

    @posterior.setter
    def posterior(self, new_posterior):
        self.params['posterior'] = new_posterior

    def expectedLogLikelihood(self, X, weight=1.0):
        """Expected value of the log-likelihood of X.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.
        weight : float
            Weight to apply to the expected log-likelihood.

        Returns
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood for each frame.
        data : object
            Data needed by the model to evaluate the statistics for the
            VB update.

        """
        _, log_precision = self.posterior.expectedLogX()
        m, precision = self.posterior.expectedX()
        log_norm = (log_precision - 1 / self.posterior.kappa).sum()
        log_norm -= np.log(2 * np.pi)
        E_llh = .5 * (log_norm - (precision * (X - m)**2).sum(axis=1))
        return weight * E_llh, X

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns
        -------
        KL : float
            KL divergence.

        """
        return super().KLPosteriorPrior()

    def updatePosterior(self, stats):
        """Update the parameters of the posterior density given the
        accumulated statistics.

        Parameters
        ----------
        stats : obj
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:`Prior`
            New posterior density/distribution.

        """
        super().KLPosteriorPrior()