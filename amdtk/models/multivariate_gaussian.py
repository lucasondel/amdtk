
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

    @classmethod
    def loadParams(cls, config, data):
        """Load the parameters of the model.

        Parameters
        ----------
        config : dict like
            Dictionary like object containing specific values of the
            model.
        data : dict
            Extra data that may be used for initializing the model.

        Returns
        -------
        params : dict
            Dictioanry of the model's parameters.

        """
        params = {}
        params['prior'] = Model.create(config['prior'], data)
        params['posterior'] = Model.create(config['prior'], data)
        return params

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

    def stats(self, stats, X, data, weights):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        stats : dict
            Dictionary where to store the stats for each model.
        X : numpy.ndarray
            Data on which to accumulate the stats.
        data : dict
            Ignored.
        weights : numpy.ndarray
            Weights to apply per frame.
        model_id : int
            Use the specified model_id to store the statistics.

        Returns
        -------
        stats : dict
            Dictionary containing the mapping model_id -> stats.

        """
        self.posterior.stats(stats, X, data, weights)

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
        if self.posterior.uuid in stats:
            self.posterior = self.prior.newPosterior(
                stats[self.posterior.uuid])
    
    def gradUpdatePosterior(self, stats, lrate, total_nframes, grad_nframes):
        """Gradient update the parameters of the posterior density given 
        the accumulated statistics.

        Parameters
        ----------
        stats : obj
            Accumulated sufficient statistics for the update.
        lrate : float
            Scale of the gradient.
        total_nframes : int
            Number of frames for the whole training set.
        grad_nframes : int
            Number of frames used to compute the gradient.

        Returns
        -------
        post : :class:`Prior`
            New posterior density/distribution.

        """
        if self.posterior.uuid in stats:
            self.posterior = self.prior.newPosteriorFromGrad(
                stats[self.posterior.uuid], self.posterior, lrate, 
                total_nframes, grad_nframes)

