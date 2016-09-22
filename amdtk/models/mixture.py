
"""Mixture of distributions/densities."""

import numpy as np
from scipy.misc import logsumexp
from .model import Model
from .model import VBModel
from .model import DiscreteLatentModel
from .model import MissingModelParameterError
from .model import InvalidModelError
from .model import InvalidModelParameterError
from .model import DiscreteLatentModelEmptyListError
from .dirichlet import Dirichlet
from .dirichlet_process import DirichletProcess

accepted_priors = [Dirichlet, DirichletProcess]


class BayesianMixture(Model, VBModel, DiscreteLatentModel):
    """Bayesian mixture of probability distributions/densities.


    Attributes
    ----------
    prior : :class:`Dirichlet`
        Prior density.
    posterior : :class:`Dirichlet`
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
        ncomponents = len(params['prior'].expectedLogX())
        comps = [Model.create(config['component'], data) for i in
                 range(ncomponents)]
        params['components'] = comps
        return params

    def __init__(self, params):
        """Initialize a Bayesian Mixture model.

        Parameters
        ----------
        params : dict
            Dictionary containing:
              * prior: class:`Dirichlet`
              * posterior: :class:`Dirichlet`
              * components: List of VBModel objects

        """
        super().__init__(params)
        missing_param = None
        try:
            # Empty statement to make sure the components are defined.
            self.components

            if self.prior.__class__ not in accepted_priors:
                raise InvalidModelParameterError(self, 'prior', self.prior)
            if self.posterior.__class__ not in accepted_priors:
                raise InvalidModelParameterError(self, 'posterior',
                                                 self.posterior)
        except KeyError as e:
            missing_param = e.__str__()

        if missing_param is not None:
            raise MissingModelParameterError(self, missing_param)

        if self.k == 0:
            raise DiscreteLatentModelEmptyListError(self)

        for component in self.components:
            if not isinstance(component, VBModel):
                raise InvalidModelError(component, VBModel)

    @property
    def prior(self):
        return self.params['prior']

    @property
    def posterior(self):
        return self.params['posterior']

    @posterior.setter
    def posterior(self, new_posterior):
        self.params['posterior'] = new_posterior

    @property
    def components(self):
        return self.params['components']

    def stats(self, stats, X, data, weights, model_id=None):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        stats : dict
            Dictionary where to store the stats for each model.
        X : numpy.ndarray
            Data on which to accumulate the stats.
        data : dict
            Data specific to each sub-model with to use to accumulate
            the stats.
        weights : numpy.ndarray
            Weights to apply per frame.

        Returns
        -------
        stats : dict
            Dictionary containing the mapping model_id -> stats.

        """
        E_P_Z, g_data = data
        self.posterior.stats(stats, X, E_P_Z, weights)
        for i, component in enumerate(self.components):
            comp_weights = weights * E_P_Z[:, i]
            component.stats(stats, X, g_data[component.uuid], comp_weights)

    def expectedLogLikelihood(self, X, weight=1.0):
        """Expected value of the log-likelihood of the data given the
        model.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.
        weight : float
            Scaling weight for the log-likelihood

        Returns
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood for each frame.
        E_log_P_Z: numpy.ndarray
            Probability distribution of the latent states given the
            data.

        """
        E_log_weights = self.posterior.expectedLogX()
        E_log_p_X = np.zeros((X.shape[0], self.k))
        g_data = {}
        for i, pdf in enumerate(self.components):
            c_llh, g_data[pdf.uuid] = pdf.expectedLogLikelihood(X)
            E_log_p_X[:, i] += E_log_weights[i]
            E_log_p_X[:, i] += c_llh
            E_log_p_X[:, i] *= weight
        log_norm = logsumexp(E_log_p_X, axis=1)
        E_log_P_Z = (E_log_p_X.T - log_norm).T
        return log_norm.sum(), (np.exp(E_log_P_Z), g_data)

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

    def updatePosterior(self, stats):
        """Update the parameters of the posterior distribution.

        Parameters
        ----------
        stats : dict
            Dictionary of stats.

        """
        if self.posterior.uuid in stats:
            self.posterior = self.prior.newPosterior(
                stats[self.posterior.uuid])
        for component in self.components:
            component.updatePosterior(stats)
