
"""HMM implementation."""

import numpy as np
from scipy.misc import logsumexp
from .model import Model
from .model import VBModel
from .model import DiscreteLatentModel
from .model import MissingModelParameterError
from .model import InvalidModelError
from .model import InvalidModelParameterError
from .model import DiscreteLatentModelEmptyListError
from .graph import Graph


class LeftToRightHMM(Model, VBModel, DiscreteLatentModel):
    """Implementation of a left-to-right HMM.

    """
    hmm_count = 0

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
        cls.hmm_count += 1
        params['name'] = config['prefix_name'] + str(cls.hmm_count)
        nstates = config.getint('nstates')
        comps = [Model.create(config['emission'], data) for i in
                 range(nstates)]
        params['nstates'] = nstates
        params['emissions'] = comps
        return params

    def __init__(self, params):
        """Initialize the model.

        Parameters
        ----------
        params : dict
            Dictionary containing:
              * name: str
              * nstates: int
              * emissions: List of Model objects for each state

        """
        super().__init__(params)
        missing_param = None
        try:
            # Empty statement to make sure the components are defined.
            self.components

            if not isinstance(params['name'], str):
                raise InvalidModelParameterError(self, 'name', self.name)
            if not isinstance(self.nstates, int):
                raise InvalidModelParameterError(self, 'nstates',
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

        self._build(params['name'])

    def _build(self, name):
        self.graph = Graph(name)
        previous_state = None
        for i in range(self.nstates):
            state_name = name + '_' + str(i+1)
            state = self.graph.addState(state_name, self.components[i])
            self.graph.addLink(state, state, 0.)
            if i == 0:
                self.graph.setInitState(state)
            if i == self.nstates - 1:
                self.graph.setFinalState(state, 0.)
            if previous_state is not None:
                self.graph.addLink(previous_state, state, 0.)
            previous_state = state

        self.graph.setUniformProbInitStates()
        self.graph.normalize()

    @property
    def nstates(self):
        return self.params['nstates']

    @property
    def components(self):
        return self.params['emissions']

    def prior(self):
        return None

    def posterior(self):
        return None

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
            Weights to apply when building the stats.
        model_id : int
            Use the specified model_id to store the statistics.

        """
        resps, states_data = data
        for i, state in enumerate(self.graph.sorted_states):
            model = self.components[i]
            model.stats(stats, X, states_data[model.uuid], resps[:, i],
                        self.uuid)

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
            The expected value of the log-likelihood.
        data : object
            Data needed by the model to evaluate the statistics for the
            VB update.

        """
        E_log_p_X_given_Z = np.zeros((X.shape[0], len(self.graph.states)))
        data = {}
        for i, state in enumerate(self.graph.sorted_states):
            model = self.components[i]
            llh, state_data = model.expectedLogLikelihood(X, weight)
            E_log_p_X_given_Z[:, i] = llh
            data[model.uuid] = state_data

        log_alphas = self.graph.forward(E_log_p_X_given_Z)
        log_betas = self.graph.backward(E_log_p_X_given_Z)
        E_llh = logsumexp(log_alphas[-1])
        log_P_Z = log_alphas + log_betas
        log_resps = (log_P_Z.T - logsumexp(log_P_Z, axis=1)).T
        return E_llh, (np.exp(log_resps), data)

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns
        -------
        KL : float
            KL divergence.

        """
        KL = 0
        for model in self.components:
            KL += model.posterior.KL(model.prior)
        return KL

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
        for model in self.components:
            model.updatePosterior(stats)
