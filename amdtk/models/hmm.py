
"""HMM implementation."""

import abc
import numpy as np
from scipy.misc import logsumexp
from .prior import EmptyPrior
from .model import Model
from .model import VBModel
from .model import DiscreteLatentModel
from .model import MissingModelParameterError
from .model import InvalidModelError
from .model import InvalidModelParameterError
from .model import DiscreteLatentModelEmptyListError
from .graph import Graph
from .dirichlet import Dirichlet
from .dirichlet_process import DirichletProcess


class HMM(Model, VBModel, DiscreteLatentModel):
    """Base class for HMM models.

    """

    def __init__(self, params):
        super().__init__(params)

    @abc.abstractmethod
    def build(self):
        """Build the specific structure of the HMM."""
        pass

    @property
    def components(self):
        return self.params['emissions']

    @abc.abstractproperty
    def prior(self):
        return None

    @abc.abstractproperty
    def posterior(self):
        return None

    def stats(self, stats, X, data, weights):
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
        self.posterior.stats(stats, X, resps, weights)
        for i, state in enumerate(self.graph.sorted_states):
            model = self.components[i]
            model.stats(stats, X, states_data[model.uuid], resps[:, i])

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
        KL = self.posterior.KL(self.prior)
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
        if self.posterior.uuid in stats:
            self.posterior = self.prior.newPosterior(
                stats[self.posterior.uuid])
        for model in self.components:
            model.updatePosterior(stats)


class LeftToRightHMM(HMM):
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
                                                 self.nstates)
        except KeyError as e:
            missing_param = e.__str__()

        if missing_param is not None:
            raise MissingModelParameterError(self, missing_param)

        if self.k == 0:
            raise DiscreteLatentModelEmptyListError(self)

        for component in self.components:
            if not isinstance(component, VBModel):
                raise InvalidModelError(component, VBModel)

        self._prior = EmptyPrior({})
        self.build()

    @property
    def prior(self):
        return self._prior

    @property
    def posterior(self):
        return self._prior

    @posterior.setter
    def posterior(self, new_posterior):
        self._prior = new_posterior

    @property
    def name(self):
        return self.params['name']

    @property
    def nstates(self):
        return self.params['nstates']

    def build(self):
        self.graph = Graph(self.name)
        previous_state = None
        for i in range(self.nstates):
            state_name = self.name + '_' + str(i+1)
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


accepted_priors = [Dirichlet, DirichletProcess]


class BayesianPhoneLoop(HMM):
    """Implementation of a left-to-right HMM.

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
        nunits = config.getint('nunits', 100)
        subhmms = [Model.create(config['subhmms'], data) for i in
                   range(nunits)]
        params['nunits'] = nunits
        params['subhmms'] = subhmms
        components = []
        for hmm in subhmms:
            for component in hmm.components:
                components.append(component)
        params['emissions'] = components
        return params

    def __init__(self, params):
        """Initialize the model.

        Parameters
        ----------
        params : dict
            Dictionary containing:
              * nunits: int
              * prior : :class:`Prior`
              * posterior : :class:`Prior`
              * subhmms: List of Model objects for each state

        """
        super().__init__(params)
        missing_param = None
        try:
            # Empty statement to make sure the components are defined.
            self.subhmms
            self.components

            if not isinstance(self.nunits, int):
                raise InvalidModelParameterError(self, 'nunits',
                                                 self.nunits)
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

        self.build()

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
    def subhmms(self):
        return self.params['subhmms']

    @property
    def nunits(self):
        return self.params['nunits']

    def build(self):
        self.graph = Graph('phone_loop')
        for hmm in self.subhmms:
            self.graph.addGraph(hmm.graph, 0.)

        self.graph.setUniformProbInitStates()
        self.graph.normalize()
        self.setUnitTransitions()

    def setUnitTransitions(self):
        log_pi = self.posterior.expectedLogX()
        log_pi -= logsumexp(log_pi)
        sorted_names = [hmm.name for hmm in self.subhmms]
        for src_uuid in self.graph.final_states:
            for i, dest_uuid in enumerate(self.graph.init_states):
                src = self.graph.states[src_uuid]
                dest = self.graph.states[dest_uuid]
                unit_name = src.name.split('_')[0]
                idx = sorted_names.index(unit_name)
                self.graph.addLink(src, dest, log_pi[idx])

    def updatePosterior(self, stats):
        super().updatePosterior(stats)
        self.setUnitTransitions()
        self.graph.normalize()