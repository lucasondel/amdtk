
"""HMM implementation."""

import numpy as np
from .model import Model
from .model import VBModel
from .model import DiscreteLatentModel
from .model import MissingModelParameterError
from .model import InvalidModelError
from .model import InvalidModelParameterError
from .model import DiscreteLatentModelEmptyListError
from .graph import Graph


class LeftToRightHMM(Model, DiscreteLatentModel):
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

    @property
    def nstates(self):
        return self.params['nstates']

    @property
    def components(self):
        return self.params['emissions']

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
                self.graph.setFinalState(state)
            if previous_state is not None:
                self.graph.addLink(previous_state, state, 0.)
            previous_state = state

        self.graph.normalize()

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
            Weights to apply per frame.

        Returns
        -------
        stats : dict
            Dictionary containing the mapping model_id -> stats.

        """
        raise NotImplementedError()
