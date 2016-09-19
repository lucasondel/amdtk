
import numpy as np
import uuid
from scipy.misc import logsumexp


class State(object):
    """HMM state.

    Attributes
    ----------
    uuid : int
        Unique identifier of the state.
    name : str
        Label of the state.
    next_states : dict
        Mapping from state_id -> weight containing all the outgoing arcs
        of the state.
    previous_states : dict
        Mapping from state_id -> weight containing all the incoming arcs
        of the state.
    model_id : int
        Id of the model associated to the state.

    """

    def __init__(self, name, emission):
        """Hmm state object.

        Parameters
        ----------
        name : str
            String identifier of the state.
        emission : :class:`Model`

        """
        self.__uuid = uuid.uuid4().int >> 64
        self.name = name
        self.next_states = {}
        self.previous_states = {}
        self.emission = emission

    @property
    def uuid(self):
        """Unique identifier of the state."""
        return self.__uuid

    def addLink(self, state, weight):
        """Add a link starting from the current state and going to
        "state".

        A state cannot have two links connecting the same
        state. When called again with the same "state". The old
        link will be removed.

        Parameters
        ----------
        state : :class:`State`
            State where the link points to.
        weight : float
            Log probability of going from the current state to "state".

        """
        self.next_states[state.uuid] = weight
        state.previous_states[self.uuid] = weight

    def normalize(self, states):
        """Normalize the weights of the outgoing links so that they sum
        up to one (when exponentiated).

        Parameters
        ----------
        states : dict
            Mapping uuid -> state.

        """
        if len(self.next_states) == 0:
            return
        log_weights = np.zeros(len(self.next_states))
        for i, uuid in enumerate(self.next_states):
            log_weights[i] = self.next_states[uuid]
        log_norm = logsumexp(log_weights)
        for i, uuid in enumerate(self.next_states):
            state = states[uuid]
            self.next_states[uuid] -= log_norm
            state.previous_states[self.uuid] = self.next_states[uuid]


class Graph(object):

    def __init__(self, name):
        """Initialize an HMM."""
        self.name = name
        self.states = {}
        self.sorted_states = []
        self.init_states = set()
        self.final_states = set()
        self.logProbInit = {}

    @property
    def states_names(self):
        return [state.names for state in self.sorted_states]

    def addState(self, name, emission):
        """Add a state to the HMM graph.

        Parameters
        ----------
        name : str
            Name of the state.
        emission : :class:`Model`

        Returns
        -------
        state : :class:`State`
            The state created.

        """
        state = State(name, emission)
        self.states[state.uuid] = state
        self.sorted_states = \
            sorted([state for _, state in self.states.items()],
                   key=lambda state: state.name)
        return state

    def setInitState(self, state):
        """Mark the state as a possible initial state.

        Parameters
        ----------
        state : :class:`State`
            New initial state.

        """
        self.init_states.add(state.uuid)

    def setFinalState(self, state):
        """Mark the state as a possible final state.

        Parameters
        ----------
        state : :class:`State`
            New final state.

        """
        self.final_states.add(state.uuid)

    def addLink(self, state, next_state, weight=0.):
        """Add a new directed link between two states.

        Parameters
        ----------
        state : :class:`State`
            New source state.
        next_state : :class:`State`
            Destination state.
        weight : float
            Log probability to move from "state" to "next_state".

        """
        state.addLink(next_state, weight)

    def setUniformProbInitStates(self):
        """Set the probability of the initial state to a flat
        distribution.

        """
        state_log_pi = np.log(1 / len(self.init_states))
        self.logProbInit = {}
        for state in self.init_states:
            self.logProbInit[state] = state_log_pi

    def normalize(self):
        """Normalize the transitions probabilities.

        """
        for state_uuid, state in self.states.items():
            state.normalize(self.states)

    def getLogProbTrans(self):
        """Logarithm of the transitions probabilit matrix.

        Returns
        -------
        log_A : numpy.ndarray
            Log porbability matrix.
        """
        nstates = len(self.states)
        retval = np.zeros((nstates, nstates)) + float('-inf')
        for i in range(nstates):
            state = self.sorted_states[i]
            for j in range(nstates):
                next_state = self.sorted_states[j]
                if next_state.uuid in state.next_states:
                    retval[i, j] = state.next_states[next_state.uuid]
        return retval
