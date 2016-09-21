
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
        self.final_weight = float('-inf')

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
        up to one (when exponentiated). If the state is a final state
        then we normalize assuming an implicit extra arc going to an
        abstract final state.

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
        log_norm = np.logaddexp(log_norm, self.final_weight)
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

    @property
    def models(self):
        ms = set()
        for state in self.sorted_states:
            ms.add(state.model)
        return ms

    def addGraph(self, graph, weight=0.):
        """Merge the given graph into the current graph.

        Parameters
        ----------
        graph : :class:`Graph`
            Graph object to merge.
        weight : float
            Log probability of going from one graph to the other. Previous
            weights will be erased.

        """
        self.states = {**self.states, **graph.states}
        self.sorted_states = \
            sorted([state for _, state in self.states.items()],
                   key=lambda state: state.name)
        self.init_states = set.union(self.init_states, graph.init_states)
        self.final_states = set.union(self.final_states, graph.final_states)

        for src_uuid in self.final_states:
            for dest_uuid in self.init_states:
                src = self.states[src_uuid]
                dest = self.states[dest_uuid]
                self.addLink(src, dest, weight)

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

    def setFinalState(self, state, final_weight=-0.6931471805599453):
        """Mark the state as a possible final state.

        Parameters
        ----------
        state : :class:`State`
            New final state.
        final_weight : float
            Probability to finish (default: log(0.5))

        """
        state.final_weight = final_weight
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
        for state_uuid in self.init_states:
            self.logProbInit[state_uuid] = state_log_pi

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

    def evaluateEmissions(self, X, ac_weight=1.0):
        """Evalue the (expected value) of the log-likelihood of each
        emission probability model.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.
        ac_weight : float
            Scaling of the acoustic scores.

        Returns
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood for each frame.
        log_p_Z ; numpy.ndarray
            Log probability of the discrete latent variables for each
            model.

        """
        E_log_p_X_given_Z = np.zeros((X.shape[0], len(self.states)))
        data = []
        for i, state in enumerate(self.sorted_states):
            llh, state_data = state.model.expLogLikelihood(X, ac_weight)
            E_log_p_X_given_Z[:, i] = llh
            data.append(state_data)

        return E_log_p_X_given_Z, data

    def forward(self, llhs):
        """Forward recursion given the log emission probabilities and
        the HMM.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM.

        Returns
        -------
        log_alphas : numpy.ndarray
            The log alphas values of the recursions.

        """
        # Initial probabilities
        log_pi = np.zeros(len(self.sorted_states)) + float('-inf')
        for i, state in enumerate(self.sorted_states):
            try:
                log_pi[i] = self.logProbInit[state.uuid]
            except KeyError:
                pass

        log_alphas = llhs.copy()
        log_alphas[0] += log_pi
        log_A = self.getLogProbTrans()
        for t in range(1, len(llhs)):
            log_alphas[t] += logsumexp(log_A.T + log_alphas[t - 1], axis=1)

        return log_alphas

    def backward(self, llhs):
        """Backward recursion given the log emission probabilities and
        the HMM.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM.

        Returns
        -------
        log_betas : numpy.ndarray
            The log alphas values of the recursions.

        """
        # Index of the final states.
        final_indices = []
        for state_uuid in self.final_states:
            final_indices.append(self.sorted_states.index(
                self.states[state_uuid]))

        log_betas = np.zeros_like(llhs) + float('-inf')
        log_betas[-1, final_indices] = 0.
        log_A = self.getLogProbTrans()
        for t in reversed(range(llhs.shape[0]-1)):
            log_betas[t] = logsumexp(log_A + llhs[t + 1] +
                                     log_betas[t + 1], axis=1)
        return log_betas
