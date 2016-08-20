
import numpy as np
from scipy.misc import logsumexp
import pyximport

pyximport.install(setup_args={"include_dirs":np.get_include()})

from .hmm_graph_utils import _fast_logsumexp_axis1


class State(object):
    """HMM state.

    Attributes
    ----------
    state_id : int
        Unique identifier of the state.
    name : str
        Label of the state.
    parent_name : str
        Name of the HMM structure embedding this state.
    next_states : dict
        Mapping from state_id -> weight containing all the outgoing arcs
        of the state.
    previous_states : dict
        Mapping from state_id -> weight containing all the incoming arcs
        of the state.
    model : object
        Model of the emission probability of the state. Some model can
        be shared by several states.

    """

    def __init__(self, state_id, name, parent_name=None):
        """Hmm state object.

        Parameters
        ----------
        state_id : int
            Identifier of the state.
        name : str
            String identifier of the state.
        parent_name : str
            String identifier of the state. This field is usually used
            for HMM embedding sub-HMM. It allows to keep track of the
            name of the sub-HMM.

        """
        self.state_id = state_id
        self.name = name
        self.parent_name = parent_name
        self.next_states = {}
        self.previous_states = {}
        self.model = None

    def addLink(self, state, weight):
        """Add a link starting from the current state and going to
        "state".

        A state cannot have two links connecting the same
        state. When called again with the same "state". The previous
        link will be removed.

        Parameters
        ----------
        state : :class:`State`
            State where the link points to.
        weight : float
            Log probability of going from the current state to "state".

        """
        self.next_states[state.state_id] = weight
        state.previous_states[self.state_id] = weight


class HmmGraph(object):
    """HMM model.

    Attributes
    ----------
    states : list
        List of states in the HMM.
    id_state : dict
        Mapping identifier - state.
    init_states : list
        List of initial states of the HMM.
    final_states : list
        List of final states of the HMM.
    names : tuple
        A tuple containing the list of each state's parent name and each
        state's name. Both lists are of the same size even if some units
        share the same "parent_name" attribute.

    """

    state_count = 0

    @classmethod
    def _nextStateId(cls):
        cls.state_count += 1
        return cls.state_count

    @classmethod
    def selfLoop(cls, name):
        """Create a HMM composed of a single state looping on itself.

        Parameters
        ----------
        name : str
            Name of the model.

        Returns
        -------
        hmm : :class:`HmmGraph`
            An initialized HMM.

        """
        graph = cls()
        state = graph.addState(name, parent_name=name)
        graph.addLink(state, state)
        graph.init_states = graph.states
        graph.final_states = graph.states
        graph._prepare()
        return graph

    @classmethod
    def leftToRightHmm(cls, name, nstates):
        """Create a left-to-rifht HMM where each state is ooping on
        itself.

        Parameters
        ----------
        name : str
            Name of the model.
        nstates : int
            Number of states in the HMM.

        Returns
        -------
        hmm : :class:`HmmGraph`
            An initialized HMM.

        """
        graph = cls()

        for i in range(nstates):
            state_name = name + '_' + str(i+1)
            state = graph.addState(state_name, parent_name=name)
            if i == 0:
                graph.setInitState(state)
            if i == nstates - 1:
                graph.setFinalState(state)

        for i, state in enumerate(graph.states):
            graph.addLink(state, state)
            if state not in graph.final_states:
                next_state = graph.states[i + 1]
                graph.addLink(state, next_state)

        graph._prepare()

        return graph

    @classmethod
    def standardPhoneLoop(cls, prefix, nunits, nstates):
        """Create the HMM corresponding to a unit-loop model.

        Each unit is represented by a left-to-right hmm.

        Parameters
        ----------
        prefix : str
            Prefix for the name of the units. The unit will be named
            "prefix + (n)" where "n" is the number of the unit
            (starting from 1).
        nunits : int
            Number of units in the phone loop.

        nstates : int
            Number of state in each left-to-right HMM representing a
            single unit.

        Returns
        -------
        hmm : :class:`HmmGraph`
            An initialized HMM.

        """
        graph = cls()

        for i in range(nunits):
            unit_name = prefix + str(i+1)
            hmm_graph = cls.leftToRightHmm(unit_name, nstates)
            graph.id_state = {**graph.id_state, **hmm_graph.id_state}
            graph.states += hmm_graph.states
            graph.init_states += hmm_graph.init_states
            graph.final_states += hmm_graph.final_states

        for final_state in graph.final_states:
            for init_state in graph.init_states:
                graph.addLink(final_state, init_state)

        graph._prepare()

        return graph

    def __init__(self):
        """Initialize an HMM."""
        self.states = []
        self.id_state = {}
        self.init_states = []
        self.final_states = []

    @property
    def names(self):
        state_names = [state.name for state in self.states]
        parent_names = [name.split('_')[0] for name in state_names]
        return parent_names, state_names

    def _computelogProbInitStates(self):
        state_log_pi = 1 / len(self.init_states)
        self._state_log_pi = {}
        for state in self.init_states:
            self._state_log_pi[state] = state_log_pi

    def _normalize(self):
        for state in self.states:
            next_states = []
            weights = np.zeros(len(state.next_states))
            for i, next_state_id in enumerate(state.next_states):
                next_states.append(next_state_id)
                weights[i] = state.next_states[next_state_id]
            weights -= logsumexp(weights)
            for i, next_state_id in enumerate(next_states):
                state.next_states[next_state_id] = weights[i]
                next_state = self.id_state[next_state_id]
                next_state.previous_states[state.state_id] = weights[i]

    def _prepare(self):
        self._normalize()
        self._computelogProbInitStates()
        self._log_pi = self.logProbInit()
        self._log_A = self.logProbTrans()
        self._final_state_idx = []
        self._final_state_idx = [self.states.index(state) 
                                 for state in self.final_states]

    def setEmissions(self, name_model):
        """Associate an emission probability model for each state of the
        HMM.

        Parameters
        ----------
        name_model : dict
            Mapping "state_name" -> model object.

        """
        for state in self.states:
            state.model = name_model[state.name]

    def evaluateEmissions(self, X):
        """Evalue the (expected value of the )log-likelihood of each
        emission probability model.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.

        Returns
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood for each frame.
        log_p_Z ; numpy.ndarray
            Log probability of the discrete latent variables for each
            model.

        """
        E_log_p_X_given_Z = np.zeros((X.shape[0], len(self.states)),
                                     dtype=np.float32)
        log_resps = []
        for i, state in enumerate(self.states):
            llh, log_resp = state.model.expLogLikelihood(X)
            E_log_p_X_given_Z[:, i] = llh
            log_resps.append(log_resp)

        return E_log_p_X_given_Z, log_resps

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
        log_alphas = np.zeros_like(llhs, order='C')
        log_alphas[0] += self._log_pi
        log_A_T = self._log_A.T.copy(order='C')
        add = np.add
        buffer = np.zeros((len(self.states), len(self.states)),  
                          dtype=np.float32)
        for t in range(1, len(llhs)):
            buffer.fill(0)
            add(log_A_T, llhs[t], out=buffer)
            add(buffer, log_alphas[t - 1], out=buffer)
            _fast_logsumexp_axis1(buffer,
                                  log_alphas[t])
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
        log_betas = np.zeros_like(llhs, order='C', dtype=np.float32) 
        log_betas -= float('inf')
        log_betas[-1, self._final_state_idx] = 0.
        log_A = self._log_A
        add = np.add
        buffer = np.zeros((len(self.states), len(self.states)),  
                          dtype=np.float32)
        for t in reversed(range(llhs.shape[0]-1)):
            buffer.fill(0)
            add(log_A, llhs[t + 1], out=buffer)
            add(buffer, log_betas[t + 1], out=buffer)
            _fast_logsumexp_axis1(buffer, log_betas[t])
        return log_betas

    def viterbi(self, llhs, use_parent_name=False):
        """Viterbi algorithm.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM.

        Returns
        -------
        path : list
            List of the state of the mostr probable path.

        """
        backtrack = np.zeros_like(llhs, dtype=int)
        omega = llhs[0] + self._log_pi 
        for t in range(1, llhs.shape[0]):
            hypothesis = omega + self._log_A.T
            backtrack[i] = np.argmax(hypothesis, axis=1)
            omega = llhs[i] + hypothesis[range(len(self.log_A)),
                                         backtrack[i]]

        path_idx = [self.final_state_idx[np.argmax(log_omegas[-1, 
            self.final_state_idx])]]
        for i in reversed(range(1, len(llhs))):
            path_idx.insert(0, backtrack[i, path_idx[0]])

        path = []
        for idx in path_idx:
            if use_parent_name:
                name = self.states[idx].parent_name
            else:
                name = self.states[idx].name
            path.append(name)

        return path

    def logProbInit(self):
        """Log probability of the initial states.

        Returns
        -------
        log_pi : numpy.ndarray
            An array of the size of the number of states of the HMM.

        """
        log_pi = np.zeros(len(self.states)) - float('inf')
        for idx, state in enumerate(self.states):
            if state in self.init_states:
                log_pi[idx] = self._state_log_pi[state]

        return log_pi

    def logProbTrans(self):
        """Transition matrix in the log probability domain.

        Returns
        -------
        log_A : numpy.ndarray
            Log transitions.

        """
        log_A = np.zeros((len(self.states), len(self.states)), 
                         dtype=np.float32)
        log_A -= float('inf')

        for idx1, state in enumerate(self.states):
            for next_state_id, weight in state.next_states.items():
                next_state = self.id_state[next_state_id]
                idx2 = self.states.index(next_state)
                log_A[idx1, idx2] = weight 

        return log_A

    def addState(self, name, parent_name=None):
        """Add a state to the HMM graph.

        Parameters
        ----------
        name : str
            Name of the state.
        parent_name : str
            Parent name of the state.

        Returns
        -------
        state : :class:`State`
            The state created.
        """
        state_id = self._nextStateId()
        state = State(state_id, name, parent_name=parent_name)
        self.id_state[state_id] = state
        self.states.append(state)
        return state

    def setInitState(self, state):
        """Mark the state as a possible initial state.

        Parameters
        ----------
        state : :class:`State`
            New initial state.

        """
        self.init_states.append(state)

    def setFinalState(self, state):
        """Mark the state as a possible final state.

        Parameters
        ----------
        state : :class:`State`
            New final state.

        """
        self.final_states.append(state)

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

    def addSilenceState(self, name, nstates):
        """Add a silence state to the HMM.

        We force the HMM to start and to finish in the silence state.

        Parameters
        ----------
        name : str
            Name of the state.
        nstates : int
            Number of states in the silence model. 

        """
        sil_model = self.leftToRightHmm(name, nstates)
        self.states += sil_model.states
        self.id_state = {**self.id_state, **sil_model.id_state}
        sil_state = sil_model.states[0]

        for state in self.final_states:
            self.addLink(state, sil_state)
        for state in self.init_states:
            self.addLink(sil_state, state)
        self.init_states = [sil_state]
        self.final_states = [sil_state]

        self._prepare()

    def setUnigramWeights(self, weights):
        """Set unigram language model for the unit-loop HMM.

        Parameters
        ----------
        weights : dict
            Mapping unit_name -> log_probability

        """
        for final_state in self.final_states:
            for init_state in self.init_states:
                weight = weights[init_state.parent_name]
                self.addLink(final_state, init_state, weight)

        self._prepare()

    def toFst(self):
        """Convert the HMM graph to an OpenFst object.

        You need to have installed the OpenFst python extension to used
        this method.

        Returns
        -------
        graph : pywrapfst.Fst
            The FST representation of the HMM graph. An super initial
            state and a super final state will be added though they are
            not present in the HMM.

        """

        import pywrapfst as fst

        f = fst.Fst('log')

        start_state = f.add_state()
        f.set_start(start_state)
        end_state = f.add_state()
        f.set_final(end_state)

        state_fstid = {}
        for state in self.states:
            fstid = f.add_state()
            state_fstid[state.state_id] = fstid

        for state in self.states:
            for next_state_id, weight in state.next_states.items():
                fstid = state_fstid[state.state_id]
                next_fstid = state_fstid[next_state_id]
                arc = fst.Arc(0, 0, fst.Weight('log', -weight), next_fstid)
                f.add_arc(fstid, arc)

        for state in self.init_states:
            fstid = state_fstid[state.state_id]
            arc = fst.Arc(0, 0, fst.Weight.One('log'), fstid)
            f.add_arc(start_state, arc)

        for state in self.final_states:
            fstid = state_fstid[state.state_id]
            arc = fst.Arc(0, 0, fst.Weight.One('log'), end_state)
            f.add_arc(fstid, arc)

        return f
