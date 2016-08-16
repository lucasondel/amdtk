
"""Implementation of a different phone-loop structure as FST that can
be used to discover acoustic units."""

import abc
import tempfile
import numpy as np
import pywrapfst as fst


class DecodableGraph(metaclass=abc.ABCMeta):

    def forward(self):
        pass

    def backward(self):
        pass

    def updatePosterior(self, stats):
        """Update the parameters of the posterior distribution according
        to the accumulated statistics.

        Parameters
        ----------
        stats : object
            Accumulated statistics.

        """
        pass


class UnigramDecodableGraph(DecodableGraph):

    @property
    def graph(self):
        return self._graph

    # ------------------------------------------------------------------
    # We override the pickling behavior of the object to be able to deal
    # with OpenFst component. This is inefficient as we have to write
    # the object on the disk but it avoid to modify the OpenFst API.

    def __getstate__(self):
        state = self.__dict__.copy()

        out = tempfile.NamedTemporaryFile()
        state['_graph'].write(out.name)
        with open(out.name, 'rb') as f:
            state['_graph'] = f.read()

        out = tempfile.NamedTemporaryFile()
        state['_r_graph'].write(out.name)
        with open(out.name, 'rb') as f:
            state['_r_graph'] = f.read()

        return state

    def __setstate__(self, newstate):
        out = tempfile.NamedTemporaryFile()
        with open(out.name, 'wb') as f:
            f.write(newstate['_graph'])
        newstate['_graph'] = fst.Fst.read(out.name)

        out = tempfile.NamedTemporaryFile()
        with open(out.name, 'wb') as f:
            f.write(newstate['_r_graph'])
        newstate['_r_graph'] = fst.Fst.read(out.name)

        self.__dict__.update(newstate)

    # ------------------------------------------------------------------

    def __init__(self, nunits, nstates):
        self.nunits = nunits
        self.nstates = nstates

        # Create the fst.
        self._graph = fst.Fst('log')

        # Set the initial/final state.
        self._start_state = self._graph.add_state()
        self._graph.set_start(self._start_state)
        self._end_state = self._graph.add_state()
        self._graph.set_final(self._end_state)

        # Create the states of the graphs.
        for i in range(nunits * nstates):
            self._graph.add_state()

        # Define the possible initial and final states.
        self._init_states = [self._start_state + 2]
        self._final_states = [self._start_state + nstates + 1]
        for i in range(1, nunits):
            self._final_states.append(self._final_states[i - 1] + nstates)
            self._init_states.append(self._final_states[-1] - nstates + 1)

        # Create the mapping state -> index that will be used during the
        # forward-backward algorithm.
        self._state_index = {}
        for i in range(nunits * nstates):
            self._state_index[i + 2] = i

        # Connect the graph's states.
        self.__connectStates()

        # For the backward recursion we need the reversed graph.
        self._r_graph = fst.reverse(self._graph, require_superinitial=False)

    def __connectStates(self):
        # Weight shared for all HMM transitions.
        weight = fst.Weight('log', -np.log(.5))

        # Connect all the initial states to the starting state.
        for state in self._init_states:
            arc = fst.Arc(state, state, weight, state)
            self._graph.add_arc(self._start_state, arc)

        for state in self._final_states:
            arc = fst.Arc(0, 0, fst.Weight.One('log'), self._end_state)
            self._graph.add_arc(state, arc)

        # Connect all the final states to the initial states to complete
        # the loop.
        for final_state in self._final_states:
            for init_state in self._init_states:
                arc = fst.Arc(init_state, init_state, fst.Weight.One('log'),
                              init_state)
                self._graph.add_arc(final_state, arc)

        # HMM connections.
        for init_state in self._init_states:
            for state in range(init_state, init_state + self.nstates):
                # Self loops.
                arc = fst.Arc(state, state, weight, state)
                self._graph.add_arc(state, arc)

                # State to state transition.
                if state not in self._final_states:
                    next_state = state + 1
                    arc = fst.Arc(next_state, next_state, weight, next_state)
                    self._graph.add_arc(state, arc)

    def __distanceToFinalState(self, state):
        return ((state - 1) % self.nstates) + 1

    def __distanceToInitialState(self, state):
        return self.nstates - ((state - 1) % self.nstates)

    def __consumeEpsilons(self, graph, arc):
        if arc.ilabel == 0:
            arcs = []
            for new_arc in graph.arcs(arc.nextstate):
                arcs += self.__consumeEpsilons(graph, new_arc)
            return arcs
        else:
            return [arc]

    def __stepForward(self, state, frame_index, log_alphas, llhs,
                      active_states):
        # Forward value up to the current state.
        if frame_index == 0:
            log_alpha = fst.Weight.One('log')
        else:
            state_idx = self._state_index[state]
            log_alpha = log_alphas[frame_index - 1, state_idx]

        # Propagate the weight for each outgoing arcs.
        for arc in self._graph.arcs(state):
            # Make sure the path is a valid.
            if arc.nextstate == self._end_state:
                continue
            remaining_frame = len(llhs) - frame_index
            if self.__distanceToFinalState(arc.nextstate) > remaining_frame:
                continue

            # Index of the state we propagate the forward recursion.
            next_state_idx = self._state_index[arc.nextstate]

            # Convert the acoustic weight into OpenFst weight.
            ac_weight = fst.Weight('log', -llhs[frame_index, next_state_idx])

            # Weight to add for this state.
            weight = fst.times(arc.weight, log_alpha)
            weight = fst.times(weight, ac_weight)

            if log_alphas[frame_index, next_state_idx] is None:
                log_alphas[frame_index, next_state_idx] = weight
            else:
                current_weight = log_alphas[frame_index, next_state_idx]
                log_alphas[frame_index, next_state_idx] = \
                    fst.plus(current_weight, weight)

            # Add the state to the active state set.
            active_states.add(arc.nextstate)

    def __stepBackward(self, state, frame_index, log_betas, llhs,
                       active_states):
        # Index of the current state.
        state_idx = self._state_index[state]

        # Backward value up to the current state.
        log_beta = log_betas[frame_index, state_idx]

        # Propagate the weight for each outgoing arcs.
        for arc in self._r_graph.arcs(state):
            assert arc.ilabel != 0, "Oops !"

            # Make sure the path is a valid.
            if arc.nextstate == self._start_state:
                continue
            remaining_frame = frame_index
            if self.__distanceToInitialState(arc.nextstate) > remaining_frame:
                continue

            # Index of the state we propagate the backward recursion.
            next_state_idx = self._state_index[arc.nextstate]

            # Convert the acoustic weight into OpenFst weight.
            ac_weight = \
                fst.Weight('log', -llhs[frame_index, state_idx])

            # Weight to add for this state.
            weight = fst.times(arc.weight, log_beta)
            weight = fst.times(weight, ac_weight)

            if log_betas[frame_index - 1, next_state_idx] is None:
                log_betas[frame_index - 1, next_state_idx] = weight
            else:
                current_weight = log_betas[frame_index - 1, next_state_idx]
                log_betas[frame_index - 1, next_state_idx] = \
                    fst.plus(current_weight, weight)

            # Add the state to the active state set.
            active_states.add(arc.nextstate)

    def forward(self, llhs):
        """Forward recursion given the log emission probabilities and
        the HMM model.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM model.

        Returns
        -------
        log_alphas : numpy.ndarray
            The log alphas values of the recursions.

        """
        # Allocate a matrix where we will store the results of the
        # forward recursion.
        log_alphas = np.empty_like(llhs, dtype=object)

        # Starting point of the forward recursion.
        active_states = set([self._start_state])

        # Current frame index.
        frame_index = 0

        # Forward recursion.
        while frame_index < len(llhs):
            # Current state from which to expand the forward recursion.
            states = list(active_states)

            for state in states:
                active_states.remove(state)
                self.__stepForward(state, frame_index, log_alphas, llhs,
                                   active_states)
            frame_index += 1

        # For debugging.
        retval = np.empty_like(log_alphas, dtype=float)
        for i in range(log_alphas.shape[0]):
            for j in range(log_alphas.shape[1]):
                if log_alphas[i, j] is None:
                    retval[i, j] = float('-inf')
                else:
                    retval[i, j] = -float(log_alphas[i, j].string)

        return retval

    def backward(self, llhs):
        """Backward recursion given the log emission probabilities and
        the HMM model.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM model.

        Returns
        -------
        log_alphas : numpy.ndarray
            The log alphas values of the recursions.

        """
        # Allocate a matrix where we will store the results of the
        # forward recursion.
        log_betas = np.empty_like(llhs, dtype=object)

        # Initialize the recurstion.
        for state in self._final_states:
            state_idx = self._state_index[state]
            log_betas[-1, state_idx] = fst.Weight.One('log')

        # Starting point of the backward recursion.
        active_states = set(self._final_states)

        # Current frame index.
        frame_index = len(llhs) - 1

        # Backward recursion.
        while frame_index >= 0:
            print('frame_index:', frame_index)
            # Current state from which to expand the backward recursion.
            states = list(active_states)
            for state in states:
                active_states.remove(state)
                self.__stepBackward(state, frame_index, log_betas, llhs,
                                    active_states)
            frame_index -= 1

        # For debugging.
        retval = np.empty_like(log_betas, dtype=float)
        for i in range(log_betas.shape[0]):
            for j in range(log_betas.shape[1]):
                if log_betas[i, j] is None:
                    retval[i, j] = float('-inf')
                else:
                    retval[i, j] = -float(log_betas[i, j].string)

        return retval
