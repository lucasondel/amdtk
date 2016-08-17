
import numpy as np
from scipy.misc import logsumexp


class State(object):

    def __init__(self, name):
        self.name = name
        self.next_states = {}
        self.previous_states = {}

    def addLink(self, state, weight):
        self.next_states[state] = weight
        state.previous_states[self] = weight

    def removeLink(self, state):
        self.next_states.pop(state)
        state.previous_states.pop(self)


class HmmGraph(object):

    @classmethod
    def selfLoop(cls, name):
        graph = cls()
        state = graph.addState(name)
        graph.addLink(state, state)
        graph.init_states = graph.states
        graph.final_states = graph.states
        graph._prepare()
        return graph

    @classmethod
    def leftToRightHmm(cls, name, nstates):
        graph = cls()

        for i in range(nstates):
            state_name = name + '_' + str(i+1)
            state = graph.addState(state_name)
            if i == 0:
                graph.setInitState(state)
            if i == nstates - 1:
                graph.setFinalState(state)

        for i, state in enumerate(graph.states):
            graph.addLink(state, state)
            if not graph.isFinalState(state):
                next_state = graph.states[i + 1]
                graph.addLink(state, next_state)

        graph._prepare()

        return graph

    @classmethod
    def standardPhoneLoop(cls, prefix, nunits, nstates):
        graph = cls()

        for i in range(nunits):
            unit_name = prefix + str(i+1)
            hmm_graph = cls.leftToRightHmm(unit_name, nstates)
            graph.states += hmm_graph.states
            graph.init_states += hmm_graph.init_states
            graph.final_states += hmm_graph.final_states

        for final_state in graph.final_states:
            for init_state in graph.init_states:
                graph.addLink(final_state, init_state)

        graph._prepare()

        return graph

    def __init__(self):
        self.states = []
        self.init_states = []
        self.final_states = []
        self.models = []
        self.name_states = {}

    @property
    def names(self):
        return list(self.name_states.keys())

    def _computelogProbInitStates(self):
        state_log_pi = 1 / len(self.init_states)
        self._state_log_pi = {}
        for state in self.init_states:
            self._state_log_pi[state] = state_log_pi

    def _updateNameStatesMapping(self):
        self.name_states = {}
        for state in self.states:
            try:
                self.name_states[state.name].append(state)
            except KeyError:
                self.name_states[state.name] = [state]

    def _normalize(self):
        for state in self.states:
            next_states = []
            weights = np.zeros(len(state.next_states))
            for i, next_state in enumerate(state.next_states):
                next_states.append(next_state)
                weights[i] = state.next_states[next_state]
            weights /= weights.sum()
            for i, next_state in enumerate(next_states):
                state.next_states[next_state] = weights[i]
                next_state.previous_states[state] = weights[i]

    def _prepare(self):
        self._updateNameStatesMapping()
        self._normalize()
        self._computelogProbInitStates()

    def logProbInit(self, index_name):
        pi = np.zeros(len(index_name))
        for i, name in index_name.items():
            for state in self.name_states[name]:
                if state in self.init_states:
                    pi[i] += self._state_log_pi[state]
        return np.log(pi)

    def logProbTransitions(self, name, name_index, incoming=True):
        prob = np.zeros(len(self.name_states))
        for state in self.name_states[name]:
            if incoming:
                next_states = state.previous_states
            else:
                next_states = state.next_states
            for next_state, weight in next_states.items():
                idx = name_index[next_state.name]
                prob[idx] += weight
        return np.log(prob)

    def finalNames(self):
        final_names = set()
        for state in self.final_states:
            final_names.add(state.name)
        return final_names

    def addState(self, name):
        state = State(name)
        self.states.append(state)
        return state

    def setInitState(self, state):
        self.init_states.append(state)

    def isInitState(self, state):
        return state in self.init_states

    def setFinalState(self, state):
        self.final_states.append(state)

    def isFinalState(self, state):
        return state in self.final_states

    def addLink(self, state, next_state, weight=1.):
        state.addLink(next_state, weight)

    def addSilenceState(self):
        sil_state1 = self.selfLoop('sil')
        sil_state2 = self.selfLoop('sil')
        sil_state3 = self.selfLoop('sil')
        self.states += sil_state1.states
        self.states += sil_state2.states
        self.states += sil_state3.states

        self.init_states += sil_state3.init_states
        for state in self.final_states:
            for next_state in sil_state3.states:
                self.addLink(state, next_state)
        for state in sil_state3.final_states:
            for next_state in self.init_states:
                self.addLink(state, next_state)
        self.final_states += sil_state3.init_states

        for state in sil_state1.states:
            for next_state in self.init_states:
                self.addLink(state, next_state)

        for state in self.final_states:
            for next_state in sil_state2.init_states:
                self.addLink(state, next_state)

        self.init_states = sil_state1.init_states
        self.final_states = sil_state2.final_states

        self._prepare()

    def toFst(self):
        import pywrapfst as fst

        f = fst.Fst('log')

        start_state = f.add_state()
        f.set_start(start_state)
        end_state = f.add_state()
        f.set_final(end_state)

        state_fstid = {}
        for state in self.states:
            fstid = f.add_state()
            state_fstid[state] = fstid

        for state in self.states:
            for next_state, weight in state.next_states.items():
                fstid = state_fstid[state]
                next_fstid = state_fstid[next_state]
                arc = fst.Arc(0, 0, fst.Weight('log', -weight), next_fstid)
                f.add_arc(fstid, arc)

        for state in self.init_states:
            fstid = state_fstid[state]
            arc = fst.Arc(0, 0, fst.Weight.One('log'), fstid)
            f.add_arc(start_state, arc)

        for state in self.final_states:
            fstid = state_fstid[state]
            arc = fst.Arc(0, 0, fst.Weight.One('log'), end_state)
            f.add_arc(fstid, arc)

        return f
