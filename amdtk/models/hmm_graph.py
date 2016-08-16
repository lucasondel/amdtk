

class State(object):

    def __init__(self, identifier):
        self.identifier = identifier
        self.next_states = {}
        self.previous_state = {}

    def addLink(self, state, weight):
        self.next_states[state.identifier] = weight
        state.previous_states[self.identifier] = weight

    def removeLink(self, state):
        self.next_states.pop(state)
        state.previous_states.pop(self)


class HmmGraph(object):

    def __init__(self):
        self.states = {}
        self.init_states = []
        self.final_states = []

    def addState(self):
        state_id = len(self._states)
        self._states[state_id] = State(state_id)
        return state_id

    def setInitState(self, state_id):
        self.init_states.append(state_id)

    def setFinalState(self, state_id):
        self.final_states.append(state_id)

    def addLink(self, state_id, next_state_id, weight):
        src = self.states[state_id]
        dest = self.states[next_state_id]
        src.addLink(dest)

    def forward(self, llhs):
        log_alphas = np.zeros_like(llhs, dtype=float)

        self.active_states = set(self._graph.init_states)
        for frame_idx in len(llhs):
