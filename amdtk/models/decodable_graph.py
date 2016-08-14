
"""Non-parametric Bayesian phone-loop model."""

import abc
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
    
    def __init__(self, nunits, nstates):
        self.nunits = nunits
        self.nstates = nstates
        
        # Create the fst.
        self._graph = fst.Fst('log')
        
        # Set the initial/final state. For the unigram phone-loop the initial 
        # and final state are the same.
        self._start_state = self._graph.add_state()
        self._graph.set_start(self._start_state)
        self._graph.set_final(self._start_state)
        
        # Create the states of the graphs.
        for i in range(nunits * nstates):        
            self._graph.add_state()
        
        # Define the possible initial and final states.
        self._init_states = [self._start_state + 1]
        self._final_states = [nstates]
        for i in range(1, nunits):
            self._final_states.append(self._final_states[i - 1] + nstates)
            self._init_states.append(self._final_states[-1] - nstates + 1)
        
        # Connect the graph's states.
        self.__connectStates()
        
        
    def __connectStates(self):      
        # Weight shared for all HMM transitions.
        weight = fst.Weight('log', -np.log(.5))
        
        # Connect all the initial states to the starting state.
        for state in self._init_states:
            arc = fst.Arc(state, state, fst.Weight.One('log'), state)
            self._graph.add_arc(self._start_state, arc)
        
        # Connect all the final states to the ending state.
        for state in self._final_states:
            arc = fst.Arc(self._start_state, self._start_state, 
                          fst.Weight.One('log'), self._start_state)
            self._graph.add_arc(state, arc)
        
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
                    