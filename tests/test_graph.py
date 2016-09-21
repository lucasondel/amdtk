
import unittest
import numpy as np
from scipy.misc import logsumexp
from amdtk.models import Model
from amdtk.models import State
from amdtk.models import Graph


class TestState(unittest.TestCase):

    def testUuid(self):
        nobjs = 10000
        uuids = np.zeros(nobjs)
        for i in range(nobjs):
            uuids[i] = State('state' + str(i), None).uuid
        self.assertEqual(len(uuids), len(np.unique(uuids)),
                         msg='some uuids are not unique')

    def testAddLink(self):
        s1 = State('state1', None)
        s2 = State('state2', None)

        s1.addLink(s2, -5)
        self.assertTrue(s2.uuid in s1.next_states)
        self.assertTrue(s1.next_states[s2.uuid], -5)
        self.assertTrue(s1.uuid in s2.previous_states)
        self.assertTrue(s2.previous_states[s1.uuid] == -5)

    def testNormalize(self):
        s1 = State('state1', None)
        s2 = State('state2', None)
        s3 = State('state3', None)
        states = {s1.uuid: s1, s2.uuid: s2, s3.uuid: s3}

        s1.addLink(s2, -5)
        s1.addLink(s3, -5)
        s1.normalize(states)
        log_weights = np.zeros(2)
        for i, uuid in enumerate(s1.next_states):
            log_weights[i] = s1.next_states[uuid]
        self.assertAlmostEqual(np.exp(log_weights).sum(), 1.0)
        log_weights = np.zeros(2)
        log_weights[0] = s2.previous_states[s1.uuid]
        log_weights[1] = s3.previous_states[s1.uuid]
        self.assertAlmostEqual(np.exp(log_weights).sum(), 1.0)


class TestGraph(unittest.TestCase):

    def testInit(self):
        graph = Graph('hmm')
        self.assertEqual(graph.name, 'hmm')

    def testAddState(self):
        graph = Graph('hmm')
        state1 = graph.addState('hmm_state1', None)
        state2 = graph.addState('hmm_state2', None)
        self.assertTrue(state1.uuid in graph.states)
        self.assertTrue(state2.uuid in graph.states)

        self.assertEqual(graph.sorted_states, [state1, state2])

        graph.setInitState(state1)
        self.assertTrue(state1.uuid in graph.init_states)

        graph.setFinalState(state2)
        self.assertTrue(state2.uuid in graph.final_states)

    def testAddGraph(self):
        graph1 = Graph('hmm')
        state1 = graph1.addState('hmm1_state1', None)
        state2 = graph1.addState('hmm1_state2', None)
        graph1.setInitState(state1)
        graph1.setFinalState(state2)
        graph1.addLink(state1, state2)
        graph2 = Graph('hmm')
        state3 = graph2.addState('hmm2_state1', None)
        state4 = graph2.addState('hmm2_state2', None)
        graph2.setInitState(state3)
        graph2.setFinalState(state4)
        graph2.addLink(state3, state4)
        graph1.addGraph(graph2)

        self.assertTrue(state1.uuid in graph1.states)
        self.assertTrue(state2.uuid in graph1.states)
        self.assertTrue(state3.uuid in graph1.states)
        self.assertTrue(state4.uuid in graph1.states)

        self.assertEqual(graph1.sorted_states,
                         [state1, state2, state3, state4])

        self.assertTrue(state1.uuid in graph1.init_states)
        self.assertTrue(state3.uuid in graph1.init_states)

        self.assertTrue(state2.uuid in graph1.final_states)
        self.assertTrue(state4.uuid in graph1.final_states)

        self.assertTrue(state1.uuid in state2.next_states)
        self.assertTrue(state1.uuid in state4.next_states)
        self.assertTrue(state3.uuid in state2.next_states)
        self.assertTrue(state3.uuid in state4.next_states)

    def testSetUniformPobInit(self):
        graph = Graph('hmm')
        state1 = graph.addState('hmm_state', None)
        state2 = graph.addState('hmm_state', None)
        graph.setInitState(state1)
        graph.setInitState(state2)
        graph.setUniformProbInitStates()
        test_val = float('-inf')
        for state_uuid in graph.init_states:
            test_val = np.logaddexp(graph.logProbInit[state_uuid], test_val)
        self.assertAlmostEqual(np.exp(test_val), 1.0)

    def testNormalize(self):
        graph = Graph('hmm')
        state1 = graph.addState('state1', None)
        state2 = graph.addState('state2', None)
        state3 = graph.addState('state3', None)
        graph.addLink(state1, state1, -5)
        graph.addLink(state1, state2, -5)
        graph.addLink(state2, state2, -5)
        graph.addLink(state2, state3, -5)
        graph.addLink(state3, state3, -5)
        graph.normalize()
        log_A = graph.getLogProbTrans()
        for i in range(log_A.shape[0]):
            self.assertAlmostEqual(logsumexp(log_A[i]), 0, msg='HMM is not '
                                   'properly normalized')

    def testForwardBackward(self):
        data = {
            'mean': np.array([0., 0.]),
            'var': np.array([1., 1.])
        }
        config_file = 'tests/data/multivariate_gaussian.cfg'
        emission = Model.create(config_file, data)
        graph = Graph('hmm')
        state1 = graph.addState('state1', emission)
        state2 = graph.addState('state2', emission)
        state3 = graph.addState('state3', emission)
        graph.addLink(state1, state1, -5)
        graph.addLink(state1, state2, -5)
        graph.addLink(state2, state2, -5)
        graph.addLink(state2, state3, -5)
        graph.addLink(state3, state3, -5)
        graph.setInitState(state1)
        graph.setFinalState(state3, -5)
        graph.setUniformProbInitStates()
        graph.normalize()

        llhs = np.zeros((4, 3))
        log_alphas = graph.forward(llhs)
        log_betas = graph.backward(llhs)
        log_P_Z = log_alphas + log_betas
        log_P_Z = (log_P_Z.T - logsumexp(log_P_Z, axis=1)).T
        expected_result = np.array([
            [  1,   0,   0],
            [1/3, 2/3,   0],
            [  0, 2/3, 1/3],
            [  0,   0,   1]
        ], dtype=float)

        self.assertTrue(np.isclose(np.exp(log_P_Z), expected_result).all())


if __name__ == '__main__':
    unittest.main()
