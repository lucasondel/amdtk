
import unittest
import numpy as np
from scipy.special import psi
from amdtk.models import Dirichlet
from amdtk.models import DirichletProcess
from amdtk.models import DirichletProcessStats
from amdtk.models import Model
from amdtk.models import InvalidModelError
from amdtk.models import InvalidModelParameterError
from amdtk.models import MissingModelParameterError
from amdtk.models import Prior
from amdtk.models import PriorStats


class FakeModel(Model):

    @classmethod
    def loadParams(cls, config, data):
        pass

    def __init__(self, params):
        super().__init__(params)

    def stats(stats, x, data, weights):
        pass


class TestDirichletProcessStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.T = 3
        cls.X = X = np.random.dirichlet([1] * cls.T, size=1000)
        cls.weights = .5 * np.ones(len(X))
        cls.weighted_X = (cls.weights * cls.X.T).T

    def testInit(self):
        X = self.X
        weights = self.weights
        weighted_X = self.weighted_X

        stats1 = DirichletProcessStats(X)
        stats2 = DirichletProcessStats(X, 1.0)
        self.assertTrue(np.isclose(stats1[0], stats2[0]).all())
        self.assertTrue(np.isclose(stats1[1], stats2[1]).all())

        stats1 = weighted_X.sum(axis=0)
        stats2 = np.zeros_like(stats1)
        for i in range(len(stats1)-1):
            stats2[i] += stats1[i+1:].sum()
        stats = DirichletProcessStats(X, weights)

        self.assertTrue(isinstance(stats, PriorStats))

        self.assertTrue(np.isclose(stats[0], stats1).all(), 'invalid stats')
        self.assertTrue(np.isclose(stats[1], stats2).all(), 'invalid stats')

    def testAccumulate(self):
        X = self.X
        weights = self.weights

        stats1 = DirichletProcessStats(X, weights)
        stats2 = DirichletProcessStats(X, weights)
        stats2 += stats2

        self.assertTrue(np.isclose(2 * stats1[0], stats2[0]).all())
        self.assertTrue(np.isclose(2 * stats1[1], stats2[1]).all())

    def testFailIndex(self):
        X = self.X
        weights = self.weights

        stats1 = DirichletProcessStats(X, weights)

        with self.assertRaises(IndexError):
            stats1[-1]
        with self.assertRaises(IndexError):
            stats1[2]
        with self.assertRaises(TypeError):
            stats1['asdf']


class TestDirichletProcess(unittest.TestCase):

    def testCreateFromConfigFile(self):
        config_file = 'tests/data/dirichlet_process.cfg'
        model = Model.create(config_file, {})
        self.assertTrue(isinstance(model, DirichletProcess))
        self.assertEqual(model.T, 10)
        self.assertAlmostEqual(model.gamma, 2)

    def testInit(self):
        params = {
            'T': 10,
            'gamma': 1
        }
        model = DirichletProcess(params)
        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, Prior))

    def testInitInvalidParameter(self):
        params = {
            'T': 10,
            'gamma': 0
        }
        with self.assertRaises(InvalidModelParameterError):
            DirichletProcess(params)

        params = {
            'T': 10,
            'gamma': -1
        }
        with self.assertRaises(InvalidModelParameterError):
            DirichletProcess(params)

        params = {
            'T': 0,
            'gamma': 1
        }
        with self.assertRaises(InvalidModelParameterError):
            DirichletProcess(params)

        params = {
            'T': -1,
            'gamma': 1
        }
        with self.assertRaises(InvalidModelParameterError):
            DirichletProcess(params)

    def testInitMissingParameter(self):
        params = {
            'T': 10
        }
        with self.assertRaises(MissingModelParameterError):
            DirichletProcess(params)

        params = {
            'gamma': 1
        }
        with self.assertRaises(MissingModelParameterError):
            DirichletProcess(params)

    def testExpectations(self):
        params = {
            'T': 10,
            'gamma': 1
        }
        model = DirichletProcess(params)

        E_weights = model.expectedX()
        comp = model.g1 / (model.g1 + model.g2)
        self.assertTrue(np.isclose(E_weights, comp).all())

        E_log_weights = model.expectedLogX()
        comp = psi(model.g1) - psi(model.g1 + model.g2)
        nv = psi(model.g2) - psi(model.g1 + model.g2)
        for i in range(1, model.T):
            comp[i] += nv[:i].sum()
        self.assertTrue(np.isclose(E_log_weights, comp).all())

    def testKL(self):
        params = {
            'T': 10,
            'gamma': 1,
        }
        p = DirichletProcess(params)

        params = {
            'T': 10,
            'gamma': 2,
        }
        q = DirichletProcess(params)

        KL = 0
        for i in range(p.T):
            a1 = np.array([p.g1[i], p.g2[i]])
            a2 = np.array([q.g1[i], q.g2[i]])
            d1 = Dirichlet({'alphas': a1})
            d2 = Dirichlet({'alphas': a2})
            KL += d1.KL(d2)
        return KL

        self.assertAlmostEqual(p.KL(q), KL)
        self.assertAlmostEqual(p.KL(p), 0.)

        fake_model = FakeModel({})
        with self.assertRaises(InvalidModelError):
            p.KL(fake_model)

if __name__ == '__main__':
    unittest.main()
