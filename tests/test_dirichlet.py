
import unittest
import numpy as np
from scipy.special import gammaln, psi
from amdtk.models import Dirichlet
from amdtk.models import DirichletStats
from amdtk.models import Model
from amdtk.models import InvalidModelError
from amdtk.models import InvalidModelParameterError
from amdtk.models import MissingModelParameterError
from amdtk.models import Prior
from amdtk.models import PriorStats


class FakeModel(Model):

    def __init__(self, params):
        super().__init__(params)


class TestDirichletStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = X = np.random.dirichlet([1, 1], size=1000)
        cls.weights = .5 * np.ones(len(X))
        cls.weighted_X = (cls.weights * cls.X.T).T

    def testInit(self):
        X = self.X
        weights = self.weights
        weighted_X = self.weighted_X

        stats1 = DirichletStats(X)
        stats2 = DirichletStats(X, 1.0)
        self.assertTrue(np.isclose(stats1[0], stats2[0]).all())

        stats = DirichletStats(X, weights)
        self.assertTrue(isinstance(stats, PriorStats))

        self.assertTrue(np.isclose(stats[0], weighted_X.sum(axis=0)).all(),
                        'invalid stats')

    def testAccumulate(self):
        X = self.X
        weights = self.weights

        stats1 = DirichletStats(X, weights)
        stats2 = DirichletStats(X, weights)
        stats2 += stats2

        self.assertTrue(np.isclose(2 * stats1[0], stats2[0]).all())

    def testFailIndex(self):
        X = self.X
        weights = self.weights

        stats1 = DirichletStats(X, weights)

        with self.assertRaises(IndexError):
            stats1[-1]
        with self.assertRaises(IndexError):
            stats1[1]
        with self.assertRaises(TypeError):
            stats1['asdf']


class TestDirichlet(unittest.TestCase):

    def testInit(self):
        params = {
            'alphas': np.array([1, 1]),
        }
        model = Dirichlet(params)
        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, Prior))

    def testInitInvalidParameter(self):
        params = {
            'alphas': np.array([0, 1]),
        }
        with self.assertRaises(InvalidModelParameterError):
            Dirichlet(params)

        params = {
            'alphas': np.array([-1, 1]),
        }
        with self.assertRaises(InvalidModelParameterError):
            Dirichlet(params)

    def testInitMissingParameter(self):
        params = {}
        with self.assertRaises(MissingModelParameterError):
            Dirichlet(params)

    def testExpectations(self):
        params = {
            'alphas': np.array([1, 1])
        }
        model = Dirichlet(params)

        E_weights = model.expectedX()
        comp = model.alphas / model.alphas.sum()
        self.assertTrue(np.isclose(E_weights, comp).all())

        E_log_weights = model.expectedLogX()
        comp = psi(model.alphas) - psi(model.alphas.sum())
        self.assertTrue(np.isclose(E_log_weights, comp).all())

    def testKL(self):
        params = {
            'alphas': np.array([1, 1])
        }
        p = Dirichlet(params)

        params = {
            'alphas': np.array([1, 1])
        }
        q = Dirichlet(params)

        E_mean, E_prec = p.expectedX()
        _, E_log_prec = p.expectedLogX()

        E_log_weights = p.expectedLogX()
        KL = gammaln(p.alphas.sum())
        KL -= gammaln(q.alphas.sum())
        KL -= gammaln(p.alphas).sum()
        KL += gammaln(q.alphas).sum()
        KL += (E_log_weights * (p.alphas - q.alphas)).sum()

        self.assertAlmostEqual(p.KL(q), KL)
        self.assertAlmostEqual(p.KL(p), 0.)

        fake_model = FakeModel({})
        with self.assertRaises(InvalidModelError):
            p.KL(fake_model)

if __name__ == '__main__':
    unittest.main()
