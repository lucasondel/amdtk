
import unittest
import numpy as np
from scipy.special import gammaln, psi
from amdtk.models import NormalGamma
from amdtk.models import NormalGammaStats
from amdtk.models import Model
from amdtk.models import InvalidModelError
from amdtk.models import InvalidModelParameterError
from amdtk.models import MissingModelParameterError
from amdtk.models import Prior
from amdtk.models import PriorStats


class FakeModel(Model):

    def __init__(self, params):
        super().__init__(params)


class TestNormalGammaStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]],
                                                  size=1000)
        cls.weights = .5 * np.ones(len(X))
        cls.weighted_X = (cls.weights * cls.X.T).T

    def testInit(self):
        X = self.X
        weights = self.weights
        weighted_X = self.weighted_X

        stats = NormalGammaStats(X, weights)

        self.assertTrue(isinstance(stats, PriorStats))

        self.assertAlmostEqual(stats[0], weights.sum(), 'invalid stats')
        self.assertTrue(np.isclose(stats[1], weighted_X.sum(axis=0)).all(),
                        'invalid stats')
        self.assertTrue(np.isclose(stats[2],
                        (X * weighted_X).sum(axis=0)).all(),
                        'invalid stats')

    def testAccumulate(self):
        X = self.X
        weights = self.weights

        stats1 = NormalGammaStats(X, weights)
        stats2 = NormalGammaStats(X, weights)
        stats2 += stats2

        self.assertEqual(2 * stats1[0], stats2[0])
        self.assertTrue(np.isclose(2 * stats1[1], stats2[1]).all())
        self.assertTrue(np.isclose(2 * stats1[2], stats2[2]).all())

    def testFailIndex(self):
        X = self.X
        weights = self.weights

        stats1 = NormalGammaStats(X, weights)

        with self.assertRaises(IndexError):
            stats1[-1]
        with self.assertRaises(IndexError):
            stats1[3]
        with self.assertRaises(TypeError):
            stats1['asdf']


class TestNormalGamma(unittest.TestCase):

    def testInit(self):
        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        model = NormalGamma(params)
        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, Prior))

    def testInitInvalidParameter(self):
        params = {
            'mu': np.array([0, 0]),
            'kappa': 0,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        with self.assertRaises(InvalidModelParameterError):
            NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': -1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        with self.assertRaises(InvalidModelParameterError):
            NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 0,
            'beta': np.array([1, 1])
        }
        with self.assertRaises(InvalidModelParameterError):
            NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': -1,
            'beta': np.array([1, 1])
        }
        with self.assertRaises(InvalidModelParameterError):
            NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 0])
        }
        with self.assertRaises(InvalidModelParameterError):
            NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, -1])
        }
        with self.assertRaises(InvalidModelParameterError):
            NormalGamma(params)

    def testInitMissingParameter(self):
        params = {
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        with self.assertRaises(MissingModelParameterError):
            NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        with self.assertRaises(MissingModelParameterError):
            NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'beta': np.array([1, 1])
        }
        with self.assertRaises(MissingModelParameterError):
            NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
        }
        with self.assertRaises(MissingModelParameterError):
            NormalGamma(params)

    def testExpectations(self):
        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        model = NormalGamma(params)

        E_mean, E_prec = model.expectedX()
        self.assertTrue(np.isclose(E_mean, model.mu).all())
        self.assertTrue(np.isclose(E_prec, model.alpha / model.beta).all())

        E_log_mean, E_log_prec = model.expectedLogX()
        self.assertIsNone(E_log_mean)
        comp = psi(model.alpha) - np.log(model.beta)
        self.assertTrue(np.isclose(E_log_prec, comp).all())

    def testKL(self):
        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        p = NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([2, 2])
        }
        q = NormalGamma(params)

        E_mean, E_prec = p.expectedX()
        _, E_log_prec = p.expectedLogX()

        KL = .5 * (np.log(p.kappa) - np.log(q.kappa))
        KL += - .5 * (1 - q.kappa * (1. / p.kappa + E_prec * (p.mu - q.mu)**2))
        KL += - gammaln(p.alpha) - gammaln(q.alpha)
        KL += p.alpha * np.log(p.beta) - q.alpha * np.log(q.beta)
        KL += E_log_prec * (p.alpha - q.alpha)
        KL += - E_prec * (p.beta - q.beta)
        KL = KL.sum()

        self.assertAlmostEqual(p.KL(q), KL)
        self.assertAlmostEqual(p.KL(p), 0.)

        fake_model = FakeModel({})
        with self.assertRaises(InvalidModelError):
            p.KL(fake_model)

if __name__ == '__main__':
    unittest.main()
