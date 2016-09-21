
import unittest
import numpy as np
from amdtk.models import Model
from amdtk import VariationalBayes
from amdtk import StandardVariationalBayes
from amdtk.models import BayesianMixture
from amdtk.models import LeftToRightHMM
from amdtk.models import BayesianGaussianDiagCov
from amdtk.models import Dirichlet
from amdtk.models import DirichletProcess
from amdtk.models import NormalGamma
from amdtk.models import InvalidModelError


class FakeModel(Model):

    @classmethod
    def loadParams(cls, config, data):
        pass

    def __init__(self, params):
        super().__init__(params)

    def stats(stats, x, data, weights):
        pass


class TestStandardVariationalBayes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X1 = np.random.multivariate_normal([-2, -2],
                                           [[1, 0], [0, 1]],
                                           size=1000)
        X2 = np.random.multivariate_normal([2, 2],
                                           [[1, 0], [0, 1]],
                                           size=1000)
        cls.X = np.vstack([X1, X2])

    def testInit(self):
        alg = StandardVariationalBayes()
        self.assertTrue(isinstance(alg, VariationalBayes))

    def testInvalidModel(self):
        model = FakeModel({})
        alg = StandardVariationalBayes()
        with self.assertRaises(InvalidModelError):
            alg.expectation(model, self.X, 1.0)
        with self.assertRaises(InvalidModelError):
            alg.maximization(model, None)

    def testGmm(self):
        g_prior = NormalGamma({
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g_posterior1 = NormalGamma({
            'mu': np.array([-5, -5]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g_posterior2 = NormalGamma({
            'mu': np.array([5, 5]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g1 = BayesianGaussianDiagCov({
            'prior': g_prior,
            'posterior': g_posterior1
        })

        g2 = BayesianGaussianDiagCov({
            'prior': g_prior,
            'posterior': g_posterior2
        })

        prior = Dirichlet({
            'alphas': np.array([1, 1]),
        })
        posterior = Dirichlet({
            'alphas': np.array([1, 1]),
        })

        model = BayesianMixture({
            'prior': prior,
            'posterior': posterior,
            'components': [g1, g2]
        })
        alg = StandardVariationalBayes()

        X = self.X
        previous_E_llh = float('-inf')
        stop = False
        max_iter = 100
        niter = 0
        threshold = 1e-6
        while not stop and niter < max_iter:
            niter += 1
            F, stats = alg.expectation(model, X, 1.0)
            current_E_llh = (F.sum() - model.KLPosteriorPrior())
            diff = current_E_llh - previous_E_llh
            self.assertGreaterEqual(diff, 0, 'VB algorithm failed to improve '
                                    'the lower-bound')
            if diff < threshold:
                stop = True
            previous_E_llh = current_E_llh
            alg.maximization(model, stats)

        weights = np.exp(model.posterior.expectedLogX())
        self.assertLess(np.linalg.norm(weights - np.array([0.5, 0.5])), 0.01)

        m = model.components[0].posterior.mu
        norm = np.linalg.norm(m - np.array([-2, -2]))
        self.assertLess(norm, 0.2)

        self.assertGreater(niter, 3, 'VB algorithm has converged to quickly. '
                                     'This is suspicious.')

    def testDPGmm(self):
        g_prior = NormalGamma({
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g_posterior1 = NormalGamma({
            'mu': np.array([-5, -5]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g_posterior2 = NormalGamma({
            'mu': np.array([1, 1]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g1 = BayesianGaussianDiagCov({
            'prior': g_prior,
            'posterior': g_posterior1
        })

        g2 = BayesianGaussianDiagCov({
            'prior': g_prior,
            'posterior': g_posterior2
        })

        truncation = 2
        prior = DirichletProcess({
            'T': truncation,
            'gamma': 1
        })
        posterior = DirichletProcess({
            'T': truncation,
            'gamma': 1
        })

        model = BayesianMixture({
            'prior': prior,
            'posterior': posterior,
            'components': [g1, g2]
        })
        alg = StandardVariationalBayes()

        weights = np.exp(model.posterior.expectedLogX())

        X = self.X
        previous_E_llh = float('-inf')
        stop = False
        max_iter = 100
        niter = 0
        threshold = 1e-6
        while not stop and niter < max_iter:
            niter += 1
            F, stats = alg.expectation(model, X, 1.0)
            current_E_llh = (F.sum() - model.KLPosteriorPrior())
            diff = current_E_llh - previous_E_llh
            self.assertGreaterEqual(diff, 0, 'VB algorithm failed to improve '
                                    'the lower-bound')
            if diff < threshold:
                stop = True
            previous_E_llh = current_E_llh
            alg.maximization(model, stats)

        weights = np.exp(model.posterior.expectedLogX())
        self.assertLess(np.linalg.norm(weights - np.array([0.5, 0.5])), 0.01)

        m = model.components[0].posterior.mu
        norm = np.linalg.norm(m - np.array([-2, -2]))
        self.assertLess(norm, 0.2)

        m = model.components[1].posterior.mu
        norm = np.linalg.norm(m - np.array([2, 2]))
        self.assertLess(norm, 0.2)

        self.assertGreater(niter, 3, 'VB algorithm has converged to quickly. '
                                     'This is suspicious.')

    def testLeftToRightHMM(self):
        g_prior = NormalGamma({
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g_posterior1 = NormalGamma({
            'mu': np.array([-5, -5]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g_posterior2 = NormalGamma({
            'mu': np.array([5, 5]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        g1 = BayesianGaussianDiagCov({
            'prior': g_prior,
            'posterior': g_posterior1
        })

        g2 = BayesianGaussianDiagCov({
            'prior': g_prior,
            'posterior': g_posterior2
        })

        prior = Dirichlet({
            'alphas': np.array([1, 1]),
        })
        posterior = Dirichlet({
            'alphas': np.array([1, 1]),
        })

        model = LeftToRightHMM({
            'name': 'test',
            'nstates': 2,
            'emissions': [g1, g2]
        })
        alg = StandardVariationalBayes()

        X = self.X
        previous_E_llh = float('-inf')
        stop = False
        max_iter = 100
        niter = 0
        threshold = 1e-6
        while not stop and niter < max_iter:
            niter += 1
            F, stats = alg.expectation(model, X, 1.0)
            current_E_llh = (F.sum() - model.KLPosteriorPrior())
            diff = current_E_llh - previous_E_llh
            if abs(diff) < threshold:
                stop = True
            else:
                self.assertGreaterEqual(diff, 0, 'VB algorithm failed to '
                                        'improve the lower-bound')

            previous_E_llh = current_E_llh
            alg.maximization(model, stats)

        m = model.components[0].posterior.mu
        norm = np.linalg.norm(m - np.array([-2, -2]))
        self.assertLess(norm, 0.2)

        m = model.components[1].posterior.mu
        norm = np.linalg.norm(m - np.array([2, 2]))
        self.assertLess(norm, 0.2)

        self.assertGreater(niter, 2, 'VB algorithm has converged to quickly. '
                                     'This is suspicious.')


if __name__ == '__main__':
    unittest.main()
