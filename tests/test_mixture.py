
import unittest
import numpy as np
from scipy.misc import logsumexp
from amdtk.models import Model
from amdtk.models import VBModel
from amdtk.models import BayesianMixture
from amdtk.models import BayesianGaussianDiagCov
from amdtk.models import Dirichlet
from amdtk.models import DirichletProcess
from amdtk.models import InvalidModelParameterError
from amdtk.models import MissingModelParameterError
from amdtk.models import DiscreteLatentModelEmptyListError
from amdtk.models import NormalGamma


class FakeModel(Model):

    @classmethod
    def loadParams(cls, config, data):
        pass

    def __init__(self, params):
        super().__init__(params)

    def stats(stats, x, data, weights):
        pass


class TestBayesianMixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        g_prior = NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        g_posterior = NormalGamma(params)

        params = {
            'prior': g_prior,
            'posterior': g_posterior
        }
        g1 = BayesianGaussianDiagCov(params)
        g2 = BayesianGaussianDiagCov(params)
        cls.components = [g1, g2]

        params = {
            'alphas': np.array([1, 1]),
        }
        cls.prior = Dirichlet(params)
        cls.posterior = Dirichlet(params)

        params = {
            'T': 2,
            'gamma': 1
        }
        cls.dp_prior = DirichletProcess(params)
        cls.dp_posterior = DirichletProcess(params)

        cls.X = np.random.multivariate_normal([0, 0],
                                              [[1, 0], [0, 1]],
                                              size=1000)

    def testCreateFromConfigFile(self):
        data = {
            'mean': np.array([0., 0.]),
            'var': np.array([1., 1.])
        }
        config_file = 'tests/data/gmm.cfg'
        model = Model.create(config_file, data)
        self.assertTrue(isinstance(model.prior, Dirichlet))
        self.assertTrue(isinstance(model.prior, Dirichlet))

    def testInit(self):
        params = {
            'prior': self.prior,
            'posterior': self.posterior,
            'components': self.components
        }
        model = BayesianMixture(params)
        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, VBModel))

        params = {
            'prior': self.dp_prior,
            'posterior': self.dp_posterior,
            'components': self.components
        }
        model = BayesianMixture(params)
        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, VBModel))

    def testInitMissingParameter(self):
        params = {
            'posterior': self.posterior,
            'components': self.components
        }
        with self.assertRaises(MissingModelParameterError):
            BayesianMixture(params)

        params = {
            'prior': self.prior,
            'components': self.components
        }
        with self.assertRaises(MissingModelParameterError):
            BayesianMixture(params)

        params = {
            'prior': self.prior,
            'posterior': self.posterior
        }
        with self.assertRaises(MissingModelParameterError):
            BayesianMixture(params)

    def testInitInvalidParameter(self):
        params = {
            'prior': FakeModel({}),
            'posterior': self.posterior,
            'components': self.components
        }
        with self.assertRaises(InvalidModelParameterError):
            BayesianMixture(params)

        params = {
            'prior': self.prior,
            'posterior': FakeModel({}),
            'components': self.components
        }
        with self.assertRaises(InvalidModelParameterError):
            BayesianMixture(params)

        params = {
            'prior': self.prior,
            'posterior': self.posterior,
            'components': []
        }
        with self.assertRaises(DiscreteLatentModelEmptyListError):
            BayesianMixture(params)

    def testExpLogLikelihood(self):
        params = {
            'prior': self.prior,
            'posterior': self.posterior,
            'components': self.components
        }
        model = BayesianMixture(params)

        X = self.X
        llh, data = model.expectedLogLikelihood(X, weight=1.0)
        log_weights = model.posterior.expectedLogX()
        comp = np.zeros((X.shape[0], model.k))
        for i, component in enumerate(model.components):
            comp_llh, data = component.expectedLogLikelihood(X)
            comp[:, i] += log_weights[i] + comp_llh
        comp = logsumexp(comp, axis=1)
        self.assertTrue(np.isclose(llh, comp).all())


if __name__ == '__main__':
    unittest.main()
