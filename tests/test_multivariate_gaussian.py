
import unittest
import numpy as np
from amdtk.models import Model
from amdtk.models import VBModel
from amdtk.models import BayesianGaussianDiagCov
from amdtk.models import InvalidModelParameterError
from amdtk.models import MissingModelParameterError
from amdtk.models import NormalGamma


class FakeModel(Model):

    @classmethod
    def loadParams(cls, config, data):
        pass

    def __init__(self, params):
        super().__init__(params)

    def stats(stats, x, data, weights):
        pass


class TestBayesianGaussianDiagCov(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        cls.prior = NormalGamma(params)

        params = {
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        }
        cls.posterior = NormalGamma(params)

        cls.X = np.random.multivariate_normal([0, 0],
                                              [[1, 0], [0, 1]],
                                              size=1000)

    def testCreateFromConfigFile(self):
        data = {
            'mean': np.array([0., 0.]),
            'var': np.array([1., 1.])
        }
        config_file = 'tests/data/multivariate_gaussian.cfg'
        model = Model.create(config_file, data)
        self.assertTrue(isinstance(model.prior, NormalGamma))
        self.assertTrue(isinstance(model.prior, NormalGamma))

    def testInit(self):
        params = {
            'prior': self.prior,
            'posterior': self.posterior
        }
        model = BayesianGaussianDiagCov(params)
        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, VBModel))

    def testInitMissingParameter(self):
        params = {
            'prior': self.prior
        }
        with self.assertRaises(MissingModelParameterError):
            BayesianGaussianDiagCov(params)

        params = {
            'posterior': self.posterior
        }
        with self.assertRaises(MissingModelParameterError):
            BayesianGaussianDiagCov(params)

    def testInitInvalidParameter(self):
        params = {
            'prior': FakeModel({}),
            'posterior': self.posterior
        }
        with self.assertRaises(InvalidModelParameterError):
            BayesianGaussianDiagCov(params)

        params = {
            'prior': self.prior,
            'posterior': FakeModel({})
        }
        with self.assertRaises(InvalidModelParameterError):
            BayesianGaussianDiagCov(params)

    def testExpLogLikelihood(self):
        params = {
            'prior': self.prior,
            'posterior': self.posterior
        }
        model = BayesianGaussianDiagCov(params)

        X = self.X
        llh, data = model.expectedLogLikelihood(X, weight=1.0)
        _, log_prec = model.posterior.expectedLogX()
        m, prec = model.posterior.expectedX()
        norm = 0
        norm += .5 * log_prec.sum()
        norm += - .5 * np.log(2 * np.pi)
        comp = norm - .5 * ((1 / prec).sum() + (prec * (X - m)**2).sum(axis=1))
        self.assertTrue(np.isclose(llh, comp).all())


if __name__ == '__main__':
    unittest.main()
