
import unittest
import numpy as np
from amdtk.models import Model
from amdtk.models import VBModel
from amdtk.models import DiscreteLatentModel
from amdtk.models import LeftToRightHMM
from amdtk.models import BayesianPhoneLoop
from amdtk.models import BayesianGaussianDiagCov
from amdtk.models import NormalGamma
from amdtk.models import InvalidModelError
from amdtk.models import Dirichlet
from amdtk.models import DirichletProcess


class FakeModel(Model):

    @classmethod
    def loadParams(cls, config, data):
        pass

    def __init__(self, params):
        super().__init__(params)

    def stats(stats, x, data, weights):
        pass


class TestLefToRightHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        g_prior = NormalGamma({
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        cls.emissions = []
        for i in range(3):
            g_posterior1 = NormalGamma({
                'mu': np.array([-5, -5]),
                'kappa': 1,
                'alpha': 1,
                'beta': np.array([1, 1])
            })

            g1 = BayesianGaussianDiagCov({
                'prior': g_prior,
                'posterior': g_posterior1
            })
            cls.emissions.append(g1)

    def testCreateFromConfigFile(self):
        data = {
            'mean': np.array([0., 0.]),
            'var': np.array([1., 1.])
        }
        config_file = 'tests/data/hmm.cfg'
        model = Model.create(config_file, data)
        self.assertTrue(isinstance(model, LeftToRightHMM))
        self.assertTrue(np.isclose(model.nstates, 3))

    def testInit(self):
        model = LeftToRightHMM({
            'name': 'hmm',
            'nstates': 3,
            'emissions': self.emissions
        })

        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, VBModel))
        self.assertTrue(isinstance(model, DiscreteLatentModel))


class TestPhoneLoop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        g_prior = NormalGamma({
            'mu': np.array([0, 0]),
            'kappa': 1,
            'alpha': 1,
            'beta': np.array([1, 1])
        })

        cls.emissions = []
        for i in range(3):
            g_posterior1 = NormalGamma({
                'mu': np.array([-5, -5]),
                'kappa': 1,
                'alpha': 1,
                'beta': np.array([1, 1])
            })

            g1 = BayesianGaussianDiagCov({
                'prior': g_prior,
                'posterior': g_posterior1
            })
            cls.emissions.append(g1)

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

    def testCreateFromConfigFile(self):
        data = {
            'mean': np.array([0., 0.]),
            'var': np.array([1., 1.])
        }
        config_file = 'tests/data/phone_loop.cfg'
        model = Model.create(config_file, data)
        self.assertTrue(isinstance(model, BayesianPhoneLoop))
        self.assertEqual(model.nunits, 2)

    def testInit(self):
        hmm = LeftToRightHMM({
            'name': 'hmm',
            'nstates': 3,
            'emissions': self.emissions
        })
        model = BayesianPhoneLoop({
            'nunits': 2,
            'emissions': hmm.components + hmm.components,
            'subhmms': [hmm, hmm],
            'prior': self.prior,
            'posterior': self.posterior
        })

        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, VBModel))
        self.assertTrue(isinstance(model, DiscreteLatentModel))


if __name__ == '__main__':
    unittest.main()
