
import unittest
import numpy as np
from scipy.misc import logsumexp
from amdtk.models import Model
from amdtk.models import DiscreteLatentModel
from amdtk.models import LeftToRightHMM
from amdtk import StandardVariationalBayes
from amdtk.models import BayesianMixture
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

    def testInit(self):
        model = LeftToRightHMM({
            'name': 'hmm',
            'nstates': 3,
            'emissions': self.emissions
        })

        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(model, DiscreteLatentModel))

if __name__ == '__main__':
    unittest.main()
