
import unittest
from amdtk.models import Model
from amdtk.models import DiscreteLatentModel
from amdtk.models import DiscreteLatentModelEmptyListError


class FakeDiscreteLatentModel(Model, DiscreteLatentModel):

    def __init__(self, params):
        super().__init__(params)

    @property
    def components(self):
        return self.params['components']

    def stats(stats, x, data, weights):
        pass


class TestDiscreteLatentModel(unittest.TestCase):

    def testSizeLatentVariable(self):
        size = 1
        params = {
            'components': [None]*size
        }
        fake_model = FakeDiscreteLatentModel(params)
        self.assertAlmostEqual(fake_model.k, size)

        size = 10
        params = {
            'components': [None]*size
        }
        fake_model = FakeDiscreteLatentModel(params)
        self.assertEqual(fake_model.k, size)

        size = 100
        params = {
            'components': [None]*size
        }
        fake_model = FakeDiscreteLatentModel(params)
        self.assertEqual(fake_model.k, size)

if __name__ == '__main__':
    unittest.main()
