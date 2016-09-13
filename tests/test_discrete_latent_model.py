#!/usr/bin/env python

from amdtk.models import DiscreteLatentModel
from amdtk.models import DiscreteLatentModelEmptyListError
import unittest


class FakeDiscreteLatentModel(DiscreteLatentModel):
    
    def __init__(self, objs):
        super().__init__(objs)
        

class TestDiscreteLatentModel(unittest.TestCase):
    
    def testEmptyList(self):
        with self.assertRaises(DiscreteLatentModelEmptyListError):
            FakeDiscreteLatentModel([])
    
    def testSizeLatentVariable(self):
        size = 1
        fake_model = FakeDiscreteLatentModel([None]*size)
        self.assertAlmostEqual(fake_model.k, size)
        
        size = 10
        fake_model = FakeDiscreteLatentModel([None]*size)
        self.assertAlmostEqual(fake_model.k, size)
        
        size = 100
        fake_model = FakeDiscreteLatentModel([None]*size)
        self.assertAlmostEqual(fake_model.k, size)
        
if __name__ == '__main__':
    unittest.main()
    