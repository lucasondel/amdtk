#!/usr/bin/env python

import amdtk.models
import unittest


class FakeDiscreteLatentModel(amdtk.models.DiscreteLatentModel):
    
    def __init__(self, objs):
        super().__init__(objs)
        

class TestDiscreteLatentModel(unittest.TestCase):
    
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
    