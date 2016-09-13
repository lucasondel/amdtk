#!/usr/bin/env python

import unittest
import numpy as np
from amdtk.models import Model


class FakeModel(Model):
    
    def __init__(self):
        super().__init__()
        

class TestModel(unittest.TestCase):
    
    def testUuid(self):
        nobjs = 10000
        uuids = np.zeros(nobjs)
        for i in range(nobjs):
            uuids[i] = FakeModel().uuid
        self.assertEqual(len(uuids), len(np.unique(uuids)),
                         msg='some uuids are not unique')
        
if __name__ == '__main__':
    unittest.main()
    
