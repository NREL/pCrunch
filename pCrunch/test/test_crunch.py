import os
import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd

from pCrunch import AeroelasticOutput, FatigueParams, Crunch

DIR = os.path.split(__file__)[0]
DATA = os.path.join(DIR, "data")

FOUT  = ['AOC_WSt.out', 'DLC2.3_1.out', 'DLC2.3_2.out', 'DLC2.3_3.out']
FOUTB = ['AOC_WSt.outb', 'Test1.outb', 'Test2.outb', 'Test3.outb', 'step_0.outb']

data = {
    "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "WindVxi": [7, 7, 7, 7, 7, 8, 8, 8, 8, 8],
    "WindVyi": [0] * 10,
    "WindVzi": [0] * 10,
}
mc = {"Wind": ["WindVxi", "WindVyi", "WindVzi"]}

class Test_Crunch(unittest.TestCase):
    
    def testConstructor(self):
        pass
        
if __name__ == "__main__":
    unittest.main()
    
