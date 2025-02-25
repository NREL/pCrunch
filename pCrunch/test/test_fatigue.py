import os
import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from pCrunch import AeroelasticOutput, FatigueParams, read

DIR = os.path.split(__file__)[0]
DATA = os.path.join(DIR, "data")

FOUT  = ['AOC_WSt.out', 'DLC2.3_1.out', 'DLC2.3_2.out', 'DLC2.3_3.out']

class Test_Fatigue(unittest.TestCase):

    def test_init(self):
        # Test DNV curves
        myfat = FatigueParams(dnv_type='Air', dnv_name='d')
        self.assertEqual(myfat.load2stress, 1.0)
        self.assertEqual(myfat.S_ult, 1.0)
        self.assertEqual(myfat.bins, 100.0)
        self.assertEqual(myfat.goodman, False)
        self.assertEqual(myfat.curve.m1, 3.0)
        self.assertEqual(myfat.curve.m2, 5.0)

        myfat = FatigueParams(load2stress=25.0, dnv_type='sea', dnv_name='c1')
        self.assertEqual(myfat.load2stress, 25.0)
        self.assertEqual(myfat.S_ult, 1.0)
        self.assertEqual(myfat.bins, 100.0)
        self.assertEqual(myfat.goodman, False)
        self.assertEqual(myfat.curve.m, 3.0)

        myfat = FatigueParams(bins=256, goodman=True, ultimate_stress=1e6, dnv_type='cathodic', dnv_name='b2')
        self.assertEqual(myfat.load2stress, 1.0)
        self.assertEqual(myfat.S_ult, 1e6)
        self.assertEqual(myfat.bins, 256.0)
        self.assertEqual(myfat.goodman, True)
        self.assertEqual(myfat.curve.m1, 4.0)
        self.assertEqual(myfat.curve.m2, 5.0)

        # Test Sc-Nc
        myfat = FatigueParams(Sc=1e9, Nc=2e6, slope=3)
        self.assertEqual(myfat.load2stress, 1.0)
        self.assertEqual(myfat.S_ult, 1.0)
        self.assertEqual(myfat.bins, 100.0)
        self.assertEqual(myfat.goodman, False)
        self.assertEqual(myfat.curve.get_stress(2e6), 1e9)
        self.assertEqual(myfat.curve.m, 3)

        # Test S_intercept
        myfat = FatigueParams(ultimate_stress=1e10, slope=4)
        self.assertEqual(myfat.load2stress, 1.0)
        self.assertEqual(myfat.S_ult, 1e10)
        self.assertEqual(myfat.bins, 100.0)
        self.assertEqual(myfat.goodman, False)
        self.assertEqual(myfat.curve.get_stress(1), 1e10)
        self.assertEqual(myfat.curve.m, 4)

        myfat = FatigueParams(ultimate_stress=1e10, slope=4, S_intercept=1e12)
        self.assertEqual(myfat.load2stress, 1.0)
        self.assertEqual(myfat.S_ult, 1e10)
        self.assertEqual(myfat.bins, 100.0)
        self.assertEqual(myfat.goodman, False)
        self.assertEqual(myfat.curve.get_stress(1), 1e12)
        self.assertEqual(myfat.curve.m, 4)

        # Test priority of curve setting
        myfat = FatigueParams(dnv_type='Air', dnv_name='d', Sc=1e9, Nc=2e6, slope=3, ultimate_stress=1e10, S_intercept=1e12)
        self.assertEqual(myfat.load2stress, 1.0)
        self.assertEqual(myfat.S_ult, 1e10)
        self.assertEqual(myfat.bins, 100.0)
        self.assertEqual(myfat.goodman, False)
        self.assertEqual(myfat.curve.m1, 3.0)
        self.assertEqual(myfat.curve.m2, 5.0)

        myfat = FatigueParams(Sc=1e9, Nc=2e6, slope=3, ultimate_stress=1e10, S_intercept=1e12)
        self.assertEqual(myfat.load2stress, 1.0)
        self.assertEqual(myfat.S_ult, 1e10)
        self.assertEqual(myfat.bins, 100.0)
        self.assertEqual(myfat.goodman, False)
        self.assertEqual(myfat.curve.get_stress(2e6), 1e9)
        self.assertEqual(myfat.curve.m, 3)

    def test_plotting(self):
        nn = 2*10**np.arange(10)
        myfat = FatigueParams(Sc=1e4, Nc=2e6, slope=3)
        ss = myfat.get_stress(nn)
        truth = np.logspace(6,3,nn.size)
        npt.assert_almost_equal(ss, truth)
        

    def test_rainflows(self):
        myout = read(os.path.join(DATA, FOUT[2]))
        myfat = FatigueParams(Sc=1e4, Nc=2e6, slope=3, ultimate_stress=1e7)
        N, S = myfat.get_rainflow_counts(myout['TwrBsFyt'], 50)
        self.assertEqual(N.size, 50)
        self.assertEqual(S.size, 50)

        
        
    def test_dels(self):
        myparam = FatigueParams(load2stress = 25.0,
                                slope = 3.0,
                                ultimate_stress = 6e8,
                                S_intercept = 5e9,
                                goodman_correction = False,
                                )
        t = np.linspace(0, 600, 10000)
        y0 = 80e3 * np.sin(2*np.pi*t/60.0)
        y80 = y0 + 80e3
        zeros = np.zeros(y0.shape)
        mydata = {"Time":t,
                  "Signal0":y0,
                  "Signal80":y80,
                  "Zeros":zeros}

        mymagnitudes = {"Mag0":["Signal0", "Zeros"],
                        "Mag80":["Signal80", "Zeros"]}

        myfatigues = {"Signal0":myparam,
                      "Signal80":myparam,
                      "Mag0":myparam,
                      "Mag80":myparam}

        myobj = AeroelasticOutput(mydata, magnitude_channels=mymagnitudes,
                                  fatigue_channels=myfatigues)

        # Test rainflow counts here too
        N, S = myparam.get_rainflow_counts(myobj['Signal80'], 50)
        idx = np.nonzero(N)[0]
        self.assertEqual(N.size, 50)
        self.assertEqual(S.size, 50)
        self.assertEqual(idx.size, 1)
        self.assertEqual(N[idx], 10)
        npt.assert_almost_equal(S[idx], 2*80e3, 2)
        
        dels, dams = myobj.get_DELs(return_damage = True)


        self.assertAlmostEqual(dels['Signal0'], dels['Signal80'])
        self.assertGreater(dels['Signal0'], dels['Mag0'])
        self.assertAlmostEqual(dels['Signal0'], dels['Mag80'])
        
        self.assertAlmostEqual(dams['Signal0'], dams['Signal80'])
        self.assertGreater(dams['Signal0'], dams['Mag0'])
        self.assertAlmostEqual(dams['Signal0'], dams['Mag80'])
        
        dels2, dams2 = myobj.get_DELs(goodman_correction=True, return_damage = True)
        
        self.assertGreater(dels2['Signal80'], dels2['Signal0'])
        self.assertGreater(dams2['Signal80'], dams2['Signal0'])
        
if __name__ == "__main__":
    unittest.main()
