import os
import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from pCrunch import AeroelasticOutput, FatigueParams, Crunch, read

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
        # Empty
        myobj = Crunch()
        self.assertEqual(myobj.lean_flag, False)
        self.assertEqual(myobj.outputs, [])
        self.assertEqual(myobj.noutputs, 0)
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, None)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})
        npt.assert_equal(myobj.prob, np.array([]))
        pdt.assert_frame_equal(myobj.summary_stats, pd.DataFrame())
        self.assertEqual(myobj.extremes, {})
        pdt.assert_frame_equal(myobj.dels, pd.DataFrame())
        pdt.assert_frame_equal(myobj.damage, pd.DataFrame())

        # Just one
        myout = AeroelasticOutput(data)
        myobj = Crunch(myout, magnitude_channels=mc)
        self.assertEqual(myobj.lean_flag, False)
        self.assertEqual(myobj.noutputs, 1)
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, mc)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})
        npt.assert_equal(myobj.prob, [1.0])
        pdt.assert_frame_equal(myobj.summary_stats, pd.DataFrame())
        self.assertEqual(myobj.extremes, {})
        pdt.assert_frame_equal(myobj.dels, pd.DataFrame())
        pdt.assert_frame_equal(myobj.damage, pd.DataFrame())

        # Multiple
        myobj = Crunch([myout, myout], magnitude_channels=mc)
        self.assertEqual(myobj.lean_flag, False)
        self.assertEqual(myobj.noutputs, 2)
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, mc)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})
        npt.assert_equal(myobj.prob, [0.5, 0.5])
        pdt.assert_frame_equal(myobj.summary_stats, pd.DataFrame())
        self.assertEqual(myobj.extremes, {})
        pdt.assert_frame_equal(myobj.dels, pd.DataFrame())
        pdt.assert_frame_equal(myobj.damage, pd.DataFrame())

        # With trimming
        myobj = Crunch([myout, myout], magnitude_channels=mc, trim_data=[3,6])
        self.assertEqual(myobj.lean_flag, False)
        self.assertEqual(myobj.noutputs, 2)
        self.assertEqual(myobj.td, [3,6])
        self.assertEqual(myobj.mc, mc)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})
        npt.assert_equal(myobj.prob, [0.5, 0.5])
        pdt.assert_frame_equal(myobj.summary_stats, pd.DataFrame())
        self.assertEqual(myobj.extremes, {})
        pdt.assert_frame_equal(myobj.dels, pd.DataFrame())
        pdt.assert_frame_equal(myobj.damage, pd.DataFrame())
        self.assertEqual(myobj.outputs[0].data.shape, (4,5))
        self.assertEqual(myobj.outputs[1].data.shape, (4,5))
        npt.assert_equal(myobj.outputs[0].time, np.arange(3,7))
        npt.assert_equal(myobj.outputs[1].time, np.arange(3,7))

        
    def testProcessOutputs(self):
        nrep = len(FOUTB)
        myouts = [read(os.path.join(DATA, m)) for m in FOUTB]
        
        # Serial
        myobj = Crunch(myouts, magnitude_channels=mc)
        myobj.process_outputs()
        self.assertEqual(myobj.noutputs, nrep)
        self.assertEqual(len(myobj.outputs), nrep)
        ss1, ext1, del1, dam1 = myobj.summary_stats, myobj.extremes, myobj.dels, myobj.damage

        # Parallel
        myobj = Crunch(myouts, magnitude_channels=mc)
        myobj.process_outputs(cores=2)
        self.assertEqual(myobj.noutputs, nrep)
        self.assertEqual(len(myobj.outputs), nrep)
        ss2, ext2, del2, dam2 = myobj.summary_stats, myobj.extremes, myobj.dels, myobj.damage

        # Lean
        myobj = Crunch(myouts, magnitude_channels=mc, lean=True)
        myobj.process_outputs()
        self.assertEqual(myobj.noutputs, nrep)
        self.assertEqual(len(myobj.outputs), 0)
        ss3, ext3, del3, dam3 = myobj.summary_stats, myobj.extremes, myobj.dels, myobj.damage

        pdt.assert_frame_equal(ss1, ss2)
        pdt.assert_frame_equal(ss1, ss3)

        #self.maxDiff = None
        #self.assertDictEqual(ext1, ext2)
        #self.assertDictEqual(ext1, ext3)
        
        pdt.assert_frame_equal(del1, del2)
        pdt.assert_frame_equal(del1, del3)
        
        pdt.assert_frame_equal(dam1, dam2)
        pdt.assert_frame_equal(dam1, dam3)

    def testStreamingByOutput(self):
        nrep       = len(FOUTB)
        myouts     = [read(os.path.join(DATA, m)) for m in FOUTB]
        myobj_base = Crunch(myouts, magnitude_channels=mc)
        myobj_base.process_outputs()
        myobj      = Crunch(lean=True, magnitude_channels=mc)

        nrep = len(FOUTB)
        for k in range(nrep):
            myobj.add_output(myouts[k])

        self.assertEqual(myobj_base.noutputs, nrep)
        self.assertEqual(myobj.noutputs, nrep)
        self.assertEqual(len(myobj_base.outputs), nrep)
        self.assertEqual(len(myobj.outputs), 0)

        pdt.assert_frame_equal(myobj_base.summary_stats, myobj.summary_stats, check_like=True)
        pdt.assert_frame_equal(myobj_base.dels, myobj.dels, check_column_type=False)
        pdt.assert_frame_equal(myobj_base.damage, myobj.damage, check_column_type=False)

    def testStreamingByStats(self):
        nrep       = len(FOUTB)
        myouts     = [read(os.path.join(DATA, m)) for m in FOUTB]
        myobj_base = Crunch(myouts, magnitude_channels=mc)
        myobj_base.process_outputs()
        myobj      = Crunch(lean=True, magnitude_channels=mc)

        nrep = len(FOUTB)
        for k in range(nrep):
            fname, stats, extremes, dels, damage =  myobj.process_single(myouts[k])
            myobj.add_output_stats(fname, stats, extremes, dels, damage)

        self.assertEqual(myobj_base.noutputs, nrep)
        self.assertEqual(myobj.noutputs, nrep)
        self.assertEqual(len(myobj_base.outputs), nrep)
        self.assertEqual(len(myobj.outputs), 0)

        pdt.assert_frame_equal(myobj_base.summary_stats, myobj.summary_stats, check_like=True)
        pdt.assert_frame_equal(myobj_base.dels, myobj.dels, check_column_type=False)
        pdt.assert_frame_equal(myobj_base.damage, myobj.damage, check_column_type=False)

    def testDels(self):
        self.assertTrue(True)

    def testLoadRankings(self):
        self.assertTrue(True)

    def testWindspeeds(self):
        myout = AeroelasticOutput(data)
        myouts = [myout]*10
        
        # Read the output
        myobj = Crunch(myouts, magnitude_channels=mc)
        windout = myobj._get_windspeeds('Wind')
        npt.assert_equal(windout, 7.5*np.ones(10))
        windout = myobj._get_windspeeds('Wind', idx=[1,4])
        npt.assert_equal(windout, 7.5*np.ones(2))

        # Read the stats log
        myobj = Crunch(myouts, magnitude_channels=mc, lean=True)
        myobj.process_outputs()
        windout = myobj._get_windspeeds('Wind')
        npt.assert_equal(windout, 7.5*np.ones(10))
        windout = myobj._get_windspeeds('Wind', idx=[1,4])
        npt.assert_equal(windout, 7.5*np.ones(2))

        # Provide the input
        myobj = Crunch([myout]*5, magnitude_channels=mc, lean=True)
        myobj.process_outputs()
        windout = myobj._get_windspeeds([1,2,3,4,5])
        npt.assert_equal(windout, [1,2,3,4,5])
        windout = myobj._get_windspeeds([1,2,3,4,5], idx=[1,4])
        npt.assert_equal(windout, [2,5])
        windout = myobj._get_windspeeds([4,5], idx=[1,4])
        npt.assert_equal(windout, [4,5])

        # Expect error
        with self.assertRaises(ValueError):
            windout = myobj._get_windspeeds([1,2,3,4])
        with self.assertRaises(ValueError):
            windout = myobj._get_windspeeds([1,2,3,4], idx=[1,4])

    def test_prob(self):
        myout = AeroelasticOutput(data)
        myouts = [myout]*10
        
        myobj = Crunch(myouts, magnitude_channels=mc)
        myobj.process_outputs()
        
        myobj.set_probability_distribution('Wind', 7.5, kind='weibull')
        npt.assert_equal(myobj.prob, 0.1*np.ones(10))
        
        myobj.set_probability_distribution('Wind', 7.5, kind='weibull', idx=[1,4])
        npt.assert_equal(myobj.prob[[1,4]], 0.5*np.ones(2))
        
        myobj.set_probability_distribution('Wind', 7.5, kind='rayleigh')
        npt.assert_equal(myobj.prob, 0.1*np.ones(10))
        
        myobj.set_probability_distribution('Wind', 7.5, kind='rayleigh', idx=[1,4])
        npt.assert_equal(myobj.prob[[1,4]], 0.5*np.ones(2))
        
        myobj.set_probability_distribution('Wind', 7.5, kind='uniform')
        npt.assert_equal(myobj.prob, 0.1*np.ones(10))
        
        myobj.set_probability_distribution('Wind', 7.5, kind='uniform', idx=[1,4])
        npt.assert_equal(myobj.prob[[1,4]], 0.5*np.ones(2))
        
        myobj.set_probability_distribution('Wind', 7.5, kind='bleh')
        npt.assert_equal(myobj.prob, 0.1*np.ones(10))
        
        myobj.set_probability_distribution('Wind', 7.5, kind='bleh', idx=[1,4])
        npt.assert_equal(myobj.prob[[1,4]], 0.5*np.ones(2))

        myobj.set_probability_turbine_class('Wind', 1)
        myobj.set_probability_turbine_class('Wind', 2)
        myobj.set_probability_turbine_class('Wind', 3)
        myobj.set_probability_turbine_class('Wind', 1, idx=[1,4])

    def test_aep(self):
        myout = AeroelasticOutput(data)
        myouts = [myout]*10

        aep_true = myout.compute_energy('WindVxi') * 8760*60*60/myout.elapsed_time
        
        myobj = Crunch(myouts, magnitude_channels=mc)
        aep_w1, aep_uw1 = myobj.compute_aep('Wind')
        self.assertAlmostEqual(aep_true, aep_w1, 5)
        self.assertAlmostEqual(aep_w1, aep_uw1, 5)
        
        myobj = Crunch(myouts, magnitude_channels=mc, lean=True)
        myobj.process_outputs()
        aep_w2, aep_uw2 = myobj.compute_aep('Wind')
        self.assertAlmostEqual(aep_w1, aep_w2, 5)
        self.assertAlmostEqual(aep_w2, aep_uw2, 5)
        
        aep_w3, aep_uw3 = myobj.compute_aep('Wind', loss_factor=0.5)
        self.assertAlmostEqual(0.5*aep_w1, aep_w3, 5)
        self.assertAlmostEqual(aep_w3, aep_uw3, 5)
        
        aep_w4, aep_uw4 = myobj.compute_aep('Wind', idx=[1,4])
        self.assertAlmostEqual(aep_w1, aep_w4, 5)
        self.assertAlmostEqual(aep_w4, aep_uw4, 5)
        
        myobj.prob[0] = 0.0
        aep_w5, aep_uw5 = myobj.compute_aep('Wind')
        self.assertAlmostEqual(aep_w1, aep_w5, 5)
        self.assertAlmostEqual(aep_w4, aep_uw4, 5)

        
    
    def test_total_fatigue(self):
        pass

    def testProperties(self):
        myout = AeroelasticOutput(data)
        myouts = [myout]*10
        
        myobj = Crunch(myouts, magnitude_channels=mc)
        npt.assert_equal(myobj.num_timesteps(), [10]*10)
        npt.assert_equal(myobj.elapsed_time(), [9]*10)
        npt.assert_equal(myobj.num_channels(), [5]*10)
        npt.assert_equal(myobj.idxmins(), [np.zeros(5)]*10)
        npt.assert_equal(myobj.idxmaxs(), [[9, 5, 0, 0, 5]]*10)
        npt.assert_equal(myobj.minima(), [[1, 7, 0, 0, 7]]*10)
        npt.assert_equal(myobj.maxima(), [[10, 8, 0, 0, 8]]*10)
        npt.assert_equal(myobj.absmaxima(), [[10, 8, 0, 0, 8]]*10)
        npt.assert_equal(myobj.ranges(), [[9, 1, 0, 0, 1]]*10)
        npt.assert_equal(myobj.variable(), [[0, 1, 4]]*10)
        npt.assert_equal(myobj.constant(), [[2, 3]]*10)
        npt.assert_equal(np.array(myobj.sums()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.sums_squared()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.sums_cubed()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.sums_fourth()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.second_moments()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.third_moments()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.fourth_moments()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.means()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.medians()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.stddevs()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.skews()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.kurtosis()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.integrated()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.compute_energy('WindVxi')), myobj.integrated()[0][1])

        # Can only do some stats without the output list
        myobj = Crunch(myouts, magnitude_channels=mc, lean=True)
        myobj.process_outputs()
        #npt.assert_equal(myobj.num_timesteps(), [10]*10)
        npt.assert_equal(myobj.elapsed_time(), [9]*10)
        #npt.assert_equal(myobj.num_channels(), [5]*10)
        #npt.assert_equal(myobj.idxmins(), [np.zeros(5)]*10)
        #npt.assert_equal(myobj.idxmaxs(), [[9, 5, 0, 0, 5]]*10)
        npt.assert_equal(myobj.minima(), [[1, 7, 0, 0, 7]]*10)
        npt.assert_equal(myobj.maxima(), [[10, 8, 0, 0, 8]]*10)
        #npt.assert_equal(myobj.absmaxima(), [[10, 8, 0, 0, 8]]*10)
        npt.assert_equal(myobj.ranges(), [[9, 1, 0, 0, 1]]*10)
        #npt.assert_equal(myobj.variable(), [[0, 1, 4]]*10)
        #npt.assert_equal(myobj.constant(), [[2, 3]]*10)
        #npt.assert_equal(np.array(myobj.sums()).shape, (10,myobj.num_channels()[0]))
        #npt.assert_equal(np.array(myobj.sums_squared()).shape, (10,myobj.num_channels()[0]))
        #npt.assert_equal(np.array(myobj.sums_cubed()).shape, (10,myobj.num_channels()[0]))
        #npt.assert_equal(np.array(myobj.sums_fourth()).shape, (10,myobj.num_channels()[0]))
        #npt.assert_equal(np.array(myobj.second_moments()).shape, (10,myobj.num_channels()[0]))
        #npt.assert_equal(np.array(myobj.third_moments()).shape, (10,myobj.num_channels()[0]))
        #npt.assert_equal(np.array(myobj.fourth_moments()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.means()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.medians()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.stddevs()).shape, (10,myobj.num_channels()[0]))
        #npt.assert_equal(np.array(myobj.skews()).shape, (10,myobj.num_channels()[0]))
        #npt.assert_equal(np.array(myobj.kurtosis()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.integrated()).shape, (10,myobj.num_channels()[0]))
        npt.assert_equal(np.array(myobj.compute_energy('WindVxi')), myobj.integrated()[0][1])
    
if __name__ == "__main__":
    unittest.main()
    
