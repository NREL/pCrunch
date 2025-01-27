import os
import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd

from pCrunch import AeroelasticOutput, FatigueParams

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

class Test_AeroelasticOutput(unittest.TestCase):
    
    def testConstructor(self):
        # Empty
        myobj = AeroelasticOutput()
        self.assertEqual(myobj.data, None)
        self.assertEqual(myobj.channels, None)
        self.assertEqual(myobj.units, None)
        self.assertEqual(myobj.description, "")
        self.assertEqual(myobj.filepath, "")

        # Test keywords
        myobj = AeroelasticOutput(description="test", units=["m"]*5, name="label")
        self.assertEqual(myobj.data, None)
        self.assertEqual(myobj.channels, None)
        self.assertEqual(myobj.units, ["m"]*5)
        self.assertEqual(myobj.description, "test")
        self.assertEqual(myobj.filepath, "label")

        # As dict
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        self.assertEqual(myobj.data.shape, (10,5))
        npt.assert_equal(myobj.data[:,0], np.array(data["Time"]))
        npt.assert_equal(myobj.data[:,1], np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,2], np.zeros(10))
        npt.assert_equal(myobj.data[:,3], np.zeros(10))
        npt.assert_equal(myobj.data[:,4], np.array(data["WindVxi"]))
        self.assertEqual(myobj.channels, list(data.keys())+["Wind"])
        self.assertEqual(myobj.units, None)
        self.assertEqual(myobj.description, "")
        self.assertEqual(myobj.filepath, "")

        # As DataFrame
        myobj = AeroelasticOutput(pd.DataFrame(data), magnitude_channels=mc)
        self.assertEqual(myobj.data.shape, (10,5))
        npt.assert_equal(myobj.data[:,0], np.array(data["Time"]))
        npt.assert_equal(myobj.data[:,1], np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,2], np.zeros(10))
        npt.assert_equal(myobj.data[:,3], np.zeros(10))
        npt.assert_equal(myobj.data[:,4], np.array(data["WindVxi"]))
        self.assertEqual(myobj.channels, list(data.keys())+["Wind"])
        self.assertEqual(myobj.units, None)
        self.assertEqual(myobj.description, "")
        self.assertEqual(myobj.filepath, "")

        # As Numpy
        myobj = AeroelasticOutput(myobj.data[:,:-1], list(data.keys()), magnitude_channels=mc)
        self.assertEqual(myobj.data.shape, (10,5))
        npt.assert_equal(myobj.data[:,0], np.array(data["Time"]))
        npt.assert_equal(myobj.data[:,1], np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,2], np.zeros(10))
        npt.assert_equal(myobj.data[:,3], np.zeros(10))
        npt.assert_equal(myobj.data[:,4], np.array(data["WindVxi"]))
        self.assertEqual(myobj.channels, list(data.keys())+["Wind"])
        self.assertEqual(myobj.units, None)
        self.assertEqual(myobj.description, "")
        self.assertEqual(myobj.filepath, "")

        # As lists
        myobj = AeroelasticOutput([m for m in data.values()], list(data.keys()), magnitude_channels=mc)
        self.assertEqual(myobj.data.shape, (10,5))
        npt.assert_equal(myobj.data[:,0], np.array(data["Time"]))
        npt.assert_equal(myobj.data[:,1], np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,2], np.zeros(10))
        npt.assert_equal(myobj.data[:,3], np.zeros(10))
        npt.assert_equal(myobj.data[:,4], np.array(data["WindVxi"]))
        self.assertEqual(myobj.channels, list(data.keys())+["Wind"])
        self.assertEqual(myobj.units, None)
        self.assertEqual(myobj.description, "")
        self.assertEqual(myobj.filepath, "")
        
    def testGetters(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc, dlc="/testdir/testfile")
        npt.assert_equal(myobj.time, data["Time"])
        npt.assert_equal(myobj.filepath, "/testdir/testfile")
        npt.assert_equal(myobj.filename, "testfile")
        for k in data.keys():
            npt.assert_equal(myobj[k], data[k])
        npt.assert_equal(myobj["Wind"], data["WindVxi"])

        mydict = myobj.to_dict()
        for k in data.keys():
            npt.assert_equal(mydict[k], data[k])

    def testAddChannel(self):
        # Adding same ones
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        myobj.add_channel(data)
        self.assertEqual(myobj.data.shape, (10,5))
        npt.assert_equal(myobj.data[:,0], np.array(data["Time"]))
        npt.assert_equal(myobj.data[:,1], np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,2], np.zeros(10))
        npt.assert_equal(myobj.data[:,3], np.zeros(10))
        self.assertEqual(myobj.channels, list(data.keys())+["Wind"])

        # Add in different types
        myobj.add_channel({"Test1":data["WindVxi"]})
        myobj.add_channel(pd.DataFrame({"Test2":data["WindVxi"]}))
        myobj.add_channel(data["WindVxi"], "Test3")
        myobj.add_channel(data["WindVxi"], ["Test4"])
        myobj.add_channel(data["WindVxi"], ("Test5",))
        myobj.add_channel(np.array(data["WindVxi"]), "Test6")
        myobj.add_channel(np.array(data["WindVxi"]), ["Test7"])
        self.assertEqual(myobj.data.shape, (10,12))
        for k in range(5,12):
            npt.assert_equal(myobj.data[:,k], np.array(data["WindVxi"]))

    def testCalcChannel(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        myobj.calculate_channel("2*WindVxi", "Test1")
        myobj.calculate_channel("WindVxi + WindVyi", "Test2")
        myobj.calculate_channel("WindVxi - WindVzi", "Test3")
        myobj.calculate_channel("WindVxi**2", "Test4")
        self.assertEqual(myobj.data.shape, (10,9))
        npt.assert_equal(myobj.data[:,5], 2*np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,6], np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,7], np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,8], np.array(data["WindVxi"])**2)

    def testTrim(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        myobj.trim_data(3, 6)
        self.assertEqual(myobj.data.shape, (4,5))
        npt.assert_equal(myobj.time, np.arange(3,7))

    def testLoadRose(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        myobj.add_load_rose({"WindLR":["WindVxi", "WindVyi"]})
        self.assertEqual(myobj.data.shape, (10,11))
        npt.assert_equal(myobj.data[:,5], np.array(data["WindVxi"]))
        npt.assert_equal(myobj.data[:,6], np.array(data["WindVxi"])*np.cos(np.deg2rad(60)))
        npt.assert_equal(myobj.data[:,7], np.array(data["WindVxi"])*np.cos(np.deg2rad(120)))
        npt.assert_equal(myobj.data[:,8], np.array(data["WindVxi"])*np.cos(np.deg2rad(180)))
        npt.assert_equal(myobj.data[:,9], np.array(data["WindVxi"])*np.cos(np.deg2rad(240)))
        npt.assert_equal(myobj.data[:,10], np.array(data["WindVxi"])*np.cos(np.deg2rad(300)))

    def testProperties(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        npt.assert_equal(myobj.num_timesteps, 10)
        npt.assert_equal(myobj.elapsed_time, 9)
        npt.assert_equal(myobj.num_channels, 5)
        npt.assert_equal(myobj.idxmins, np.zeros(5))
        npt.assert_equal(myobj.idxmaxs, [9, 5, 0, 0, 5])
        npt.assert_equal(myobj.minima, [1, 7, 0, 0, 7])
        npt.assert_equal(myobj.maxima, [10, 8, 0, 0, 8])
        npt.assert_equal(myobj.absmaxima, [10, 8, 0, 0, 8])
        npt.assert_equal(myobj.ranges, [9, 1, 0, 0, 1])
        npt.assert_equal(myobj.variable, [0, 1, 4])
        npt.assert_equal(myobj.constant, [2, 3])
        npt.assert_equal(myobj.sums.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.sums_squared.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.sums_cubed.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.sums_fourth.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.second_moments.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.third_moments.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.fourth_moments.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.means.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.medians.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.stddevs.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.skews.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.kurtosis.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.integrated.shape, (myobj.num_channels,))
        npt.assert_equal(myobj.compute_energy('WindVxi'), myobj.integrated[1])

    def test_windowing(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        myobj_w1 = myobj.time_averaging(2.0)
        npt.assert_equal(myobj_w1.data[:,0], np.array(data["Time"][:-1]) + 0.5)
        npt.assert_equal(myobj_w1.data[:,2], 0.0)
        npt.assert_equal(myobj_w1.data[:,3], 0.0)

        myobj_w2 = myobj.time_binning(2.0)
        npt.assert_equal(myobj_w2.data[:,0], np.array(data["Time"][::2]) + 0.5)
        npt.assert_equal(myobj_w2.data[:,2], 0.0)
        npt.assert_equal(myobj_w2.data[:,3], 0.0)

    def test_psd(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        f, P = myobj.psd()
        self.assertEqual(f.shape[0], P.shape[0])
        self.assertEqual(myobj.num_channels, P.shape[1])
        
    def test_stats_extremes(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)

        stats = myobj.get_summary_stats()
        self.assertEqual(stats["Time"]['min'], 1.)
        self.assertEqual(stats["Time"]['max'], 10.)
        self.assertEqual(stats["Time"]['abs'], 10.)
        self.assertEqual(stats["Time"]['mean'], 5.5)
        self.assertEqual(stats["Time"]['median'], 5.5)
        self.assertEqual(stats["Time"]['integrated'], 49.5)

        ext = myobj.extremes()
        self.assertEqual(ext["Time"], {"Time":10., "WindVxi":8., "WindVyi":0., "WindVzi":0, "Wind":8.})

    def test_dels(self):
        myparam = FatigueParams(lifetime=0.0,
                                load2stress = 2446,
                                slope = 3.0,
                                ult_stress = 6e8,
                                S_intercept = 5e9,
                                )
        t = np.linspace(0, 600, 10000)
        y0 = 40e3 * np.sin(2*np.pi*t/60.0)
        y1 = y0 + 40e3
        zeros = np.zeros(y0.shape)
        mydata = {"Time":t,
                  "Signal0":y0,
                  "Signal40":y1,
                  "Zeros":zeros}

        mymagnitudes = {"Mag0":["Signal0", "Zeros", "Zeros"],
                        "Mag40":["Signal40", "Zeros", "Zeros"]}

        myfatigues = {"Signal0":myparam,
                      "Signal40":myparam,
                      "Mag0":myparam,
                      "Mag40":myparam}

        myobj = AeroelasticOutput(mydata, magnitude_channels=mymagnitudes)
        
        dels = np.zeros(len(myfatigues))
        dams = np.zeros(len(myfatigues))
        for ik, k in enumerate(myfatigues.keys()):
            dels[ik], dams[ik] = myobj.compute_del(k, myparam, return_damage=True)

        self.assertAlmostEqual(dels[0], dels[1])
        self.assertGreater(dels[0], dels[2])
        self.assertAlmostEqual(dels[0], dels[3])
        
        self.assertAlmostEqual(dams[0], dams[1])
        self.assertGreater(dams[0], dams[2])
        self.assertAlmostEqual(dams[0], dams[3])
        
if __name__ == "__main__":
    unittest.main()
