import os
import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
#import pandas.testing as pdt

from pCrunch import AeroelasticOutput

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
        self.assertEqual(myobj.extreme_stat, "max")
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, {})
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})

        # Test keywords
        myobj = AeroelasticOutput(description="test", units=["m"]*5, name="label", extreme_stat='abs')
        self.assertEqual(myobj.data, None)
        self.assertEqual(myobj.channels, None)
        self.assertEqual(myobj.units, ["m"]*5)
        self.assertEqual(myobj.description, "test")
        self.assertEqual(myobj.filepath, "label")
        self.assertEqual(myobj.extreme_stat, "abs")
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, {})
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})

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
        self.assertEqual(myobj.extreme_stat, "max")
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, mc)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})

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
        self.assertEqual(myobj.extreme_stat, "max")
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, mc)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})

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
        self.assertEqual(myobj.extreme_stat, "max")
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, mc)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})

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
        self.assertEqual(myobj.extreme_stat, "max")
        self.assertEqual(myobj.td, ())
        self.assertEqual(myobj.mc, mc)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})

        # With trimming
        myobj = AeroelasticOutput(data, magnitude_channels=mc, trim_data=[3,6])
        self.assertEqual(myobj.data.shape, (4,5))
        npt.assert_equal(myobj.data[:,0], np.array(data["Time"][2:6]))
        npt.assert_equal(myobj.data[:,1], np.array(data["WindVxi"][2:6]))
        npt.assert_equal(myobj.data[:,2], np.zeros(4))
        npt.assert_equal(myobj.data[:,3], np.zeros(4))
        npt.assert_equal(myobj.data[:,4], np.array(data["WindVxi"][2:6]))
        self.assertEqual(myobj.channels, list(data.keys())+["Wind"])
        self.assertEqual(myobj.units, None)
        self.assertEqual(myobj.description, "")
        self.assertEqual(myobj.filepath, "")
        self.assertEqual(myobj.extreme_stat, "max")
        self.assertEqual(myobj.td, [3, 6])
        self.assertEqual(myobj.mc, mc)
        self.assertEqual(myobj.ec, [])
        self.assertEqual(myobj.fc, {})
        
    def testGetters(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc, dlc="/testdir/testfile")
        npt.assert_equal(myobj.time, data["Time"])
        npt.assert_equal(myobj.filepath, "/testdir/testfile")
        npt.assert_equal(myobj.filename, "testfile")
        for k in data.keys():
            npt.assert_equal(myobj[k], data[k])
        npt.assert_equal(myobj["Wind"], data["WindVxi"])
        self.assertEqual(myobj.chan_idx("WindVxi"), 1)

        mydict = myobj.to_dict()
        for k in data.keys():
            npt.assert_equal(mydict[k], data[k])

    def testAddDeleteChannel(self):
        # Adding same ones
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        self.assertEqual(myobj.data.shape, (10,5))
        myobj.add_channel(data) # SHouldn't add anything
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

        # Add gradient
        myobj.add_gradient_channel('WindVxi', 'Test8')
        npt.assert_almost_equal(myobj['Test8'], np.r_[np.zeros(4), 0.5, 0.5, np.zeros(4)])
            
        # Now delete them
        myobj.drop_channel('Test*')
        self.assertEqual(myobj.data.shape, (10,5))

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
        npt.assert_equal(myobj.data[:,5], np.array(data["WindVxi"])*np.cos(np.deg2rad(15)))
        npt.assert_equal(myobj.data[:,6], np.array(data["WindVxi"])*np.cos(np.deg2rad(45)))
        npt.assert_equal(myobj.data[:,7], np.array(data["WindVxi"])*np.cos(np.deg2rad(75)))
        npt.assert_equal(myobj.data[:,8], np.array(data["WindVxi"])*np.cos(np.deg2rad(105)))
        npt.assert_equal(myobj.data[:,9], np.array(data["WindVxi"])*np.cos(np.deg2rad(135)))
        npt.assert_equal(myobj.data[:,10], np.array(data["WindVxi"])*np.cos(np.deg2rad(165)))

    def testProperties(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        npt.assert_equal(myobj.num_timesteps, 10)
        npt.assert_equal(myobj.dt, 1)
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
        npt.assert_equal(myobj.total_travel('WindVxi'), 1.0)

    def test_windowing(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        myobj.time_averaging(2.0)
        npt.assert_equal(myobj.data[:,0], np.array(data["Time"][:-1]) + 0.5)
        npt.assert_equal(myobj.data[:,2], 0.0)
        npt.assert_equal(myobj.data[:,3], 0.0)

        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        myobj.time_binning(2.0)
        npt.assert_equal(myobj.data[:,0], np.array(data["Time"][::2]) + 0.5)
        npt.assert_equal(myobj.data[:,2], 0.0)
        npt.assert_equal(myobj.data[:,3], 0.0)

    def test_psd(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        freq_obj = myobj.psd()
        self.assertEqual(myobj.num_channels, freq_obj.num_channels)
        
    def test_stats_extremes(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)

        stats = myobj.summary_stats()
        self.assertEqual(stats["Time"]['min'], 1.)
        self.assertEqual(stats["Time"]['max'], 10.)
        self.assertEqual(stats["Time"]['abs'], 10.)
        self.assertEqual(stats["Time"]['mean'], 5.5)
        self.assertEqual(stats["Time"]['median'], 5.5)
        self.assertEqual(stats["Time"]['integrated'], 49.5)

        ext = myobj.extremes()
        self.assertEqual(ext["Time"], {"Time":10., "WindVxi":8., "WindVyi":0., "WindVzi":0, "Wind":8.})

        ext = myobj.extremes(stat='min')
        self.assertEqual(ext["Time"], {"Time":1., "WindVxi":7., "WindVyi":0., "WindVzi":0, "Wind":7.})
        self.assertEqual(myobj.extreme_stat, "min")

        ext = myobj.extremes(stat='absmax')
        self.assertEqual(ext["Time"], {"Time":10., "WindVxi":8., "WindVyi":0., "WindVzi":0, "Wind":8.})
        self.assertEqual(myobj.extreme_stat, "absmax")

        
    def test_process(self):
        myobj = AeroelasticOutput(data, magnitude_channels=mc)
        
        stats = myobj.summary_stats()
        ext_tab = myobj.extremes()
        self.assertFalse(hasattr(myobj, 'stats'))

        myobj.process()
        self.assertTrue(hasattr(myobj, 'stats'))
        self.assertEqual(myobj.stats, stats)
        self.assertEqual(myobj.ext_table["Time"], ext_tab["Time"])
        
if __name__ == "__main__":
    unittest.main()
