import os
import unittest
import numpy.testing as npt

from pCrunch import OpenFASTAscii, OpenFASTBinary, read, load_FAST_out

DIR = os.path.split(__file__)[0]
DATA = os.path.join(DIR, "data")

FOUT  = ['AOC_WSt.out', 'DLC2.3_1.out', 'DLC2.3_2.out', 'DLC2.3_3.out']
FOUTB = ['AOC_WSt.outb', 'Test1.outb', 'Test2.outb', 'Test3.outb', 'step_0.outb']


class Test_OpenFAST_readers(unittest.TestCase):
    
    def testOFAscii(self):
        for k, kf in enumerate(FOUT):
            fname = os.path.join(DATA, kf)
            with self.subTest(f"Reading: {kf}", i=k):
                output = OpenFASTAscii(fname)
                output.read()
                self.assertGreater(output.data.shape[0], 1)
                self.assertGreater(output.data.shape[1], 1)
                self.assertGreater(output.num_channels, 1)
    
    def testOFBinary(self):
        for k, kf in enumerate(FOUTB):
            fname = os.path.join(DATA, kf)
            with self.subTest(f"Reading: {kf}", i=k):
                output = OpenFASTBinary(fname)
                output.read()
                self.assertGreater(output.data.shape[0], 1)
                self.assertGreater(output.data.shape[1], 1)
                self.assertGreater(output.num_channels, 1)
    
    def testOFReader(self):
        for k, kf in enumerate(FOUT+FOUTB):
            fname = os.path.join(DATA, kf)
            with self.subTest(f"Reading: {kf}", i=k):
                output = read(fname)
                self.assertGreater(output.data.shape[0], 1)
                self.assertGreater(output.data.shape[1], 1)
                self.assertGreater(output.num_channels, 1)
    
    def testBatchReader(self):
        myfiles = [os.path.join(DATA, m) for m in FOUT+FOUTB]
        outputs = load_FAST_out(myfiles)
        for k, kf in enumerate(outputs):
            with self.subTest(f"Reading, {kf.filename}", i=k):
                self.assertGreater(kf.data.shape[0], 1)
                self.assertGreater(kf.data.shape[1], 1)
                self.assertGreater(kf.num_channels, 1)
    
    def testSame(self):
        outputA = read(os.path.join(DATA, FOUT[0]))
        outputB = read(os.path.join(DATA, FOUTB[0]))
        npt.assert_allclose(outputA.data, outputB.data, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
