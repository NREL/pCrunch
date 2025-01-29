import os
import unittest
import importlib
from time import time
from pathlib import Path

thisdir = os.path.dirname(os.path.realpath(__file__))
# Find examples directory- outside the module path

root_dir = os.path.dirname(os.path.dirname(thisdir))
examples_dir = os.path.join(root_dir, "examples")
all_examples = Path(examples_dir).rglob("*.py") if os.path.exists(examples_dir) else []

def execute_script(fscript):
    # Go to location due to relative path use for airfoil files
    print("\n\n")
    print("NOW RUNNING:", fscript)
    print()
    fullpath = os.path.join(examples_dir, fscript)
    basepath = os.path.join(examples_dir, str(fscript).split("/")[0])
    os.chdir(basepath)

    # Get script/module name
    froot = str(fscript).split("/")[-1]

    # Use dynamic import capabilities
    # https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
    print(froot, os.path.realpath(fullpath))
    spec = importlib.util.spec_from_file_location(froot, os.path.realpath(fullpath))
    mod = importlib.util.module_from_spec(spec)
    s = time()
    spec.loader.exec_module(mod)
    print(time() - s, "seconds to run")


class TestExamples(unittest.TestCase):
    def test_all_scripts(self):
        for ks, s in enumerate(all_examples):
            with self.subTest(f"Running: {s}", i=ks):
                try:
                    print(s)
                    execute_script(s)
                    self.assertTrue(True)
                except Exception:
                    self.assertEqual(s, "Success")

    
if __name__ == "__main__":
    unittest.main()
    
