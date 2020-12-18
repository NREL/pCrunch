__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os
import fnmatch

import pytest

from pCrunch.io import OpenFASTAscii, OpenFASTBinary

DIR = os.path.split(__file__)[0]
DATA = os.path.join(DIR, "data")


@pytest.mark.parametrize("fn", fnmatch.filter(os.listdir(DATA), "*.out"))
def test_OpenFASTAscii(fn, magnitude_channels):

    filepath = os.path.join(DATA, fn)
    output = OpenFASTAscii(filepath, magnitude_channels=magnitude_channels)
    output.read()

    assert output.data.shape
    assert output.channels.shape
    assert "Wind" in output.channels


@pytest.mark.parametrize("fn", fnmatch.filter(os.listdir(DATA), "*.outb"))
def test_OpenFASTBinary(fn, magnitude_channels):

    filepath = os.path.join(DATA, fn)
    output = OpenFASTBinary(filepath, magnitude_channels=magnitude_channels)
    output.read()

    assert output.data.shape
    assert output.channels.shape
    assert "Wind" in output.channels
