import os

import numpy as np
from pCruncio.io import TimeseriesBase

def read(filename, **kwargs):
    """
    Load a single OpenFAST file.

    Parameters
    ----------
    filename : Path object or string
        OpenFAST file to load.

    Returns
    -------
    output
        OpenFASTOutput instances
    """

    try:
        output = OpenFASTAscii(filename, **kwargs)
        output.read()

    except UnicodeDecodeError:
        output = OpenFASTBinary(filename, **kwargs)
        output.read()

    return output


def load_OpenFAST_batch(filenames, tmin=0, tmax=float('inf'), **kwargs):
    """
    Load a list of OpenFAST files.

    Parameters
    ----------
    filenames : list
        List OpenFAST files to load.
    tmin : float | int, optional
        Initial line to trim output data to.
    tmax : float | int, optional

    Returns
    -------
    list
        List of OpenFASTOutput instances
    """

    if isinstance(filenames, str):
        filenames = [filenames]

    fastout = []
    for fn in filenames:
        output = read(fn, **kwargs)
        output.trim_data(tmin, tmax)
        fastout.append(output)

    return fastout

def load_FAST_out(filenames, tmin=0, tmax=float('inf'), **kwargs):
    """ Backwards compatibility with old name """
    return load_OpenFAST_batch(filenames, tmin=tmin, tmax=tmax, **kwargs)



class OpenFASTBinary(TimeseriesBase):
    """OpenFAST binary output class."""

    def __init__(self, filepath, **kwargs):
        """
        Creates an instance of `OpenFASTBinary`.

        Parameters
        ----------
        filepath : path-like
        """

        super().__init__()
        self.set_filename( filepath )
        self._chan_chars = kwargs.get("chan_char_length", 10)
        self._unit_chars = kwargs.get("unit_char_length", 10)
        self.magnitude_channels = kwargs.get("magnitude_channels", {})
        self.read()
        self.fmt = 0

    def read(self):
        """Reads the binary file."""

        with open(self._filepath, "rb") as f:
            self.fmt = np.fromfile(f, np.int16, 1)[0]

            if self.fmt == 4:
                self._chan_chars = np.fromfile(f, np.int16, 1)[0]
                self._unit_chars = self._chan_chars

            num_channels = np.fromfile(f, np.int32, 1)[0]
            num_timesteps = np.fromfile(f, np.int32, 1)[0]
            num_points = num_channels * num_timesteps
            time_info = np.fromfile(f, np.float64, 2)

            if self.fmt == 3:
                slopes = np.ones(num_channels)
                offset = np.zeros(num_channels)

            else:
                slopes = np.fromfile(f, np.float32, num_channels)
                offset = np.fromfile(f, np.float32, num_channels)

            length = np.fromfile(f, np.int32, 1)[0]
            chars = np.fromfile(f, np.uint8, length)
            self._desc = "".join(map(chr, chars)).strip()

            self.build_headers(f, num_channels)
            time = self.build_time(f, time_info, num_timesteps)

            if self.fmt == 3:
                raw = np.fromfile(f, np.float64, count=num_points).reshape(num_timesteps, num_channels)
                self.data = np.concatenate([time.reshape(num_timesteps, 1),
                                            raw], 1)

            else:
                raw = np.fromfile(f, np.int16, count=num_points).reshape(num_timesteps, num_channels)
                self.data = np.concatenate([time.reshape(num_timesteps, 1),
                                            (raw - offset) / slopes],
                                           1)

        self.append_magnitude_channels()

    def build_headers(self, f, num_channels):
        """
        Builds the channels, units and headers arrays.

        Parameters
        ----------
        f : file
        num_channels : int
        """

        channels = np.fromfile(
            f, np.uint8, self._chan_chars * (num_channels + 1)
        ).reshape((num_channels + 1), self._chan_chars)
        channels_list = list("".join(map(chr, c)) for c in channels)
        self.channels = np.array( [c.strip() for c in channels_list] )

        units = np.fromfile(
            f, np.uint8, self._unit_chars * (num_channels + 1)
        ).reshape((num_channels + 1), self._unit_chars)
        self.units = np.array(
            list("".join(map(chr, c)).strip()[1:-1] for c in units)
        )

    def build_time(self, f, info, length):
        """
        Builds the time index based on the file format and index info.

        Parameters
        ----------
        f : file
        info : tuple
            Time index meta information, ('scale', 'offset') or ('t1', 'incr').

        Returns
        -------
        np.ndarray
            Time index for `self.data`.
        """

        if self.fmt == 1:
            scale, offset = info
            data = np.fromfile(f, np.int32, length)
            time = (data - offset) / scale

        else:
            t1, incr = info
            time = t1 + incr * np.arange(length)

        return time


class OpenFASTAscii(TimeseriesBase):
    """
    OpenFAST ASCII output class.

    Built using data from the Mlife post-processor scripts. May not emcompass
    new OpenFAST output formats.
    """

    def __init__(self, filepath, **kwargs):
        """
        Creates an instance of `OpenFASTAscii`.

        Parameters
        ----------
        filepath : path-like
        """

        super().__init__()
        self._filepath = filepath
        self.magnitude_channels = kwargs.get("magnitude_channels", {})
        self.read()

    @property
    def time(self):
        return self.data[:, 0]

    def read(self):
        """Reads the ASCII file."""

        with open(self._filepath, "rb") as f:
            chandata, unitdata = self.parse_header(f)
            self.build_headers(chandata, unitdata)
            self.data = np.fromfile(f, float, sep="\t").reshape(
                -1, len(self.channels)
            )

        self.append_magnitude_channels()

    def parse_header(self, f):
        """Reads the header data for file."""

        _start = False
        data = []

        for _line in f:
            try:
                line = _line.replace(b"\xb7", b"-").decode(encoding="ascii").strip()
            except Exception:
                line = _line.replace(b"\xb7", b"-").decode(encoding="utf-16").strip()
            data.append(line)

            if _start:
                break

            if line.startswith("Time"):
                _start = True

        self._desc = " ".join([h.replace('"', "") for h in data[:-2]]).strip()

        chandata, unitdata = data[-2:]
        return chandata, unitdata

    def build_headers(self, chandata, unitdata):
        """Unpacks channels and units and builds the combined headers."""

        self.channels = np.array([c.strip() for c in chandata.split("\t")])
        self.units = np.array([u[1:-1] for u in unitdata.split("\t")])

        
