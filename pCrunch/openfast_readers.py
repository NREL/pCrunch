import numpy as np
from .aeroelastic_output import AeroelasticOutput

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

    batch_flag = isinstance(filename, (list, tuple, set))
    filelist = filename if batch_flag else [filename]
    
    fastout = []
    for k in filelist:
        try:
            output = OpenFASTAscii(k, **kwargs)
            output.read()
            
        except UnicodeDecodeError:
            output = OpenFASTBinary(k, **kwargs)
            output.read()
            
        fastout.append(output)

    if batch_flag:
        return fastout
    else:
        return fastout[0]
    
def read_parallel(filenames, n_cores=1, **kwargs):
    """
    Read OpenFAST files in parallel.

    Parameters
    ----------
    filenames : list of str
        List of OpenFAST file paths.
    n_cores : int, optional
        Number of cores to use for parallel reading. Default is 1.

    Returns
    -------
    list of OpenFASTOutput
        List of OpenFASTOutput instances.
    """
    
    import multiprocessing
    with multiprocessing.Pool(n_cores) as pool:
        outputs = pool.map(read, filenames, **kwargs)
    
    return outputs

    
def load_FAST_out(filenames, tmin=0, tmax=float('inf'), **kwargs):
    """ Backwards compatibility with old name """
    fastout = read(filenames, tmin=tmin, tmax=tmax, **kwargs)
    for k in fastout:
        k.trim_data(tmin, tmax)
    return fastout



class OpenFASTBinary(AeroelasticOutput):
    """OpenFAST binary output class."""

    def __init__(self, filepath, **kwargs):
        """
        Creates an instance of `OpenFASTBinary`.

        Parameters
        ----------
        filepath : path-like
        """

        super().__init__()
        self.filepath = filepath
        self._chan_chars = kwargs.get("chan_char_length", 10)
        self._unit_chars = kwargs.get("unit_char_length", 10)
        mc = kwargs.get("magnitude_channels", None)
        self.read(magnitude_channels=mc)
        self.fmt = 0

    def read(self, magnitude_channels=None):
        """Reads the binary file."""

        with open(self.filepath, "rb") as f:
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
            self.description = "".join(map(chr, chars)).strip()

            # Build headers
            channels = np.fromfile(
                f, np.uint8, self._chan_chars * (num_channels + 1)
            ).reshape((num_channels + 1), self._chan_chars)
            channels_list = list("".join(map(chr, c)) for c in channels)
            self.channels = [c.strip() for c in channels_list]

            units = np.fromfile(
                f, np.uint8, self._unit_chars * (num_channels + 1)
            ).reshape((num_channels + 1), self._unit_chars)
            self.units = list("".join(map(chr, c)).strip()[1:-1] for c in units)
            
            # Build time
            if self.fmt == 1:
                scale, offset = time_info
                data = np.fromfile(f, np.int32, num_timesteps)
                time = (data - offset) / scale

            else:
                t1, incr = time_info
                time = t1 + incr * np.arange(num_timesteps)

            # Build the data
            if self.fmt == 3:
                raw = np.fromfile(f, np.float64, count=num_points).reshape(num_timesteps, num_channels)
                self.data = np.concatenate([time.reshape(num_timesteps, 1),
                                            raw], 1)

            else:
                raw = np.fromfile(f, np.int16, count=num_points).reshape(num_timesteps, num_channels)
                self.data = np.concatenate([time.reshape(num_timesteps, 1),
                                            (raw - offset) / slopes],
                                           1)

        self.append_magnitude_channels(magnitude_channels=magnitude_channels)


class OpenFASTAscii(AeroelasticOutput):
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
        self.filepath = filepath
        mc = kwargs.get("magnitude_channels", None)
        self.read(magnitude_channels=mc)

    def read(self, magnitude_channels=None):
        """Reads the ASCII file."""

        with open(self.filepath, "rb") as f:
            chandata, unitdata = self.parse_header(f)
            self.build_headers(chandata, unitdata)
            self.data = np.fromfile(f, float, sep="\t").reshape(
                -1, len(self.channels)
            )

        self.append_magnitude_channels(magnitude_channels=magnitude_channels)

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

        self.description = " ".join([h.replace('"', "") for h in data[:-2]]).strip()

        chandata, unitdata = data[-2:]
        return chandata, unitdata

    def build_headers(self, chandata, unitdata):
        """Unpacks channels and units and builds the combined headers."""

        self.channels = [c.strip() for c in chandata.split("\t")]
        self.units = [u[1:-1] for u in unitdata.split("\t")]

        
