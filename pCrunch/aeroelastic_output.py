import os
import fnmatch
import numpy as np
import pandas as pd
from scipy import stats, signal
import numexpr as ne
import fatpack

def dataproperty(f):
    @property
    def wrapper(self, *args, **kwargs):
        if getattr(self, "data", None) is None:
            raise AttributeError("Output has not been read yet.")
        return f(self, *args, **kwargs)

    return wrapper


class AeroelasticOutput:
    """Base timeseries aeroelastic simulation data output class."""

    def __init__(self, datain=None, channelsin=None, **kwargs):
        """
        Creates an instance of `AeroelasticOutput`.

        Parameters
        ----------
        data : list
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        channels : list (optional)
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        units : list (optional)
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        name : list (optional)
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        description : list (optional)
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        fatigue_channels : dict (optional)
            Dictionary with format:
            'channel': 'fatigue slope'
        extreme_stat : str (optional)
            Whether the extreme event calculation should work on [max, min, abs].
            Default, 'max'
        extreme_channels : list (optional)
            Limit calculation of extremes to the channel names in the list.
            Unspecified means all channels are processed and reported.
        magnitude_channels : dict (optional)
            Additional channels as vector magnitude of other channels.
            Format: 'new-chan': ['chan1', 'chan2', 'chan3']
        trim_data : tuple or list (optional)
            Trim processed outputs to desired times.
            Format: (min, max)
        """

        self.data        = None
        self.channels    = None
        self.description = kwargs.get("description", "")
        self.units       = kwargs.get("units", None)
        self.filepath    = ""

        for k in ["dlc", "DLC", "name", "Name", "NAME", "filename","filepath","file"]:
            self.filepath = kwargs.get(k, "")
            if len(self.filepath) > 0:
                break

        self.set_data(datain, channelsin)

        self.td = kwargs.get("trim_data", ())
        self.trim_data(*self.td)
        
        self.mc = kwargs.get("magnitude_channels", {})
        self.append_magnitude_channels()

        self.ec = kwargs.get("extreme_channels", [])
        statin  = kwargs.get("extreme_stat", "max")
        self.set_extreme_stat(statin)
        
        self.fc = kwargs.get("fatigue_channels", {})
        
    def __getitem__(self, chan):
        return self.data[:, self.chan_idx(chan)]

    def __str__(self):
        return self.description

    def copy(self):
        return AeroelasticOutput(self.data, self.channels, description=self.description,
                                 units=self.units, name=self.filepath)

    def save(self, fname):
        self.to_df().to_pickle(fname)

    def load(self, fname):
        self.set_data( pd.read_pickle(fname) )

    def set_extreme_stat(self, instr):
        if instr is None or instr == '':
            return
        
        if (not isinstance(instr, str) or
            instr.lower() not in ['max','min','abs','absmax','maximum','minimum','maxima','minima','absolute','abs max']):
            raise ValueError('Expecting stat to be a string of [max, min, abs]')
        
        self.extreme_stat = instr.lower()
        
    def chan_idx(self, chan):
        try:
            idx = self.channels.index(chan)

        except ValueError:
            raise IndexError(f"Channel '{chan}' not found.")
        return idx
    
    def set_data(self, datain, channelsin=None, dropna=True):
        if datain is None:
            return
        
        elif isinstance(datain, dict):
            self.channels = [c.strip() for c in list(datain.keys())]
            self.data     = np.array(list(datain.values())).T
            
        elif isinstance(datain, list):
            self.channels = [c.strip() for c in channelsin]
            self.data     = np.array(datain).T
            
        elif isinstance(datain, pd.DataFrame):
            self.channels = [c.strip() for c in list(datain.columns)]
            self.data     = datain.to_numpy()

        elif isinstance(datain, np.ndarray):
            self.channels = [c.strip() for c in channelsin]
            self.data     = datain

        else:
            pass
            #print("Unknown data type.  Expected dict or list or DataFrame or Numpy Array")
            #print(f"Instead found, {type(datain)}")

        if dropna:
            self.data = self.data[~np.isnan(self.data).any(axis=1),:]

    def drop_channel(self, pattern):
        """
        Drop channel based on a string pattern
        """
        idx = np.where([fnmatch.fnmatch(m, pattern) for m in self.channels])[0]
        self.channels = np.delete( np.array(self.channels), idx).tolist()
        self.data = np.delete(self.data, idx, axis=1)
        
    def delete_channel(self, pattern):
        self.drop_channel(pattern)
        
    def remove_channel(self, pattern):
        self.drop_channel(pattern)
        
    def _add_channel(self, datain, namein):
        """
        Add new channels of data.  ASSUMES NUMPY ARRAY
        """
        if namein in self.channels:
            pass
            #print(f"Channel '{namein}' already exists.")
        else:
            self.data = np.c_[self.data, datain]
            self.channels.append( namein )
        
    def add_channel(self, datain, namein=None):
        """
        Add new channels of data.

        Parameters
        ----------
        magnitude_channels : dict
            Format: 'new-chan' ['chan1', 'chan2', 'chan3'],
        """
        if isinstance(datain, dict):
            for c in datain.keys():
                self._add_channel(np.array(datain[c]), c.strip())
            
        elif isinstance(datain, list) and isinstance(datain[0], list) and isinstance(namein, (list, tuple, np.ndarray)):
            for k in range(len(namein)):
                self._add_channel(np.array(datain[k]), namein[k].strip())
            
        elif isinstance(datain, list) and isinstance(datain[0], (int, float)) and isinstance(namein, (list, tuple, np.ndarray)):
            self._add_channel(np.array(datain), namein[0].strip())
            
        elif isinstance(datain, list) and isinstance(namein, str):
            self._add_channel(np.array(datain), namein.strip())
            
        elif isinstance(datain, pd.DataFrame):
            for c in list(datain.columns):
                self._add_channel(datain[c].to_numpy(), c.strip())

        elif isinstance(datain, np.ndarray) and datain.ndim==2 and isinstance(namein, (list, tuple, np.ndarray)):
            for k in range(len(namein)):
                self._add_channel(datain[:,k], namein[k].strip())
                
        elif isinstance(datain, np.ndarray) and datain.ndim==1 and isinstance(namein, (list, tuple, np.ndarray)):
            self._add_channel(datain, namein[0].strip())
                
        elif isinstance(datain, np.ndarray) and isinstance(namein, str):
            self._add_channel(datain, namein.strip())

        elif isinstance(datain, str) and isinstance(namein, str):
            new_data = ne.evaluate(datain, local_dict=self.to_dict())
            self._add_channel(new_data, namein.strip())

        else:
            print("Unknown data type.  Expected dict or list or DataFrame or Numpy Array")
            print(f"Instead found, {type(datain)}")


    def calculate_channel(self, instr, namein):
        self.add_channel(instr, namein)
        
    def to_df(self):
        """Returns `self.data` as a DataFrame."""

        if self.channels is None:
            return pd.DataFrame(self.data)
        else:
            return pd.DataFrame(self.data, columns=self.channels)

    @dataproperty
    def df(self):
        return self.to_df()
        
    def to_dict(self):
        """Returns `self.data` as a dictionary."""

        return {self.channels[k]:self.data[:,k].tolist() for k in range(len(self.channels))}

    @dataproperty
    def time(self):
        if "Time" in self.channels:
            return self["Time"]
        else:
            return self.data[:,0]

    @property
    def filename(self):
        return os.path.split(self.filepath)[-1]
                
    @property
    def headers(self):
        if getattr(self, "units", None) is None:
            return None

        else:
            return [
                self.channels[k]
                if self.units[k] in ["", "-"]
                else f"{self.channels[k]} [{self.units[k]}]"
                for k in range(self.num_channels)
            ]
            
    def trim_data(self, tmin=0, tmax=np.inf):
        """
        Trims `self.data` to the data between `tmin` and `tmax`.

        Parameters
        ----------
        tmin : int | float
            Start time.
        tmax : int | float
            Ending time.
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.inf
        if tmin==0.0 and tmax==np.inf:
            return
        
        idx = np.where((self.time >= tmin) & (self.time <= tmax))[0]
        if len(idx) == 0:
            raise ValueError(
                f"No time steps after trimming. Provided: ({tmin}, {tmax}). Available: ({self.time.min()}, {self.time.max()})"
            )

        self.data = self.data[idx,:]

    def add_gradient_channel(self, chanstr, new_name):
        newdata = np.gradient(self[chanstr], self.time)
        self._add_channel(newdata, new_name)
        
    def append_magnitude_channels(self, magnitude_channels=None):
        """
        Append the vector magnitude of `channels` to the dataset.

        Parameters
        ----------
        magnitude_channels : dict
            Format: 'new-chan' ['chan1', 'chan2', 'chan3'],
        """

        if magnitude_channels is not None:
            if not isinstance(magnitude_channels, dict):
                raise ValueError("Expecting magnitude channels as a dictionary, 'new-chan': ['chan1', 'chan2', 'chan3']")
            
            self.mc.update( magnitude_channels)
        
        if self.mc is None:
            return
        
        for new_chan, chans in self.mc.items():

            try:
                arrays = np.array([self[a] for a in chans]).T
                normed = np.linalg.norm(arrays, axis=1).reshape(arrays.shape[0], 1)
            except:
                normed = np.nan * np.ones((self.data.shape[0],1))

            self._add_channel(normed, new_chan)
            
    def add_magnitude_channels(self, magnitude_channels=None):
        """
        Since add/append get confused quite a bit
        """
        self.append_magnitude_channels(magnitude_channels=magnitude_channels)

    def add_load_rose(self, load_rose=None, nsec=6):
        """
        Append a load rose of `nsec` sectors of x-y `channels` to the dataset.
        The x-channel will be multiplied by cos(theta) and y-channel by sin(theta)

        Parameters
        ----------
        load_rose : dict
            Format: 'new-chan' ['chan_x', 'chan_y'],
        """

        if load_rose is None:
            return
        else:
            if not isinstance(load_rose, dict):
                raise ValueError("Expecting load rose channels as a dictionary, 'new-chan': ['chan_x', 'chan_y']")

        th_pts = np.linspace(0, 180, nsec+1) # Don't need to repeat the 360 mark
        thd = 0.5*(th_pts[:-1] + th_pts[1:]) # Mid point of each sector
        
        for rootstr, chans in load_rose.items():
            for td in thd:
                tr = np.deg2rad(td)
                new_chan = rootstr + str(int(td))
                new_data = self[chans[0]]*np.cos(tr) + self[chans[1]]*np.sin(tr)
                self._add_channel(new_data, new_chan)
                print(f"Added channel, {new_chan}")
        
    @dataproperty
    def num_timesteps(self):
        return self.data.shape[0]
    
    @dataproperty
    def dt(self):
        return self.time[1] - self.time[0]

    @dataproperty
    def elapsed_time(self):
        return self.time.max() - self.time.min()

    @dataproperty
    def num_channels(self):
        return self.data.shape[1]

    @dataproperty
    def idxmins(self):
        return self.data.argmin(axis=0)

    @dataproperty
    def idxmaxs(self):
        return self.data.argmax(axis=0)

    @dataproperty
    def idxabs(self):
        return np.abs(self.data).argmax(axis=0)

    @dataproperty
    def minima(self):
        return self.data.min(axis=0)

    @dataproperty
    def maxima(self):
        return self.data.max(axis=0)

    @dataproperty
    def ranges(self):
        return self.maxima - self.minima

    @dataproperty
    def variable(self):
        return np.where(self.ranges != 0.0)[0]

    @dataproperty
    def constant(self):
        return np.where(self.ranges == 0.0)[0]

    @dataproperty
    def sums(self):
        return self.data.sum(axis=0)

    @dataproperty
    def sums_squared(self):
        return np.sum(self.data ** 2, axis=0)

    @dataproperty
    def sums_cubed(self):
        return np.sum(self.data ** 3, axis=0)

    @dataproperty
    def sums_fourth(self):
        return np.sum(self.data ** 4, axis=0)

    @dataproperty
    def second_moments(self):
        return stats.moment(self.data, moment=2, axis=0)

    @dataproperty
    def third_moments(self):
        return stats.moment(self.data, moment=3, axis=0)

    @dataproperty
    def fourth_moments(self):
        return stats.moment(self.data, moment=4, axis=0)

    @dataproperty
    def means(self):
        return self.data.mean(axis=0)

    @dataproperty
    def medians(self):
        return np.median(self.data, axis=0)

    @dataproperty
    def absmaxima(self):
        return np.abs(self.data).max(axis=0)

    @dataproperty
    def stddevs(self):
        return self.data.std(axis=0)

    @dataproperty
    def skews(self):
        return self.third_moments / np.sqrt(self.second_moments) ** 3

    @dataproperty
    def kurtosis(self):
        return self.fourth_moments / self.second_moments ** 2

    @dataproperty
    def integrated(self):
        return np.trapezoid(self.data, self.time, axis=0)

    def compute_energy(self, pwrchan):
        return np.trapezoid(self[pwrchan], self.time)

    def total_travel(self, chanstr):
        dchan = np.gradient(self[chanstr], self.time)
        return np.trapezoid(np.abs(dchan), self.time)

    def histogram(self, chanstr, bins=15):
        return np.histogram(self[chanstr], bins=bins, density=False)
    
    def density(self, chanstr, bins=15):
        return np.histogram(self[chanstr], bins=bins, density=True)
    
    def psd(self, nfft=None):
        """
        Compute power spectra density for each channel.

        Returns
        ----------
        freq : Numpy array 1-D
            Valid frequencies for spectra
        Pxx_den : Numpy array 2-D
            Power spectral density with each column corresponding to a channel
        """
        fs = 1. / np.diff(self.time)[0]
        freq, Pxx_den = signal.welch(self.data, fs, nfft=nfft, axis=0)
        fobj = self.copy()
        fobj.data = Pxx_den
        fobj.data[:,0] = freq
        fobj.channels[0] = 'Freq'
        return fobj

    def time_averaging(self, time_window):
        """
        Applies averaging/smoothing window on time-domain channels via convolution.
        Window input is seconds
        
        Parameters
        ----------
        time_windows : float
            Window for averaging/smoothing in seconds

        Returns
        ----------
        obj : AeroelasticOutput object
            New AeroelasticOutput instance with the new data
        """
        dt = np.diff(self.time)[0]
        npts = int(time_window / dt)
        window = np.ones(npts) / npts # Basic rectangular filter

        new_timesteps = max(npts,self.num_timesteps) - min(npts,self.num_timesteps) + 1
        data_avg = np.zeros((new_timesteps, self.num_channels))
        for k in range(len(self.channels)):
            data_avg[:,k] = np.convolve(self.data[:,k], window, 'valid')

        # Install the new data
        self.data = data_avg
        
    def averaging(self, window):
        self.time_averaging(window)

    
    def time_binning(self, time_window):
        """
        Bin the data into groups specified by time_window (in seconds)
        Average (or, someday, other stats of those windows)
        
        Parameters
        ----------
        time_windows : float
            Window for averaging/smoothing in seconds

        Returns
        ----------
        obj : AeroelasticOutput object
            New AeroelasticOutput instance with the new data
        """

        n_bins = int(np.ceil((self.time.max() - self.time.min()) / time_window))

        data_binned = np.zeros((n_bins,self.data.shape[1]))

        for i_bin in range(n_bins):
            t_start = i_bin * time_window  + self.time.min()
            t_end = (i_bin+1) * time_window    + self.time.min()
            time_index = np.logical_and(self.time >= t_start, self.time < t_end)
            if not np.any(time_index):
                raise Exception("Error binning time between {t_start} and {t_end}")
            data_filtered = self.data[time_index,:]
            data_binned[i_bin,:] = np.mean(data_filtered,axis=0)

        # Install the new data
        self.data = data_binned
        
    def binning(self, window):
        self.time_binning(window)

    def summary_stats(self):
        """
        Appends summary statistics to `self.summary_statistics` for each file.
        """

        mins = self.minima
        maxs = self.maxima
        stds = self.stddevs
        means = self.means
        meds  = self.medians
        abss  = self.absmaxima
        intgs = self.integrated
        
        fstats = {}
        for k in range(len(self.channels)):
            channel = self.channels[k]
            
            fstats[channel] = {
                "min": mins[k],
                "max": maxs[k],
                "std": stds[k],
                "mean": means[k],
                "median": meds[k],
                "abs": abss[k],
                "integrated": intgs[k],
            }

        return fstats

    
    def extremes(self, chanlist=None, stat=None):
        """"""
        if chanlist is not None:
            if not isinstance(chanlist, (list, set, tuple)):
                raise ValueError("Expecting extreme channels as a list, tuple, or set")
            
            self.ec = list(set(self.ec + chanlist))
        
        if self.ec is None or len(self.ec) == 0:
            channels = self.channels
        else:
            channels = self.ec

        sorter = np.argsort(self.channels)
        exists = [c for c in channels if c in self.channels]
        idx = sorter[np.searchsorted(self.channels, exists, sorter=sorter)]

        # Get statistic
        self.set_extreme_stat(stat)
        if self.extreme_stat.find('abs') >= 0:
            mystat = self.idxabs
        elif self.extreme_stat.find('min') >= 0:
            mystat = self.idxmins
        elif self.extreme_stat.find('max') >= 0:
            mystat = self.idxmaxs
        
        extremes = {}
        for chan, i in zip(exists, idx):
            idx_stat = mystat[i]
            extremes[chan] = {
                "Time": self.time[idx_stat],
                **dict(zip(exists, self.data[idx_stat, idx])),
            }

        return extremes


        
    def get_DELs(self, **kwargs):
        """
        Appends computed damage equivalent loads for fatigue channels in
        `self.fc`.

        Parameters
        ----------
        output : AerolelasticOutput
        rainflow_bins : int (optional)
            Number of bins used in rainflow analysis.
            Default: 100
        goodman_correction: boolean (optional)
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        """
        for k in kwargs:
            if k not in ["rainflow_bins", "bins",
                         "goodman_correction", "goodman"]:
                print(f"Unknown keyword argument, {k}")

        DELs = {}
        D = {}

        for chan, fatparams in self.fc.items():
            goodman = kwargs.get("goodman_correction", fatparams.goodman)
            goodman = kwargs.get("goodman", goodman)
            bins = kwargs.get("rainflow_bins", fatparams.bins)
            bins = kwargs.get("bins", bins)
                
            try:
                if np.isnan(np.sum(self[chan])):  # channel has a nan
                    raise IndexError

                DELs[chan] = fatparams.compute_del(self[chan], self.elapsed_time,
                                                   goodman_correction=goodman,
                                                   rainflow_bins=bins)
                
                if np.abs(fatparams.load2stress) > 0.0:
                    D[chan] = fatparams.compute_damage(self[chan],
                                                       goodman_correction=goodman,
                                                       rainflow_bins=bins)
                else:
                    D[chan] = 0.0

            except IndexError:
                print(f"Channel '{chan}' not included in DEL calculation.")
                DELs[chan] = np.nan
                D[chan] = np.nan

        return DELs, D

    
    def process(self, **kwargs):
        """
        Process AeroelasticOutput output for summary stats, extreme event table, DELs, and Damage
        """

        # Data manipulation list all other outputs
        self.trim_data(*self.td)
        self.append_magnitude_channels(self.mc)
        
        self.stats = self.summary_stats()

        self.ext_table = self.extremes()

        self.dels, self.damage = self.get_DELs(**kwargs)

