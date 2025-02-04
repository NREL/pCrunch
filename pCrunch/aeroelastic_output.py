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
        extreme_channels : list (optional)
            Limit calculation of extremes to the channel names in the list.  Unspecified means all channels are processed and reported.
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
        
    def chan_idx(self, chan):
        try:
            idx = self.channels.index(chan)

        except ValueError:
            raise IndexError(f"Channel '{chan}' not found.")
        return idx
    
    def set_data(self, datain, channelsin=None):
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
        return self["Time"]

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

        thd = np.linspace(0, 360, nsec+1)[:-1] # Don't need to repeat the 360 mark
        
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
        return np.trapz(self.data, self.time, axis=0)

    def compute_energy(self, pwrchan):
        return np.trapz(self[pwrchan], self.time)

    def total_travel(self, chanstr):
        dchan = np.gradient(self[chanstr], self.time)
        return np.trapz(np.abs(dchan), self.time)
    
    def psd(self):
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
        freq, Pxx_den = signal.welch(self.data, fs, axis=0)
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
                print('here')
            data_filtered = self.data[time_index,:]
            data_binned[i_bin,:] = np.mean(data_filtered,axis=0)

        # Install the new data
        self.data = data_binned


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

    
    def extremes(self, chanlist=None):
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

        extremes = {}
        for chan, i in zip(exists, idx):
            idx_max = self.idxmaxs[i]
            extremes[chan] = {
                "Time": self.time[idx_max],
                **dict(zip(exists, self.data[idx_max, idx])),
            }

        return extremes


    def compute_del(self, chan, fatparams, **kwargs):
        """
        Computes damage equivalent load of input `chan`.

        Parameters
        ----------
        chan : str or np.array
            Channel name or time series to calculate DEL for.
        fatparams : FatigueParameters object
            Instance of FatigueParameters with all material and geometry parameters
        lifetime : int | float (optional kwargs)
            Design lifetime of the component / material in years. If specified, overrides fatparams value.
        load2stress : float (optional kwargs)
            Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L
            If specified, overrides fatparams value.
        slope : int | float (optional kwargs)
            Slope of the fatigue curve.  If specified, overrides fatparams value.
        ultimate_stress : float (optional kwargs)
            Ultimate stress for use in Goodman equivalent stress calculation.  If specified, overrides fatparams value.
        S_intercept : float (optional kwargs)
            Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified.
            If specified, overrides fatparams value.
        rainflow_bins : int (optional kwargs)
            Number of bins used in rainflow analysis.
            Default: 100
        goodman_correction: boolean (optional kwargs)
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        return_damage: boolean (optional kwargs)
            Whether to compute both DEL and damage
            Default: False
        """

        lifetime      = kwargs.get("lifetime", fatparams.lifetime)
        load2stress   = kwargs.get("load2stress", fatparams.load2stress)
        slope         = kwargs.get("slope", fatparams.slope)
        Sult          = kwargs.get("ultimate_stress", fatparams.ult_stress)
        Sc            = kwargs.get("S_intercept", fatparams.S_intercept)
        Scin          = Sc if Sc > 0.0 else Sult
        bins          = kwargs.get("rainflow_bins", fatparams.bins)
        return_damage = kwargs.get("return_damage", fatparams.return_damage)
        return_damage = kwargs.get("compute_damage", return_damage)
        goodman       = kwargs.get("goodman_correction", fatparams.goodman)
        goodman       = kwargs.get("goodman", goodman)
        elapsed       = self.elapsed_time
        if isinstance(chan, str):
            chan = self[chan]

        for k in kwargs:
            if k not in ["lifetime", "load2stress", "slope", "ultimate_stress",
                         "S_intercept", "rainflow_bins", "return_damage", "compute_damage",
                         "goodman_correction", "goodman"]:
                print(f"Unknown keyword argument, {k}")
            
        # Default return values
        DEL = np.nan
        D   = np.nan

        if np.all(np.isnan(chan)):
            return DEL, D

        # Working with loads for DELs
        try:
            F, Fmean = fatpack.find_rainflow_ranges(chan, return_means=True)
        except Exception:
            F = Fmean = np.zeros(1)
        if goodman and np.abs(load2stress) > 0.0:
            F = fatpack.find_goodman_equivalent_stress(F, Fmean, Sult/np.abs(load2stress))
        Nrf, Frf = fatpack.find_range_count(F, bins)
        DELs = Frf ** slope * Nrf / elapsed
        DEL = DELs.sum() ** (1.0 / slope)
        # With fatpack do:
        #curve = fatpack.LinearEnduranceCurve(1.)
        #curve.m = slope
        #curve.Nc = elapsed
        #DEL = curve.find_miner_sum(np.c_[Frf, Nrf]) ** (1 / slope)

        # Compute Palmgren/Miner damage using stress
        D = np.nan # default return value
        if return_damage and np.abs(load2stress) > 0.0:
            try:
                S, Mrf = fatpack.find_rainflow_ranges(chan*load2stress, return_means=True)
            except Exception:
                S = Mrf = np.zeros(1)
            if goodman:
                S = fatpack.find_goodman_equivalent_stress(S, Mrf, Sult)
            Nrf, Srf = fatpack.find_range_count(S, bins)
            curve = fatpack.LinearEnduranceCurve(Scin)
            curve.m = slope
            curve.Nc = 1
            D = curve.find_miner_sum(np.c_[Srf, Nrf])
            if lifetime > 0.0:
                D *= lifetime*365.0*24.0*60.0*60.0 / elapsed

        return DEL, D


        
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
        return_damage: boolean (optional)
            Whether to compute both DEL and damage
            Default: False
        """
        for k in kwargs:
            if k not in ["rainflow_bins", "return_damage", "compute_damage", "goodman_correction", "goodman"]:
                print(f"Unknown keyword argument, {k}")
            

        DELs = {}
        D = {}

        for chan, fatparams in self.fc.items():
            goodman = kwargs.get("goodman_correction", fatparams.goodman)
            goodman = kwargs.get("goodman", goodman)
            bins = kwargs.get("rainflow_bins", fatparams.bins)
            return_damage = kwargs.get("return_damage", fatparams.return_damage)
            return_damage = kwargs.get("compute_damage", return_damage)
                
            try:
                DELs[chan], D[chan] = self.compute_del(chan, fatparams,
                                                       goodman_correction=goodman,
                                                       rainflow_bins=bins,
                                                       return_damage=return_damage)

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

