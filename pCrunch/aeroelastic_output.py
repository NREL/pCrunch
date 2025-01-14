import os
import numpy as np
import pandas as pd
from scipy import stats, signal


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
        # Initialize all properties
        self.data        = None
        self.channels    = None
        self.description = ""
        self.units       = None
        self.filepath    = ""
        self.magnitude_channels = None

        for k in ["dlc", "DLC", "name", "Name", "NAME", "filename","filepath","file"]:
            self.filepath = kwargs.get(k, "")
            if len(self.filepath) > 0:
                break

        self.set_data(datain, channelsin)

        self.append_magnitude_channels( kwargs.get("magnitude_channels", {}) )
        
    def __getitem__(self, chan):
        try:
            idx = np.where(self.channels == chan)[0][0]

        except IndexError:
            raise IndexError(f"Channel '{chan}' not found.")

        return self.data[:, idx]

    def __str__(self):
        return self.description


    def set_data(self, datain, channelsin=None):
        if isinstance(datain, dict):
            self.channels = np.array([c.strip() for c in list(datain.keys())]) 
            self.data     = np.array(list(datain.values())).T
            
        elif isinstance(datain, list):
            self.channels = np.array([c.strip() for c in channelsin])
            self.data     = np.array(datain)
            
        elif isinstance(datain, pd.DataFrame):
            self.channels = np.array([c.strip() for c in list(datain.columns)])
            self.data     = datain.to_numpy()

        elif isinstance(datain, np.ndarray):
            self.channels = np.array([c.strip() for c in channelsin])
            self.data     = datain

        else:
            pass
        
    
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

        idx = np.where((self.time >= tmin) & (self.time <= tmax))
        if tmin > max(self.time):
            raise ValueError(
                f"Initial time '{tmin}' is after the end of the simulation."
            )

        self.data = self.data[idx]

    @dataproperty
    def time(self):
        return self["Time"]

    @time.setter
    def time(self, time):
        if "Time" in self.channels:
            raise ValueError("'Time' channel already exists in output data.")

        self.data = np.append(time, self.data, axis=1)
        self.channels = np.append("Time", self.channels)

    @property
    def filename(self):
        return os.path.split(self.filepath)[-1]
        
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
            
            if self.magnitude_channels is None:
                self.magnitude_channels = magnitude_channels
            else:
                self.magnitude_channels.update(magnitude_channels)
        
        for new_chan, chans in self.magnitude_channels.items():

            if new_chan in self.channels:
                print(f"Channel '{new_chan}' already exists.")
                continue
            try:
                arrays = np.array([self[a] for a in chans]).T
                normed = np.linalg.norm(arrays, axis=1).reshape(arrays.shape[0], 1)
            except:
                normed = np.nan * np.ones((self.data.shape[0],1))

            self.data = np.append(self.data, normed, axis=1)
            self.channels = np.append(self.channels, new_chan)

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

    def extremes(self, channels=None):
        """"""
        if channels is None:
            channels = list(self.channels)
            
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

        return {self.channels[k]:self.data[:,k] for k in range(len(self.channels))}
        
    @dataproperty
    def num_timesteps(self):
        return self.data.shape[0]

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

    @dataproperty
    def psd(self):
        fs = 1. / np.diff(self.time)[0]
        freq, Pxx_den = signal.welch(self.data, fs, axis=0)
        return freq, Pxx_den

    def time_averaging(self, time_window):
        """
        Applies averaging window on time-domain loads channels.  Window input is seconds
        """
        dt = np.diff(self.time)[0]
        npts = time_window / dt
        window = np.ones(npts) / npts # Basic rectangular filter

        data_avg = np.zeros(self.data.shape)
        for k in range(len(self.channels)):
            data_avg[:,k] = np.convolve(self.data[:,k], window, 'valid')
            
        return data_avg

    def get_summary_stats(self):
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
            
            if channel in ["time", "Time"]:
                continue

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
