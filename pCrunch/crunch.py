import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
from .utility import weibull_mean, rayleigh_mean

    
class Crunch:
    """Implementation of `mlife` in python."""

    def __init__(self, outputs=[], **kwargs):
        """
        Creates an instance of `pyLife`.

        Parameters
        ----------
        outputs : list (optional)
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        lean : boolean (optional)
            If False (default), the outputs are kept in memory as they are processed.  If True, only summary statistics are retained.
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
        probability : list, tuple, or Numpy array (optional)
            Probability or weighting for each output time series.
            Should be a list or tuple or array of floats the same length as the `outputs` list
        """

        self.lean_flag = kwargs.get("lean", False)
        
        if isinstance(outputs, list):
            self.outputs = outputs
        else:
            self.outputs = [outputs]
        self.noutputs = len(self.outputs)

        self.td = kwargs.get("trim_data", ())
        self.trim_data(*self.td)

        self.mc = kwargs.get("magnitude_channels", None)
        self.append_magnitude_channels(self.mc)
        
        self.ec = kwargs.get("extreme_channels", [])
        self.fc = kwargs.get("fatigue_channels", {})
        
        self.prob = kwargs.get("probability", np.array([]))
        if self.prob.size == 0:
            self._reset_probabilities()
        
        # Initialize containers
        self.summary_stats = pd.DataFrame()
        self.extremes = {}
        self.dels = pd.DataFrame()
        self.damage = pd.DataFrame()

    def copy(self):
        mycp =  type(self)(outputs = self.outputs,
                           lean = self.lean_flag,
                           trim_data = self.td,
                           magnitude_channels = self.mc,
                           extreme_channels = self.ec,
                           fatigue_channels = self.fc,
                           probability = self.prob)
        mycp.summary_stats = self.summary_stats.copy()
        mycp.extremes = self.extremes.copy()
        mycp.dels = self.dels.copy()
        mycp.damage = self.damage.copy()
        return mycp
        
    def process_outputs(self, cores=1, **kwargs):
        """
        Processes all outputs for summary statistics and configured damage
        equivalent loads.
        """
        
        if cores > 1:
            pool = mp.Pool(processes=cores)
            returned = pool.map(
                partial(self.process_single, **kwargs), self.outputs
            )
            pool.close()
            pool.join()

        else:
            returned = [self.process_single(output, **kwargs) for output in self.outputs]

        stats  = {}
        extrs  = {}
        dels   = {}
        damage = {}

        for ifile, istats, iextrs, idels, idamage in returned:
            if ifile == '':
                ifile = len(stats)
            stats[ifile]  = istats
            extrs[ifile]  = iextrs
            dels[ifile]   = idels
            damage[ifile] = idamage
            
        self.summary_stats, self.extremes, self.dels, self.damage = self.post_process(stats, extrs, dels, damage)
        
        if self.lean_flag:
            self.outputs = []
        

    def process_single(self, output, **kwargs):
        """
        Process AeroelasticOutput output `f`.

        Parameters
        ----------
        f : str | AerolelasticOutput
            Path to output or direct output in dict format.
        """

        # Data manipulation list all other outputs
        output.trim_data(*self.td)
        output.append_magnitude_channels(self.mc)
        
        stats = output.summary_stats()

        if isinstance(self.ec, list) and len(self.ec) > 0:
            extremes = output.extremes(self.ec)
        else:
            extremes = output.extremes()

        dels, damage = self.get_DELs(output, **kwargs)

        return output.filename, stats, extremes, dels, damage


    def add_output(self, output, **kwargs):
        """
        Appends output to the list and processes it

        Parameters
        ----------
        output : AerolelasticOutput
        """
        
        # Data manipulation list all other outputs
        output.trim_data(*self.td)
        output.append_magnitude_channels(self.mc)
        
        # Add the output to our lists and increment counters
        if not self.lean_flag:
            self.outputs.append(output)

        # Analyze the data
        fname, stats, extremes, dels, damage =  self.process_single(output, **kwargs)

        # Add to output stat containers
        self.add_output_stats(fname, stats, extremes, dels, damage)

        
    def add_output_stats(self, fname, stats, extremes, dels, damage):

        # Append to the stats logs
        statsdf = self._process_summary_stats( {fname:stats} )
        ndf = len(statsdf.index)+len(self.summary_stats.index)
        nidx = len(set(list(statsdf.index)+list(self.summary_stats.index)))
        ignore_flag = nidx < ndf
        self.summary_stats = pd.concat((self.summary_stats, statsdf), axis=0,
                                       ignore_index=ignore_flag)

        # Extreme events- join dictionaries
        self.extremes = self._process_extremes( {'null':self.extremes, fname:extremes} )
        
        # Damage and Damage Equivalent Loads
        self.dels   = pd.concat((self.dels,   pd.DataFrame(dels, index=[fname])), axis=0)
        self.damage = pd.concat((self.damage, pd.DataFrame(damage, index=[fname])), axis=0)

        # Increment counters
        self.noutputs += 1
        self._reset_probabilities()

        
    def get_DELs(self, output, **kwargs):
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
                DELs[chan], D[chan] = output.compute_del(chan, fatparams,
                                                         goodman_correction=goodman,
                                                         rainflow_bins=bins,
                                                         return_damage=return_damage)

            except IndexError:
                print(f"Channel '{chan}' not included in DEL calculation.")
                DELs[chan] = np.nan
                D[chan] = np.nan

        return DELs, D


    def _process_summary_stats(self, statsin):
        # Summary statistics
        ss = pd.DataFrame.from_dict(statsin, orient="index").stack().to_frame()
        ss = pd.DataFrame(ss[0].values.tolist(), index=ss.index)
        return ss.unstack().swaplevel(axis=1)

    def _process_extremes(self, extremesin):
        # Extremes
        extreme_table = {}
        for _, d in extremesin.items():
            for channel, sub in d.items():
                subt = sub if isinstance(sub, list) else [sub]
                if channel not in extreme_table:
                    extreme_table[channel] = subt
                else:
                    extreme_table[channel] += subt
        return extreme_table
        
    def post_process(self, statsin, extremesin, delsin, damagein):
        """Post processes internal data to produce DataFrame outputs."""
        summary_stats = self._process_summary_stats(statsin)

        # Extreme events
        extremes = self._process_extremes(extremesin)
        
        # Damage and Damage Equivalent Loads
        dels = pd.DataFrame(delsin).T
        damage = pd.DataFrame(damagein).T

        return summary_stats, extremes, dels, damage

    
    def get_load_rankings(self, ranking_vars, ranking_stats):
        """
        Returns load rankings across all outputs in `self.outputs`.

        Parameters
        ----------
        rankings_vars : list
            List of variables to evaluate for the ranking process.
        ranking_stats : list
            Summary statistic to evalulate. Currently supports 'min', 'max',
            'abs', 'mean', 'std'.
        """
        if not isinstance(ranking_vars, (list, tuple, set)):
            ranking_vars = [ranking_vars]
            #raise ValueError('Need a list or tuple of ranking variables')
        if not isinstance(ranking_stats, (list, tuple, set)):
            ranking_stats = [ranking_stats]
            #raise ValueError('Need a list or tuple of ranking statistics')

        summary_stats = self.summary_stats.copy().swaplevel(axis=1).stack(future_stack=True)

        out = []
        for var in ranking_vars:
            for stat in ranking_stats:

                col = pd.MultiIndex.from_product([self.summary_stats.index, [var]])
                if stat in ["max", "abs"]:
                    res = (*summary_stats.loc[col][stat].idxmax(), stat,
                           summary_stats.loc[col][stat].max())

                elif stat == "min":
                    res = (*summary_stats.loc[col][stat].idxmin(), stat,
                           summary_stats.loc[col][stat].min())

                elif stat == "mean":
                    res = ("NA", var, stat, summary_stats.loc[col][stat].mean())

                elif stat == "median":
                    res = ("NA", var, stat, summary_stats.loc[col][stat].median())

                elif stat == "std":
                    res = ("NA", var, stat, summary_stats.loc[col][stat].std())

                else:
                    raise NotImplementedError(f"Statistic '{stat}' not supported for load ranking.")

                out.append(res)

        return pd.DataFrame(out, columns=["file", "channel", "stat", "val"])


    def _get_windspeeds(self, windspeed, idx=None):
        """
        Returns the wind speeds in the output list if not already known.

        Parameters
        ----------
        windspeed : str or list / Numpy array
            If input as a list or array, it is simply returned (after checking for correct length).
            If input as a string, then it should be the channel name of the u-velocity component
            ('Wind1VelX` in OpenFAST)
        idx: list or Numpy array (default None)
            Index vector into output case list
        
        Returns
        ----------
        windspeed : Numpy array
            Numpy array of the average inflow wind speed for each output.
        """

        if isinstance(windspeed, str):
            # If user passed wind u-component channel name (e.g. 'Wind1VelX'),
            # then get mean value for every case
            if len(self.summary_stats) > 0:
                mywindspeed = self.summary_stats[windspeed]['mean'].to_numpy()
            else:
                mywindspeed = np.array( [np.mean( k[windspeed] ) for k in self.outputs] )
            if idx is not None and len(idx) > 0:
                mywindspeed = mywindspeed[idx]
        else:
            # Assume user passed windspeeds directly
            mywindspeed = np.array(windspeed)
            if idx is not None and len(idx) > 0:
                ntarget = len(idx)
                if mywindspeed.size == self.noutputs:
                    mywindspeed = mywindspeed[idx]
            else:
                ntarget = self.noutputs
            if mywindspeed.size != ntarget:
                raise ValueError(f"When giving list of windspeeds, must have {ntarget} values, one for every case")
            
        return mywindspeed

    
    def _reset_probabilities(self):
        if self.noutputs > 0:
            self.prob = np.ones( self.noutputs ) / float(self.noutputs)
            
    def set_probability_distribution(self, windspeed, v_avg, weibull_k=2.0, kind="weibull", idx=None):
        """
        Sets the probability of each output in the list based on a Weibull or Rayleigh
        distribution for windspeed.

        Probability density function is sampled at each windspeed.  The resulting vector of
        density values is then scaled such that they sum to one.

        Parameters
        ----------
        windspeed : str or list / Numpy array
            If input as a list or array, it is simply returned (after checking for correct length).
            If input as a string, then it should be the channel name of the u-velocity component
            ('Wind1VelX` in OpenFAST)
        v_avg : float
            Average velocity for the wind distribution (sets the scale parameter in the distribution)
        weibull_k : float
            Shape parameter for the Weibull distribution. Defaults to 2.0
        kind : str
            Which distribution to use.  Should be either 'weibull' or 'rayleigh'
        idx: list or Numpy array (default None)
            Index vector into output case list
        """
        
        # Get windspeed from user or cases
        mywindspeed = self._get_windspeeds(windspeed, idx=idx)

        # Set probability from distributions
        if kind.lower() == "weibull":
            prob = weibull_mean(mywindspeed, weibull_k, v_avg, kind="pdf")
            
        elif kind.lower() in ["rayleigh", "rayliegh"]:
            prob = rayleigh_mean(mywindspeed, v_avg, kind="pdf")

        elif kind.lower() in ["uniform"]:
            prob = np.ones(mywindspeed.shape)

        else:
            print(f"Unknown probability distribution, {kind}, defaulting to uniform")
            prob = np.ones(mywindspeed.shape)

        # Ensure probability sums to one for all of our cases
        self._reset_probabilities()
        prob = prob / prob.sum()
        if idx is not None and len(idx) > 0:
            self.prob[idx] = prob
        else:
            self.prob = prob

        
    def set_probability_turbine_class(self, windspeed, turbine_class, idx=None):
        """
        Sets the probability of each output in the list based on a Weibull distribution
        (shape parameter 2) and average velocity as determined by IEC turbine class standards.

        Parameters
        ----------
        windspeed : str or list / Numpy array
            If input as a list or array, it is simply returned (after checking for correct length).
            If input as a string, then it should be the channel name of the u-velocity component
            ('Wind1VelX` in OpenFAST)
        turbine_class : int or str
            Turbine class either 1/'I' or 2/'II' or 3/'III'
        idx: list or Numpy array (default None)
            Index vector into output case list
        """

        # IEC average velocities
        if turbine_class in [1, "I"]:
            Vavg = 50 * 0.2

        elif turbine_class in [2, "II"]:
            Vavg = 42.5 * 0.2

        elif turbine_class in [3, "III"]:
            Vavg = 37.5 * 0.2

        else:
            raise ValueError(f"Unknown turbine class, {turbine_class}")

        self.set_probability_distribution(windspeed, Vavg, weibull_k=2.0, kind="weibull", idx=idx)

        
    def compute_aep(self, pwrchan, loss_factor=0.0, idx=None):
        """
        Computes annual energy production based on all outputs in the list.

        Parameters
        ----------
        pwrchan : string
            Name of channel containing output electrical power in the simulation
            (e.g. 'GenPwr' in OpenFAST)
        loss_factor : float
            Loss factor for availability and other losses
            (soiling, array, etc.) to apply to AEP calculation (0.0 = no losses)"
        idx: list or Numpy array (default None)
            Index vector into output case list

        Returns
        ----------
        aep_weighted : float
            Weighted AEP where each output is weighted by the probability of
            occurence determined from its average wind speed
        aep_unweighted : float
            Unweighted AEP that assumes all outputs in the list are equally
            likely (just a mean)
        """

        # Energy for every output case
        E = np.array( self.compute_energy(pwrchan) )
        T = np.array( self.elapsed_time() )
        prob = self.prob.copy()

        if idx is not None and len(idx) > 0:
            E = E[idx]
            T = T[idx]
            prob = prob[idx]

        # Make sure our probabilities sum appropriately
        prob = prob / prob.sum()

        # Calculate scaling factor for year with losses included
        fact = (1.0 - loss_factor) * 365.0 * 24.0 * 60.0 * 60.0

        # Sum with probability.  Finding probability weighted average per second, scaled by total seconds
        aep_weighted = fact * np.dot(E, prob) / np.dot(T, prob)

        # Assume equal probability
        prob = np.ones(E.shape)/E.size
        aep_unweighted = fact * np.dot(E, prob) / np.dot(T, prob)

        return aep_weighted, aep_unweighted

    
    def compute_total_fatigue(self, idx=None):
        """
        Computes total damage equivalent load and total damage based on all
        outputs in the list.

        The `process_outputs` function shuld be run before this if the
        output statisctics were not added in streaming mode.

        Parameters
        ----------
        idx: list or Numpy array (default None)
            Index vector into output case list

        Returns
        ----------
        total_dels : Pandas DataFrame
            Weighted and unweighted summations of damage equivalent loads for each fatigue channel
        total_damage : Pandas DataFrame
            Weighted and unweighted summations of damage for each fatigue channel
        """
        prob   = self.prob.copy()
        dels   = self.dels.fillna(0.0)
        damage = self.damage.fillna(0.0)
        
        if idx is not None and len(idx) > 0:
            prob = prob[idx]
        
        if len(dels) > 0:
            if idx is not None and len(idx) > 0:
               dels = dels.iloc[idx]

            dels_weighted = np.sum(prob[:,np.newaxis] * dels, axis=0)
            dels_unweighted = dels.mean(axis=0)
            
            dels_total = pd.DataFrame([dels_weighted, dels_unweighted],
                                      index=['Weighted','Unweighted'])
        else:
            dels_total = None
            
        if len(damage) > 0:
            if idx is not None and len(idx) > 0:
               damage = damage.iloc[idx]
               
            damage_weighted = np.sum(prob[:,np.newaxis] * damage, axis=0)
            damage_unweighted = damage.mean(axis=0)
            
            damage_total = pd.DataFrame([damage_weighted, damage_unweighted],
                                        index=['Weighted','Unweighted'])
        else:
            damage_total = None
            
        return dels_total, damage_total

        
    # Batch versions of AeroelasticOutput methods
    def calculate_channel(self, instr, namein):
        for k in self.outputs:
            k.calculate_channel(instr, namein)
            
    def trim_data(self, tmin=0, tmax=np.inf):
        for k in self.outputs:
            k.trim_data(tmin, tmax)
            
    def append_magnitude_channels(self, magnitude_channels=None):
        for k in self.outputs:
            k.append_magnitude_channels(magnitude_channels)
            
    def add_magnitude_channels(self, magnitude_channels=None):
        self.append_magnitude_channels(magnitude_channels)
        
    def add_load_rose(self, load_rose=None, nsec=6):
        for k in self.outputs:
            k.add_load_rose(load_rose=load_rose, nsec=nsec)

    def drop_channel(self, pattern):
        for k in self.outputs:
            k.drop_channel(pattern)
    def delete_channel(self, pattern):
        self.drop_channel(pattern)
    def remove_channel(self, pattern):
        self.drop_channel(pattern)
        
    def num_timesteps(self):
        return [m.num_timesteps for m in self.outputs]

    def num_channels(self):
        if len(self.summary_stats) > 0:
            return [len([a for a,b in self.summary_stats.columns if b=='min'])]*self.noutputs
        else:
            return [m.num_channels for m in self.outputs]
    
    def elapsed_time(self):
        if len(self.summary_stats) > 0:
            return (self.summary_stats['Time']['max'].to_numpy() -
                    self.summary_stats['Time']['min'].to_numpy())
        else:
            return np.array( [m.elapsed_time for m in self.outputs] )
        
    def idxmins(self):
        return np.array( [m.idxmins for m in self.outputs] )
    
    def idxmaxs(self):
        return np.array( [m.idxmaxs for m in self.outputs] )
    
    def minima(self):
        if len(self.summary_stats) > 0:
            return np.array([self.summary_stats[a,b].to_list() for a,b in self.summary_stats.columns if b=='min']).T
        else:
            return np.array( [m.minima for m in self.outputs] )
        
    def maxima(self):
        if len(self.summary_stats) > 0:
            return np.array([self.summary_stats[a,b].to_list() for a,b in self.summary_stats.columns if b=='max']).T
        else:
            return np.array( [m.maxima for m in self.outputs] )
        
    def ranges(self):
        return (np.array(self.maxima()) - np.array(self.minima()))
    
    def variable(self):
        return np.array( [m.variable for m in self.outputs] )
    
    def constant(self):
        return np.array( [m.constant for m in self.outputs] )
    
    def sums(self):
        return np.array( [m.sums for m in self.outputs] )
    
    def sums_squared(self):
        return np.array( [m.sums_squared for m in self.outputs] )
    
    def sums_cubed(self):
        return np.array( [m.sums_cubed for m in self.outputs] )
    
    def sums_fourth(self):
        return np.array( [m.sums_fourth for m in self.outputs] )
    
    def second_moments(self):
        return np.array( [m.second_moments for m in self.outputs] )
    
    def third_moments(self):
        return np.array( [m.third_moments for m in self.outputs] )
    
    def fourth_moments(self):
        return np.array( [m.fourth_moments for m in self.outputs] )
    
    def means(self):
        if len(self.summary_stats) > 0:
            return np.array([self.summary_stats[a,b].to_list() for a,b in self.summary_stats.columns if b=='mean']).T
        else:
            return np.array( [m.means for m in self.outputs] )
        
    def medians(self):
        if len(self.summary_stats) > 0:
            return np.array([self.summary_stats[a,b].to_list() for a,b in self.summary_stats.columns if b=='median']).T
        else:
            return np.array( [m.medians for m in self.outputs] )
        
    def absmaxima(self):
        return np.array( [m.absmaxima for m in self.outputs] )
    
    def stddevs(self):
        if len(self.summary_stats) > 0:
            return np.array([self.summary_stats[a,b].to_list() for a,b in self.summary_stats.columns if b=='std']).T
        else:
            return np.array( [m.stddevs for m in self.outputs] )
        
    def skews(self):
        return np.array( [m.skews for m in self.outputs] )
    
    def kurtosis(self):
        return np.array( [m.kurtosis for m in self.outputs] )
    
    def integrated(self):
        if len(self.summary_stats) > 0:
            return np.array([self.summary_stats[a,b].to_list() for a,b in self.summary_stats.columns if b=='integrated']).T
        else:
            return np.array( [m.integrated for m in self.outputs] )
        
    def compute_energy(self, pwrchan):
        if len(self.summary_stats) > 0:
            return self.summary_stats[pwrchan]['integrated']
        else:
            return np.array( [m.compute_energy(pwrchan) for m in self.outputs] )
        
    def time_averaging(self, time_window):
        for m in self.outputs:
            m.time_averaging(time_window)
    
    def time_binning(self, time_window):
        for m in self.outputs:
            m.time_binning(time_window)
    
    def psd(self):
        return [m.psd() for m in self.outputs]

