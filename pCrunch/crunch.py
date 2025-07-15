import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
from .utility import weibull_mean, rayleigh_mean

    
class Crunch:
    """Aeroelastic time-series simulation batch postprocessing class."""

    def __init__(self, outputs=[], **kwargs):
        """
        Creates an instance of `Crunch`.

        Parameters
        ----------
        outputs : list (optional)
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        lean : boolean (optional)
            If False (default), the outputs are kept in memory as they are processed.  If True, only summary statistics are retained.
        fatigue_channels : dict (optional)
            Dictionary with format:
            'channel': 'fatigue slope'
        extreme_stat : str (optional)
            Whether the extreme event calculation should work on [max, min, abs].
            Default, 'max'
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
        self.extreme_stat = kwargs.get("extreme_stat", "max")
        
        self.fc = kwargs.get("fatigue_channels", {})
        for k in range(self.noutputs):
            self.outputs[k].fc = self.fc
            self.outputs[k].ec = self.ec
        
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
        
        if not hasattr(output, 'stats'):
            output.trim_data(*self.td)
            output.append_magnitude_channels(self.mc)
            output.fc.update(self.fc)
            output.ec = list(set(output.ec + self.ec))
            output.set_extreme_stat(self.extreme_stat)
            output.process(**kwargs)

        return output.filename, output.stats, output.ext_table, output.dels, output.damage


    def add_output(self, output, **kwargs):
        """
        Appends output to the list and processes it

        Parameters
        ----------
        output : AerolelasticOutput
        """
        
        # Analyze the data
        fname, stats, extremes, dels, damage =  self.process_single(output, **kwargs)

        # Add to output stat containers
        self.add_output_stats(fname, stats, extremes, dels, damage)

        # Add the output to our lists
        if not self.lean_flag:
            self.outputs.append(output)

        
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

                elif stat in ["mean", "std"]:
                    res = ("NA", var, stat, summary_stats.loc[col][stat].mean())

                elif stat == "median":
                    res = ("NA", var, stat, summary_stats.loc[col][stat].median())

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
            
    def set_probability_wind_distribution(self, windspeed, v_avg, weibull_k=2.0, kind="weibull", method='pdf', idx=None, v_prob=None, probability=None):
        """
        Sets the probability of each output in the list based on a Weibull or Rayleigh or Uniform
        distribution for windspeed.

        For the 'pdf' method, the probability density function is sampled at each windspeed.
        For the 'cdf' method, the sorted vector of windspeeds is used to create probability bins, with the
        difference of the cumulative density function values at bin edges used to assign probability mass values.

        In all cases, the resulting vector of probability values is then scaled such that they sum to one.

        Parameters
        ----------
        windspeed : str or list / Numpy array
            If input as a list or array, it is simply returned (after checking for correct length).
            If input as a string, then it should be the channel name of the u-velocity component
            ('Wind1VelX` in OpenFAST)
        v_avg : float
            Average velocity for the wind distribution (sets the scale parameter in the distribution)
        weibull_k : float (optional)
            Shape parameter for the Weibull distribution. Defaults to 2.0
        kind : str (optional)
            Which distribution to use.  Should be either 'weibull' or 'rayleigh' or 'uniform' or 'user'.
            Default is 'weibull'
        method : str (optional)
            Which distribution to use.  Should be either 'pdf' or 'cdf'.
            Default is 'pdf'
        idx: list or Numpy array (optional)
            Index vector into output case list.  Default is None
        v_prob: list or Numpy array (optional)
            User defined wind speed for indexing probability
        probability: list or Numpy array (optional)
            User defined probability, indexed by v_prob
        """

        # Input consistency and sanity checks
        if isinstance(windspeed, (float, int)) or len(windspeed) == 1:
            method = 'pdf'
            kind = 'uniform'
        if kind.lower() == 'uniform':
            method = 'pdf'
        if method.lower() not in ['pdf', 'cdf']:
            print(f"Unknown method, {method.lower()}. Expected 'pdf' or 'cdf'. Defaulting to 'pdf'")
            method = 'pdf'
        if kind.lower() not in ['weibull', 'rayleigh', 'rayliegh', 'uniform', 'user']:
            print(f"Unknown probability distribution, {kind}, defaulting to uniform")
            kind = 'uniform'

        if kind == 'user':
            if not probability:
                raise Exception("The user must define a probability vs wind speed (v_prob)")
            if not v_prob:
                raise Exception("The user must define a v_prob vs wind speed (v_prob)")
            if len(probability) != len(v_prob):
                raise Exception(f"The length of probability ({len(probability)}) must match the length of v_prob ({len(v_prob)})")
            
        # Get windspeed from user or cases
        mywindspeed = self._get_windspeeds(windspeed, idx=idx)

        # Set probability from distributions
        if method.lower() == 'pdf':
            if kind.lower() == "weibull":
                prob = weibull_mean(mywindspeed, weibull_k, v_avg, kind="pdf")

            elif kind.lower() in ["rayleigh", "rayliegh"]:
                prob = rayleigh_mean(mywindspeed, v_avg, kind="pdf")

            elif kind.lower() == "uniform":
                prob = np.ones(mywindspeed.shape)


        elif method.lower() == 'cdf':
            # Get unique wind speeds
            wind_sort, unique2wind = np.unique(mywindspeed, return_inverse=True)

            # Create wind speed bins for the CDF
            bins = 0.5 * (wind_sort[:-1] + wind_sort[1:])
            bins = np.r_[0.0, bins, np.inf]

            # Get bin edge probabilities
            if kind.lower() == "weibull":
                prob_edge = weibull_mean(wind_sort, weibull_k, v_avg, kind="cdf")

            elif kind.lower() in ["rayleigh", "rayliegh"]:
                prob_edge = rayleigh_mean(wind_sort, v_avg, kind="cdf")

            # Get bin integral probabilities
            prob_unique = np.diff(prob_edge)

            # Put back in regular indexing
            prob = prob_unique[unique2wind]

        if kind == 'user':
            prob = np.interp(windspeed,v_prob,probability)

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

        self.set_probability_wind_distribution(windspeed, Vavg, weibull_k=2.0, kind="weibull", idx=idx)

        
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

    
    def compute_total_fatigue(self, lifetime=0.0, availability=1.0,
                              idx=None, idx_park=None, idx_fault=None, n_fault=0):
        """
        Computes total damage equivalent load (DEL) and Palmgren-Miner total damage based on the
        outputs in the list.  For DEL, only the optional input, 'idx' is used to select the correct outputs.
        Aside from the windspeed probability, no other lifetime scaling is applied.  For the damage calculation,
        in addition to 'lifetime' scaling, there is an allowance for operational runs with 'availability',
        parked rotor simulations for downtime with 'idx_park' scaled with (1 - availability), and expected number
        of fault events in a lifetime with 'idx_fault' and 'n_fault'.
        All optional inputs are valid for the damage calculation.

        The `process_outputs` function shuld be run before this if the
        output statisctics were not added in streaming mode.

        Parameters
        ----------
        lifetime: float (optional)
            Number of years of expected service life of the turbine
        availability: float (optional)
            Fraction of time turbine is expected to be available for power production
            versus down for maintenance.  This availability should not include expected time that
            wind speed is below cutin or above cutout.
        idx: list or Numpy array (optional)
            Index vector into output case list for operational cases (even for wind speed
            less than cutin or greater than cutout)
        idx: list or Numpy array (optional)
            Index vector into output case list for operational cases (even for wind speed
            less than cutin or greater than cutout)
        idx_park: list or Numpy array (optional)
            Index vector into output case list for cases when rotor is parked due to maintenance downtime
            (as opposed to parked to due below cutin or above cutout)
        idx_fault: list or Numpy array (optional)
            Index vector into output case list for cases when rotor is parked due to maintenance downtime
            (as opposed to parked to due below cutin or above cutout).  If this is set, must also provide 'n_fault'
        n_fault: int or float (optional)
            Number of fault events in the idx_fault list that are expected to occur in the turbine liftime.
            If this is set, must also provide 'idx_fault'

        Returns
        ----------
        total_dels : Pandas DataFrame
            Weighted and unweighted summations of damage equivalent loads for each fatigue channel
        total_damage : Pandas DataFrame
            Weighted and unweighted summations of lifetime scaled damage for each fatigue channel
        """
        dels   = self.dels.fillna(0.0)
        damage = self.damage.fillna(0.0)
        dels_total = None
        damage_total = None

        # Convert none inputs
        if idx is None:
            idx = []
        if idx_park is None:
            idx_park = []
        if idx_fault is None:
            idx_fault = []
            
        # Handle generic "run everything" input
        if len(idx) == 0 and len(idx_park) == 0 and len(idx_fault)==0:
            idx = [m for m in range(self.noutputs)]

        # Look for bad index inputs
        if np.any( np.intersect1d(idx, idx_park) ):
            raise ValueError('Cannot have common indices in idx and idx_park')
        if np.any( np.intersect1d(idx, idx_fault) ):
            raise ValueError('Cannot have common indices in idx and idx_fault')
        if np.any( np.intersect1d(idx_park, idx_fault) ):
            raise ValueError('Cannot have common indices in idx_park and idx_fault')

        
        if len(dels) > 0:
            prob = self.prob.copy()
            
            if len(idx) > 0:
               dels = dels.iloc[idx]
               prob = self.prob[idx]

            # Make sure our probabilities sum appropriately
            prob = prob / prob.sum()
                
            dels_weighted = np.sum(prob[:,np.newaxis] * dels, axis=0)
            dels_unweighted = dels.mean(axis=0)
            
            dels_total = pd.DataFrame([dels_weighted, dels_unweighted],
                                      index=['Weighted','Unweighted'])
            
        if len(damage) > 0:
            # Will need to account for any differences in elapsed time
            T_full = np.array( self.elapsed_time() )

            # Lifetime scaling factor days * hours * min * sec
            factor = lifetime * 365.0*24.0*60.0*60.0 if lifetime > 0.0 else 1.0

            # Initialize outputs
            damage_weighted_life = 0.0
            damage_unweighted_life = 0.0
            
            # First do operational cases
            if len(idx) > 0:
               damage = damage.iloc[idx]
               T = T_full[idx]
               prob = self.prob[idx]
               prob = prob / prob.sum()
                
               damage_weighted = np.sum(prob[:,np.newaxis] * damage, axis=0)
               T_weighted = np.dot(T, prob)
               damage_weighted_life += availability * damage_weighted * factor / T_weighted
               
               damage_unweighted = damage.mean(axis=0)
               T_unweighted = T.mean()
               damage_unweighted_life += availability * damage_unweighted * factor / T_unweighted

            # Now add in non-operational (parked) cases
            if len(idx_park) > 0:
                damage = self.damage.fillna(0.0).iloc[idx_park]
                T = T_full[idx_park]
                prob = self.prob[idx_park]
                prob = prob / prob.sum()

                damage_weighted = np.sum(prob[:,np.newaxis] * damage, axis=0)
                T_weighted = np.dot(T, prob)
                damage_weighted_life += (1.0 - availability) * damage_weighted * factor / T_weighted

                damage_unweighted = damage.mean(axis=0)
                T_unweighted = T.mean()
                damage_unweighted_life += (1.0 - availability) * damage_unweighted * factor / T_unweighted
                
            # Now add in fault events (fault cases)
            if len(idx_fault) > 0:
                if n_fault == 0:
                    print('Warning: fault case indices provided, but n_fault=0')
                damage = n_fault * np.sum(self.damage.fillna(0.0).iloc[idx_fault], axis=0)
                damage_weighted_life += damage
                damage_unweighted_life += damage
            
            # Results in dataframe
            damage_total = pd.DataFrame([damage_weighted_life, damage_unweighted_life],
                                        index=['Weighted','Unweighted'])

        
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
            
    def add_gradient_channels(self, chanstr, new_name):
        for k in self.outputs:
            k.add_gradient_channel(chanstr, new_name)
        
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
    
    def idxabs(self):
        return np.array( [m.idxabs for m in self.outputs] )
    
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
        
    def total_travel(self, chanstr):
        return np.array( [m.total_travel(chanstr) for m in self.outputs] )
        
    def time_averaging(self, time_window):
        for m in self.outputs:
            m.time_averaging(time_window)
    
    def time_binning(self, time_window):
        for m in self.outputs:
            m.time_binning(time_window)
    
    def psd(self, nfft=None):
        return [m.psd(nfft=nfft) for m in self.outputs]

