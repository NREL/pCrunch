import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import fatpack
from .utility import weibull_mean, rayleigh_mean

def compute_del(ts, elapsed, lifetime, load2stress, slope, Sult, Sc=0.0, **kwargs):
    """
    Computes damage equivalent load of input `ts`.

    Parameters
    ----------
    ts : np.array
        Time series to calculate DEL for.
    elapsed : int | float
        Elapsed time of the time series.
    lifetime : int | float
        Design lifetime of the component / material in years
    load2stress : float (optional)
        Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L
    slope : int | float
        Slope of the fatigue curve.
    Sult : float (optional)
        Ultimate stress for use in Goodman equivalent stress calculation
    Sc : float (optional)
        Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified
    rainflow_bins : int
        Number of bins used in rainflow analysis.
        Default: 100
    goodman_correction: boolean
        Whether to apply Goodman mean correction to loads and stress
        Default: False
    return_damage: boolean
        Whether to compute both DEL and damage
        Default: False
    """

    bins = kwargs.get("rainflow_bins", 100)
    return_damage = kwargs.get("return_damage", False)
    goodman = kwargs.get("goodman_correction", False)
    Scin = Sc if Sc > 0.0 else Sult

    # Default return values
    DEL = np.nan
    D   = np.nan

    if np.all(np.isnan(ts)):
        return DEL, D

    # Working with loads for DELs
    try:
        F, Fmean = fatpack.find_rainflow_ranges(ts, return_means=True)
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
            S, Mrf = fatpack.find_rainflow_ranges(ts*load2stress, return_means=True)
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

    
# Could use a dict or namedtuple here, but this standardizes things a bit better for users
class FatigueParams:
    """Simple data structure of parameters needed by fatigue calculation."""

    def __init__(self, **kwargs):
        """
        Creates an instance of `FatigueParams`.

        Parameters
        ----------
        lifetime :  float (optional)
            Design lifetime of the component / material in years
        load2stress : float (optional)
            Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L
        slope : float (optional)
            Wohler exponent in the traditional SN-curve of S = A * N ^ -(1/m)
        ult_stress : float (optional)
            Ultimate stress for use in Goodman equivalent stress calculation
        S_intercept : float (optional)
            Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified
        rainflow_bins : int
            Number of bins used in rainflow analysis.
            Default: 100
        goodman_correction: boolean
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        return_damage: boolean
            Whether to compute both DEL and damage
            Default: False
        """

        self.lifetime      = kwargs.get("lifetime", 0.0)
        self.load2stress   = kwargs.get("load2stress", 1.0)
        self.slope         = kwargs.get("slope", 4.0)
        self.ult_stress    = kwargs.get("ult_stress", 1.0)
        temp               = kwargs.get("S_intercept", 0.0)
        self.S_intercept   = temp if temp > 0.0 else self.ult_stress
        self.bins          = kwargs.get("rainflow_bins", 100)
        self.return_damage = kwargs.get("return_damage", False)
        self.goodman       = kwargs.get("goodman_correction", False)

    def copy(self):
        return FatigueParams(lifetime=self.lifetime,
                             load2stress=self.load2stress, slope=self.slope,
                             ult_stress=self.ult_stress, S_intercept=self.S_intercept,
                             bins=self.bins, return_damage=self.return_damage,
                             goodman=self.goodman)

    
class Crunch:
    """Implementation of `mlife` in python."""

    def __init__(self, outputs, **kwargs):
        """
        Creates an instance of `pyLife`.

        Parameters
        ----------
        outputs : list
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        directory : str (optional)
            If directory is passed, list of files will be treated as relative
            and appended to the directory.
        fatigue_channels : dict (optional)
            Dictionary with format:
            'channel': 'fatigue slope'
        magnitude_channels : dict (optional)
            Additional channels as vector magnitude of other channels.
            Format: 'new-chan': ['chan1', 'chan2', 'chan3']
        trim_data : tuple (optional)
            Trim processed outputs to desired times.
            Format: (min, max)
        probability : list, tuple, or Numpy array (optional)
            Probability or weighting for each output time series.
            Should be a list or tuple or array of floats the same length as the `outputs` list
        """

        self.outputs = outputs
        self.parse_settings(**kwargs)
        

    def parse_settings(self, **kwargs):
        """Parses settings from input kwargs."""

        self.directory = kwargs.get("directory", None)
        self.ec = kwargs.get("extreme_channels", True)
        self.mc = kwargs.get("magnitude_channels", {})
        self.fc = kwargs.get("fatigue_channels", {})
        self.td = kwargs.get("trim_data", ())
        self.prob = kwargs.get("probability", [])

        self.prep_outputs()

    def prep_outputs(self):
        """Trim the data by time, set magnitude channels and probability."""

        for k in self.outputs:
            if len(self.td) > 0:
                k.trim_data(*self.td)

            if len(self.mc) > 0:
                k.append_magnitude_channels( self.mc )

            if len(self.prob) == 0:
                noutput = len(self.outputs)
                self.prob = np.ones( noutput ) / float(noutput)
            
        
        
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
            stats[ifile]  = istats
            extrs[ifile]  = iextrs
            dels[ifile]   = idels
            damage[ifile] = idamage
            
        self.summary_stats, self.extremes, self.dels, self.damage = self.post_process(stats, extrs, dels, damage)
        

    def process_single(self, output, **kwargs):
        """
        Process OpenFAST output `f`.

        Parameters
        ----------
        f : str | OpenFASTOutput
            Path to output or direct output in dict format.
        """
        stats = output.get_summary_stats()

        if self.ec is True:
            extremes = output.extremes()

        elif isinstance(self.ec, list):
            extremes = output.extremes(self.ec)

        dels, damage = self.get_DELs(output, **kwargs)

        return output.filename, stats, extremes, dels, damage

    
    def get_DELs(self, output, **kwargs):
        """
        Appends computed damage equivalent loads for fatigue channels in
        `self.fc`.

        Parameters
        ----------
        output : OpenFASTOutput
        rainflow_bins : int
            Number of bins used in rainflow analysis.
            Default: 100
        goodman_correction: boolean
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        return_damage: boolean
            Whether to compute both DEL and damage
            Default: False
        """

        DELs = {}
        D = {}

        for chan, fatparams in self.fc.items():
            if "goodman_correction" in kwargs:
                goodman = kwargs.get("goodman_correction", False)
            else:
                goodman = fatparams.goodman

            if "rainflow_bins" in kwargs:
                bins = kwargs.get("rainflow_bins", 100)
            else:
                bins = fatparams.bins

            if "return_damage" in kwargs:
                return_damage = kwargs.get("return_damage", False)
            else:
                return_damage = fatparams.return_damage
                
            try:

                DELs[chan], D[chan] = compute_del(
                    output[chan], output.elapsed_time,
                    fatparams.lifetime,
                    fatparams.load2stress, fatparams.slope,
                    fatparams.ult_stress, fatparams.S_intercept,
                    goodman_correction=goodman, rainflow_bins=bins,
                    return_damage=return_damage,
                )

            except IndexError:
                print(f"Channel '{chan}' not included in DEL calculation.")
                DELs[chan] = np.NaN
                D[chan] = np.NaN

        return DELs, D


    @staticmethod
    def post_process(statsin, extremesin, delsin, damagein):
        """Post processes internal data to produce DataFrame outputs."""

        # Summary statistics
        ss = pd.DataFrame.from_dict(statsin, orient="index").stack().to_frame()
        ss = pd.DataFrame(ss[0].values.tolist(), index=ss.index)
        summary_stats = ss.unstack().swaplevel(axis=1)

        # Extreme events
        extreme_table = {}
        for _, d in extremesin.items():
            for channel, sub in d.items():
                if channel not in extreme_table.keys():
                    extreme_table[channel] = []

                extreme_table[channel].append(sub)
        extremes = extreme_table

        # Damage and Damage Equivalent Loads
        dels = pd.DataFrame(delsin).T
        damage = pd.DataFrame(damagein).T

        return summary_stats, extremes, dels, damage

    
    def get_load_rankings(self, ranking_vars, ranking_stats, **kwargs):
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

        summary_stats = self.summary_stats.copy().swaplevel(axis=1).stack()

        out = []
        for var, stat in zip(ranking_vars, ranking_stats):

            if not isinstance(var, list):
                var = [var]

            col = pd.MultiIndex.from_product([self.outputs, var])
            if stat in ["max", "abs"]:
                res = (*summary_stats.loc[col][stat].idxmax(),
                       stat,
                       summary_stats.loc[col][stat].max(),
                       )

            elif stat == "min":
                res = (*summary_stats.loc[col][stat].idxmin(),
                       stat,
                       summary_stats.loc[col][stat].min(),
                       )

            elif stat in ["mean", "std"]:
                res = (np.NaN, ", ".join(var), stat,
                       summary_stats.loc[col][stat].mean(),
                       )

            else:
                raise NotImplementedError(f"Statistic '{stat}' not supported for load ranking.")

            out.append(res)

        return pd.DataFrame(out, columns=["file", "channel", "stat", "val"])


    def _get_windspeeds(self, windspeed):
        """
        Returns the wind speeds in the output list if not already known.

        Parameters
        ----------
        windspeed : str or list / Numpy array
            If input as a list or array, it is simply returned (after checking for correct length).
            If input as a string, then it should be the channel name of the u-velocity component
            ('Wind1VelX` in OpenFAST)

        Returns
        ----------
        windspeed : Numpy array
            Numpy array of the average inflow wind speed for each output.
        """

        if isinstance(windspeed, str):
            # If user passed wind u-component channel name (e.g. 'Wind1VelX'),
            # then get mean value for every case
            mywindspeed = np.array( [np.mean( k[windspeed] ) for k in self.outputs] )
        else:
            # Assume user passed windspeeds directly
            noutput = len(self.outputs)
            mywindspeed = np.array(windspeed)
            if mywindspeed.size != noutput:
                raise ValueError(f"When giving list of windspeeds, must have {noutput} values, one for every case")
            
        return mywindspeed

    
    def set_probability_distribution(self, windspeed, v_avg, weibull_k=2.0, kind="weibull"):
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
        """
        
        # Get windspeed from user or cases
        mywindspeed = self._get_windspeeds(windspeed)

        # Set probability from distributions
        if kind.lower() == "weibull":
            prob = weibull_mean(mywindspeed, weibull_k, v_avg, kind="pdf")
            
        elif kind.lower() in ["rayleigh", "rayliegh"]:
            prob = rayleigh_mean(mywindspeed, v_avg, kind="pdf")

        # Ensure probability sums to one for all of our cases
        self.prob = prob / prob.sum()

        
    def set_probability_turbine_class(self, windspeed, turbine_class):
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
        """

        # IEC average velocities
        if turbine_class in [1, "I"]:
            Vavg = 50 * 0.2

        elif turbine_class in [2, "II"]:
            Vavg = 42.5 * 0.2

        elif turbine_class in [3, "III"]:
            Vavg = 37.5 * 0.2

        self.set_probability_distribution(windspeed, Vavg, weibull_k=2.0, kind="weibull")

        
    def compute_aep(self, pwrchan, loss_factor=1.0):
        """
        Computes annual energy production based on all outputs in the list.

        Parameters
        ----------
        pwrchan : string
            Name of channel containing output electrical power in the simulation
            (e.g. 'GenPwr' in OpenFAST)
        loss_factor : float
            Multiplicative loss factor for availability and other losses
            (soiling, array, etc.) to apply to AEP calculation (1.0 = no losses)"

        Returns
        ----------
        aep_weighted : float
            Weighted AEP where each output is weighted by the probability of
            occurence determined from its average wind speed
        aep_unweighted : float
            Unweighted AEP that assumes all outputs in the list are equally
            likely (just a mean)
        """
        
        # Calculate scaling factor for year with losses included
        fact = loss_factor * 365.0 * 24.0

        # Power for every output case
        P = np.array( [k.compute_power(pwrchan) for k in self.outputs] )

        # Sum with probability
        aep_weighted = fact * np.dot(P, self.prob)

        # Assume equal probability
        aep_unweighted = fact * P.mean()
        
        return aep_weighted, aep_unweighted

    
    def compute_total_fatigue(self):
        """
        Computes total damage equivalent load and total damage based on all
        outputs in the list.

        The `process_outputs` function shuld be run before this.

        Returns
        ----------
        total_dels : Pandas DataFrame
            Weighted and unweighted summations of damage equivalent loads for each fatigue channel
        total_damage : Pandas DataFrame
            Weighted and unweighted summations of damage for each fatigue channel
        """
        
        if self.dels:
            # These should come out as pandas Series
            dels_weighted = np.sum(self.prob[:,np.newaxis] * self.dels, axis=0)
            dels_unweighted = self.dels.mean()
            damage_weighted = np.sum(self.prob[:,np.newaxis] * self.damage, axis=0)
            damage_unweighted = self.damage.mean()
            
            # Combine in to DataFrames
            dels_total = pd.DataFrame([dels_weighted, dels_unweighted],
                                      index=['Weighted','Unweighted'])
            damage_total = pd.DataFrame([damage_weighted, damage_unweighted],
                                        index=['Weighted','Unweighted'])
            return dels_total, damage_total
        
        else:
            print("No DELs found.  Please run process_outputs first.")
            return None, None
