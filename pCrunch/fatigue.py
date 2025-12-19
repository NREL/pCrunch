import numpy as np
import fatpack

# Curtosy of Fatpack examples!
# The different types refer to different types of welds and hot spots.  See Appendix A
curves_in_air = dict(
    reference = "DNV-RP-C203 - Edition October 2024, Table 2-1 S-N curves in air",
    B1= dict(m1=4.0, loga1=15.117, m2=5, Nd=5e6, loga2=17.146, fl=127.21),
    B2= dict(m1=4.0, loga1=14.885, m2=5, Nd=5e6, loga2=16.856, fl=111.3),
    C = dict(m1=3.5, loga1=13.640, m2=5, Nd=5e6, loga2=16.615, fl= 96.21),
    C1= dict(m1=3.5, loga1=13.473, m2=5, Nd=5e6, loga2=16.377, fl= 86.20),
    C2= dict(m1=3.5, loga1=13.301, m2=5, Nd=5e6, loga2=16.130, fl= 76.97),
    D = dict(m1=3.0, loga1=12.164, m2=5, Nd=1e7, loga2=15.606, fl= 52.63),
    E = dict(m1=3.0, loga1=12.010, m2=5, Nd=1e7, loga2=15.350, fl= 46.78),
    F = dict(m1=3.0, loga1=11.855, m2=5, Nd=1e7, loga2=15.091, fl= 41.52),
    F1= dict(m1=3.0, loga1=11.699, m2=5, Nd=1e7, loga2=14.832, fl= 36.84),
    F3= dict(m1=3.0, loga1=11.546, m2=5, Nd=1e7, loga2=14.576, fl= 32.75),
    G = dict(m1=3.0, loga1=11.398, m2=5, Nd=1e7, loga2=14.330, fl= 29.24),
    W1= dict(m1=3.0, loga1=11.261, m2=5, Nd=1e7, loga2=14.101, fl= 26.32),
    W2= dict(m1=3.0, loga1=11.107, m2=5, Nd=1e7, loga2=13.845, fl= 23.39),
    W3= dict(m1=3.0, loga1=10.970, m2=5, Nd=1e7, loga2=13.617, fl= 21.05),
)

curves_in_seawater_with_cathodic_protection = dict(
    reference = "DNV-RP-C203 - Edition October 2024, Table 2-2 S-N in seawater with cathodic protection",
    B1= dict(m1=4.0, loga1=14.977, m2=5, Nd=1e6, loga2=17.222, fl=127.21),
    B2= dict(m1=4.0, loga1=14.745, m2=5, Nd=1e6, loga2=16.931, fl=111.3),
    C = dict(m1=3.5, loga1=13.431, m2=5, Nd=1e6, loga2=16.615, fl= 96.21),
    C1= dict(m1=3.5, loga1=13.264, m2=5, Nd=1e6, loga2=16.377, fl= 86.20),
    C2= dict(m1=3.5, loga1=13.091, m2=5, Nd=1e6, loga2=16.130, fl= 76.97),
    D = dict(m1=3.0, loga1=11.764, m2=5, Nd=1e6, loga2=15.606, fl= 52.63),
    E = dict(m1=3.0, loga1=11.610, m2=5, Nd=1e6, loga2=15.350, fl= 46.78),
    F = dict(m1=3.0, loga1=11.455, m2=5, Nd=1e6, loga2=15.091, fl= 41.52),
    F1= dict(m1=3.0, loga1=11.299, m2=5, Nd=1e6, loga2=14.832, fl= 36.84),
    F3= dict(m1=3.0, loga1=11.146, m2=5, Nd=1e6, loga2=14.576, fl= 32.75),
    G = dict(m1=3.0, loga1=10.998, m2=5, Nd=1e6, loga2=14.330, fl= 29.24),
    W1= dict(m1=3.0, loga1=10.861, m2=5, Nd=1e6, loga2=14.101, fl= 26.32),
    W2= dict(m1=3.0, loga1=10.707, m2=5, Nd=1e6, loga2=13.845, fl= 23.39),
    W3= dict(m1=3.0, loga1=10.570, m2=5, Nd=1e6, loga2=13.617, fl= 21.05),
)

curves_in_seawater_for_free_corrosion = dict(
    reference = "DNV-RP-C203 - Edition October 2024, Table 2-4 S-N in seawater with free corrosion",
    B1= dict(m=3.0, loga=12.436),
    B2= dict(m=3.0, loga=12.262),
    C = dict(m=3.0, loga=12.115),
    C1= dict(m=3.0, loga=11.972),
    C2= dict(m=3.0, loga=11.824),
    D = dict(m=3.0, loga=11.687),
    E = dict(m=3.0, loga=11.533),
    F = dict(m=3.0, loga=11.378),
    F1= dict(m=3.0, loga=11.222),
    F3= dict(m=3.0, loga=11.068),
    G = dict(m=3.0, loga=10.921),
    W1= dict(m=3.0, loga=10.784),
    W2= dict(m=3.0, loga=10.630),
    W3= dict(m=3.0, loga=10.493),
)

def dnv_in_air(name):
    """Returns a DNV endurance curve (SN curve)

    This method returns an endurance curve in air according to 
    table 2-1 in DNV RP-C203.

    Arguments
    ---------
    name : str
        Name of the endurance curve.

    Returns
    -------
    fatpack.BiLinearEnduranceCurve
        Endurance curve corresponding to `name` in DNV RP-C203

    Example
    -------
    >>>curve = DNV_EnduranceCurve.in_air("D")
    >>>N = curve.get_endurance(90.0)

    """

    data = curves_in_air[name]
    curve = fatpack.BiLinearEnduranceCurve(1e6) # 1e6 for MPa to Pa
    curve.Nc = 10 ** data["loga1"] 
    curve.Nd = data["Nd"]
    curve.m1 = data["m1"]
    curve.m2 = data["m2"]
    curve.reference = curves_in_air["reference"]
    return curve
    
    
def dnv_in_seawater_cathodic(name):
    """Returns a DNV endurance curve (SN curve)

    This method returns an endurance curve in seawater with 
    cathodic protection according to table 2-2 in DNV RP-C203.

    Arguments
    ---------
    name : str
        Name of the endurance curve.

    Returns
    -------
    fatpack.BiLinearEnduranceCurve
        Endurance curve corresponding to `name` in DNV RP-C203

    Example
    -------
    >>>curve = DNV_EnduranceCurve.in_seawater_with_cathodic_protection("D")
    >>>N = curve.get_endurance(90.0)

    """
    data = curves_in_seawater_with_cathodic_protection[name]
    curve = fatpack.BiLinearEnduranceCurve(1e6) # 1e6 for MPa to Pa
    curve.Nc = 10 ** data["loga1"]
    curve.Nd = data["Nd"]
    curve.m1 = data["m1"]
    curve.m2 = data["m2"]
    ref = curves_in_seawater_with_cathodic_protection["reference"]
    curve.reference = ref
    return curve

def dnv_in_seawater(name):
    """Returns a DNV endurance curve (SN curve)

    This method returns an endurance curve in seawater for 
    free corrosion according to table 2-4 in DNV RP-C203.

    Arguments
    ---------
    name : str
        Name of the endurance curve.

    Returns
    -------
    fatpack.LinearEnduranceCurve
        Endurance curve corresponding to `name` in DNV RP-C203

    Example
    -------
    >>>curve = DNV_EnduranceCurve.in_seawater_for_free_corrosion("D")
    >>>N = curve.get_endurance(90.0)

    """
    data = curves_in_seawater_for_free_corrosion[name]
    curve = fatpack.LinearEnduranceCurve(1e6) # 1e6 for MPa to Pa
    curve.Nc = 10 ** data["loga"]
    curve.m = data["m"]
    ref = curves_in_seawater_for_free_corrosion["reference"]
    curve.reference = ref
    return curve

# Could use a dict or namedtuple here, but this standardizes things a bit better for users
class FatigueParams:
    """Data structure of parameters needed by fatigue calculation and standard calculations."""

    def __init__(self, **kwargs):
        """
        Creates an instance of `FatigueParams`.  There are three ways to initialize a FatigueParams
        instance with an associated S-N curve in the `fatpack` library:
        
        1. Specify a curve from DNV-RP-C203, Fatigue in Offshore Steel Structures.
           Input keywords are "dnv_type" = (one of) ['air', 'seawater', 'cathodic'] and
           "dnv_name" = (one of) [B1, B2, C, C1, C2, D, E, G F1, F3, G, W1, W2, W3]

        2. Specify the slope of the S-N curve and a point on the curve.
           Required keywords are "slope", "Nc" and "Sc".  Assumes a linear S-N curve.

        3. Specify the slope and the S-intercept point assuming a perflectly linear S-N curve
           (which might not be the actual ultimate failure stress of the material.
           Required keywords are "slope" and "S_intercept".
        
        Parameters
        ----------
        load2stress : float (optional)
            Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L.
            Required for Goodman correction calculations of DELs and for all Damage calculations
        dnv_type : string (optional)
            Type of S-N curve to use 'air' for structures in air, 'seawater' for structures in seawater, or
            'cathodic' for structures with cathodic protection in seawater.  Must also specify "dnv_name"
            From DNV-RP-C203, Fatigue of Metal Structures, - Edition October 2024
        dnv_name : string (optional)
            Grade of metal and hot spot exposure to use: [B1, B2, C, C1, C2, D, E, G F1, F3, G, W1, W2, W3].  Must also specify "dnv_type"
            From DNV-RP-C203, Fatigue of Metal Structures, - Edition October 2024
        slope : float (optional)
            Wohler exponent in the traditional SN-curve of S = A * N ^ -(1/m).  Must either specify Sc-Nc or S_intercept.
        Sc : float (optional)
            The S-value (amplitude of stress osciallation) on a point on the linear S-N curve.  Must also specify "Nc" and "slope"
        Nc : float (optional)
            The N-value (number of endurance cycles to failure) on a point on the linear S-N curve.  Must also specify "Sc" and "slope"
        ultimate_stress : float (optional)
            Ultimate stress for use in Goodman equivalent stress calculation
        S_intercept : float (optional)
            Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified
        rainflow_bins : int (optional)
            Number of bins used in rainflow analysis.
            Default: 100
        goodman_correction: boolean (optional)
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        """

        self.load2stress   = kwargs.get("load2stress", 1.0)
        dnv_name           = kwargs.get("dnv_name", "").upper()
        dnv_type           = kwargs.get("dnv_type", "").lower()
        slope              = np.abs( kwargs.get("slope", 4.0) )
        Sc                 = kwargs.get("Sc", None)
        Nc                 = kwargs.get("Nc", None)
        self.S_ult         = kwargs.get("ultimate_stress", 1.0)
        temp               = kwargs.get("S_intercept", 0.0)
        S_intercept        = temp if temp > 0.0 else self.S_ult
        self.bins          = kwargs.get("rainflow_bins", 100)
        self.bins          = kwargs.get("bins", self.bins)
        self.goodman       = kwargs.get("goodman_correction", False)
        self.goodman       = kwargs.get("goodman", self.goodman)

        for k in kwargs:
            if k not in ["load2stress", "dnv_name", "dnv_type",
                         "slope", "Sc", "Nc", "ultimate_stress",
                         "S_intercept", "rainflow_bins", "bins",
                         "goodman_correction", "goodman"]:
                print(f"Unknown keyword argument, {k}")

        if dnv_name is not None and len(dnv_name) > 0:
            if dnv_type.find("cath") >= 0:
                self.curve = dnv_in_seawater_cathodic(dnv_name)
            elif dnv_type.find("sea") >= 0:
                self.curve = dnv_in_seawater(dnv_name)
            elif dnv_type.find("air") >= 0:
                self.curve = dnv_in_air(dnv_name)
            else:
                raise ValueError(f'Unknown DNV RP-C203 curve type, {dnv_type}. Expected [air, seawater, or cathodic]')

        elif Sc is not None and Sc > 0.0 and Nc is not None and Nc > 0.0:
            # Set fatpack linear curve from S-N point
            self.curve = fatpack.LinearEnduranceCurve(Sc)
            self.curve.m = slope
            self.curve.Nc = Nc

        else:
            # Set fatpath curve from S-intercept
            self.curve = fatpack.LinearEnduranceCurve(S_intercept)
            self.curve.m = slope
            self.curve.Nc = 1

        
    def copy(self):
        newobj = FatigueParams(load2stress=self.load2stress,
                               ultimate_stress=self.S_ult,
                               bins=self.bins,
                               goodman=self.goodman)
        newobj.curve = self.curve
        return newobj

    
    def get_rainflow_matrix(self, chan, bins=20):
        """
        Computes ranflow counts and bins for input `chan`.
        Typically used internally to the class, but can be used for plotting and debugging too
        
        Parameters
        ----------
        chan : np.array
            Channel time series to calculate DEL for.
        bins : int
            Number of bins used in rainflow analysis.
        goodman: boolean (optional)
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        S_ult: float (optional)
            Ultimate stress/load for the material
        """

        S, Mrf = fatpack.find_rainflow_ranges(chan, k=256, return_means=True)
        data_arr = np.c_[S, Mrf]
        rowbin, colbin, rfcmat = fatpack.find_rainflow_matrix(data_arr, bins, bins, return_bins=True)

        X, Y = np.meshgrid(rowbin, colbin, indexing='ij')

        return rfcmat, X, Y

    
    def get_rainflow_counts(self, chan, bins, S_ult=None, goodman=False):
        """
        Computes ranflow counts and bins for input `chan`.
        Typically used internally to the class, but can be used for plotting and debugging too
        
        Parameters
        ----------
        chan : np.array
            Channel time series to calculate DEL for.
        bins : int
            Number of bins used in rainflow analysis.
        goodman: boolean (optional)
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        S_ult: float (optional)
            Ultimate stress/load for the material

        Returns
        -------
        N, S : 1darray
        The count and the characteristic value for the ranges.
        """

        try:
            ranges, Mrf = fatpack.find_rainflow_ranges(chan, k=256, return_means=True)
        except Exception:
            ranges = Mrf = np.zeros(1)
            
        if goodman:
            if S_ult is None:
                S_ult = self.S_ult
                
            if S_ult == 0.0:
                raise ValueError('Must specify an ultimate_stress to use Goodman correction')
                
            ranges = fatpack.find_goodman_equivalent_stress(ranges, Mrf, S_ult)

        success = False
        while not success:
            try:
                N, S = fatpack.find_range_count(ranges, bins)
                success = True
            except ValueError:
                bins *= 0.5
                if bins < 1:
                    print(ranges)
                    print(bins)
                    raise Exception("Failed to find bins for ranges")
                else:
                    bins = int(bins)
                    
        return N, S

    
    def compute_del(self, chan, elapsed_time, **kwargs):
        """
        Computes damage equivalent load of input `chan`.

        Parameters
        ----------
        chan : np.array
            Channel time series to calculate DEL for.
        rainflow_bins : int (optional kwargs)
            Number of bins used in rainflow analysis.
            Default: 100
        goodman_correction: boolean (optional kwargs)
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        """

        bins          = kwargs.get("rainflow_bins", self.bins)
        bins          = kwargs.get("bins", bins)
        goodman       = kwargs.get("goodman_correction", self.goodman)
        goodman       = kwargs.get("goodman", goodman)

        for k in kwargs:
            if k not in ["rainflow_bins", "bins", "goodman_correction", "goodman"]:
                print(f"Unknown keyword argument, {k}")
            
        # Default return values
        DEL = np.nan
        D   = np.nan

        if np.all(np.isnan(chan)):
            return DEL, D

        # Working with loads for DELs
        load2 = 1.0 if self.load2stress==0.0 else np.abs(self.load2stress)
        Nrf, Frf = self.get_rainflow_counts(chan, bins, S_ult=self.S_ult/load2, goodman=goodman)

        slope = self.curve.m1 if hasattr(self.curve, 'm1') else self.curve.m
        DELs = Frf ** slope * Nrf / elapsed_time
        DEL = DELs.sum() ** (1.0 / slope)
        # With fatpack do:
        #curve = fatpack.LinearEnduranceCurve(1.)
        #curve.m = slope
        #curve.Nc = elapsed
        #DEL = curve.find_miner_sum(np.c_[Frf, Nrf]) ** (1 / slope)
            
        return DEL
    
        
    def compute_damage(self, chan, **kwargs):
        """
        Computes Palmgren-Miner damage of input `chan`.

        Parameters
        ----------
        chan : np.array
            Channel time series to calculate DEL for.
        rainflow_bins : int (optional kwargs)
            Number of bins used in rainflow analysis.
            Default: 100
        goodman_correction: boolean (optional kwargs)
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        """

        bins          = kwargs.get("rainflow_bins", self.bins)
        bins          = kwargs.get("bins", bins)
        goodman       = kwargs.get("goodman_correction", self.goodman)
        goodman       = kwargs.get("goodman", goodman)

        for k in kwargs:
            if k not in ["rainflow_bins", "bins", "goodman_correction", "goodman"]:
                print(f"Unknown keyword argument, {k}")
            
        # Default return values
        D   = np.nan

        if np.all(np.isnan(chan)):
            return D

        if np.abs(self.load2stress) == 0.0:
            raise ValueError('Must specify load2stress value to compute Damage')

        Nrf, Srf = self.get_rainflow_counts(chan*self.load2stress, bins, goodman=goodman)

        D = self.curve.find_miner_sum(np.c_[Srf, Nrf])
        
        return D

    
    def get_stress(self, NN):
        return self.curve.get_stress(NN)


