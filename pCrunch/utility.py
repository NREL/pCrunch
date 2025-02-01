import os

import pandas as pd
import ruamel.yaml as ry
from scipy.special import gamma
from scipy.stats import rayleigh, weibull_min
import numpy as np

    
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
        ultimate_stress : float (optional)
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
        self.ult_stress    = kwargs.get("ultimate_stress", 1.0)
        temp               = kwargs.get("S_intercept", 0.0)
        self.S_intercept   = temp if temp > 0.0 else self.ult_stress
        self.bins          = kwargs.get("rainflow_bins", 100)
        self.return_damage = kwargs.get("return_damage", False)
        self.return_damage = kwargs.get("compute_damage", self.return_damage)
        self.goodman       = kwargs.get("goodman_correction", False)
        self.goodman       = kwargs.get("goodman", self.goodman)

        for k in kwargs:
            if k not in ["lifetime", "load2stress", "slope", "ultimate_stress",
                         "S_intercept", "rainflow_bins", "return_damage", "compute_damage",
                         "goodman_correction", "goodman"]:
                print(f"Unknown keyword argument, {k}")
            
        
    def copy(self):
        return FatigueParams(lifetime=self.lifetime,
                             load2stress=self.load2stress, slope=self.slope,
                             ult_stress=self.ult_stress, S_intercept=self.S_intercept,
                             bins=self.bins, return_damage=self.return_damage,
                             goodman=self.goodman)

def weibull_mean(x, k, xbar, kind="pdf"):
    """
    Weibull probability/cumulative density function using desired mean value.
    
    Parameters
    ----------
    x : list or np.array
        Vector of points to evaluate CDF
    k : float
        Weibull shape factor
    xbar : float
        Weibull mean value
    """
    scale = xbar / gamma(1.0 + 1.0 / k)
    if kind.lower() == "pdf":
        return weibull_min.pdf(x, k, loc=0, scale=scale)
    elif kind.lower() == "cdf":
        return weibull_min.cdf(x, k, loc=0, scale=scale)
    else:
        raise ValueError("Expected pdf/cdf for kind")

def rayleigh_mean(x, xbar, kind="pdf"):
    """
    Weibull probability/cumulative density function using desired mean value.
    
    Parameters
    ----------
    x : list or np.array
        Vector of points to evaluate CDF
    xbar : float
        Rayleigh mean value
    """
    scale = np.sqrt(2.0/np.pi) * xbar
    if kind.lower() == "pdf":
        return rayleigh.pdf(x, loc=0, scale=scale)
    elif kind.lower() == "cdf":
        return rayleigh.cdf(x, loc=0, scale=scale)
    else:
        raise ValueError("Expected pdf/cdf for kind")

def df2dict(df):
    """
    Build dictionary from pandas dataframe
    Parameters:
    -----------
    df: DataFrame
        DataFrame with summary stats (probably). Cannot have len(multiindex) > 3
    Returns:
    --------
    dfd: dict
        dictionary containing re-structured dataframe inputs
    """

    if len(df.columns[0]) == 3:
        dfd = [
            {
                level: df.xs(dset, axis=1).xs(level, axis=1).to_dict("list")
                for level in df.columns.levels[1]
            }
            for dset in df.columns.levels[0]
        ]

    elif len(df.columns[0]) == 2:
        dfd = {
            level: df.xs(level, axis=1).to_dict("list")
            for level in df.columns.levels[0]
        }

    elif len(df.columns[0]) == 1:
        dfd = df.to_dict("list")

    else:
        raise TypeError(
            "Converting DataFrames with multiindex > 3 to dictionaries is not supported"
        )

    dfd["meta"] = {"filename": list(df.index)}

    return dfd


def dict2df(sumstats, names=None):
    """
    Build pandas datafrom from list of summary statistics.

    Inputs:
    -------
    sumstats: list
        List of the dictionaries loaded from post_process.load_yaml
    names: list, optional
        List of names for each run. len(sumstats)=len(names)

    Returns:
    --------
    df: pd.DataFrame
        pandas dataframe
    """

    if isinstance(sumstats, list):
        if not names:
            names = ["dataset_" + str(i) for i in range(len(sumstats))]
        data_dict = {
            (name, outerKey, innerKey): values
            for name, sumdata in zip(names, sumstats)
            for outerKey, innerDict in sumdata.items()
            for innerKey, values in innerDict.items()
        }

    else:
        data_dict = {
            (outerKey, innerKey): values
            for outerKey, innerDict in sumstats.items()
            for innerKey, values in innerDict.items()
        }

    # Make dataframe
    df = pd.DataFrame(data_dict)

    return df

def save_yaml(outdir, fname, data):
    """
    Save data as '.yaml' file.

    Parameters:
    -----------
    outdir : str
        Directory to save data.
    fname : str
        Filename for output yaml.
    data : dict
        Data to save to yaml.
    """

    filepath = os.path.join(outdir, fname)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    #with open(filepath, "w") as f:
    #    yaml.dump(data, f, Dumper=Dumper, default_flow_style=False)
        
    # Write yaml with updated values
    myyaml = ry.YAML()
    myyaml.default_flow_style = None
    myyaml.width = float("inf")
    myyaml.indent(mapping=4, sequence=6, offset=3)
    myyaml.allow_unicode = False
    with open(filepath, "w", encoding="utf-8") as f:
        myyaml.dump(data, f)


def load_yaml(filepath):
    """
    Import a .yaml file.

    Parameters:
    -----------
    filepath : str
        File to load.
    """

    reader = ry.YAML(typ="safe", pure=True)
    with open(filepath, "r", encoding="utf-8") as f:
        data = reader.load(f)
    
    #with open(filepath) as f:
    #    data = yaml.load(f, Loader=Loader)

    return data



def yaml2df(filename, names=[]):
    """
    Read yaml containing summary stats into dataframe
    Parameters:
    -------
    filename:
        Name of yaml file to load.
    """

    data_dict = load_yaml("test.yaml", package=0)

    level = data_dict
    li = 0  # level index
    while isinstance(level, dict):
        li = li + 1
        if isinstance(level, dict):
            level = level[list(level.keys())[0]]

    data_list = []
    if li == 3:
        for key in data_dict.keys():
            data_list.append(data_dict[key])
        if not names:
            names = ["dataset_" + str(i) for i in range(len(data_list))]

    elif li == 2:
        data_list.append(data_dict)
        names = []
    else:
        raise TypeError(
            "{} is improperly structured for yaml2df.".format(filename)
        )

    df = dict2df(data_list, names=names)

    return df


def get_windspeeds(case_matrix, return_df=False):
    """
    Find windspeeds from case matrix
    Parameters:
    ----------
    case_matrix: dict
        case matrix data loaded from wisdem.aeroelasticse.Util.FileTools.load_yaml

    Returns:
    --------
    windspeed: list
        list of wind speeds
    seed: seed
        list of wind seeds
    IECtype: list
        list of IEC types
    case_matrix: pd.DataFrame
        case matrix dataframe with appended wind info
    """

    if isinstance(case_matrix, dict):
        cmatrix = case_matrix

    elif isinstance(case_matrix, pd.DataFrame):
        cmatrix = case_matrix.to_dict("list")

    else:
        raise TypeError("case_matrix must be a dict or pd.DataFrame.")

    # loop through and parse each inflow filename text entry to get wind and seed #
    if ("InflowWind", "HWindSpeed") in cmatrix:
        windspeed = [float(m) for m in cmatrix[("InflowWind", "HWindSpeed")]]
        seed = [float(m) for m in cmatrix[("TurbSim", "RandSeed1")]]
        IECtype = [None] * len(seed)

    elif ("InflowWind", "Filename_Uni") in cmatrix:
        windspeed = []
        seed = []
        IECtype = []
        for fname in cmatrix[("InflowWind", "Filename_Uni")]:
            if ".bts" in fname:
                obj = fname.split("U")[-1].split("_")
                obj2 = obj[1].split("Seed")[-1].split(".bts")
                windspeed.append(float(obj[0]))
                seed.append(float(obj2[0]))
                if "NTM" in fname:
                    IECtype.append("NTM")
                elif "ETM" in fname:
                    IECtype.append("NTM")
                elif "EWM" in fname:
                    IECtype.append("EWM")
                else:
                    IECtype.append("")

            elif "ECD" in fname:
                obj = fname.split("U")[-1].split("_D")[0].split(".wnd")[0]
                windspeed.append(float(obj))
                seed.append([])
                IECtype.append("ECD")

            elif "EWS" in fname:
                obj = fname.split("U")[-1].split("_D")[0].split(".wnd")[0]
                windspeed.append(float(obj))
                seed.append([])
                IECtype.append("EWS")

            else:
                print("Shouldn't get here")
                print(fname)
                breakpoint()

    if return_df:
        case_matrix = pd.DataFrame(case_matrix)
        case_matrix[("InflowWind", "WindSpeed")] = windspeed
        case_matrix[("InflowWind", "Seed")] = seed
        case_matrix[("InflowWind", "IECtype")] = IECtype

        return windspeed, seed, IECtype, case_matrix

    else:
        return windspeed, seed, IECtype


def convert_summary_stats(df):

    out = {}
    melted = df.to_dict(orient="list")
    for col, stat in melted:

        col = str(col)
        stat = str(stat)
        try:
            out[col]

        except KeyError:
            out[col] = {}

        out[col][stat] = list(melted[(col, stat)])
        out["meta"] = {"filename": list(df.index)}

    return out


# class wsPlotting(object):
#     '''
#     General plotting scripts.
#     '''

#     def __init__(self):
#         pass

#     def stat_curve(self, windspeeds, stats, plotvar, plottype, stat_idx=0, names=[]):
#         '''
#         Plot the turbulent power curve for a set of data.
#         Can be plotted as bar (good for comparing multiple cases) or line

#         Parameters:
#         -------
#         windspeeds: list-like
#             List of wind speeds to plot
#         stats: list, dict, or pd.DataFrame
#             Dict (single case), list(multiple cases), df(single or multiple cases) containing
#             summary statistics.
#         plotvar: str
#             Type of variable to plot
#         plottype: str
#             bar or line
#         stat_idx: int, optional
#             Index of datasets in stats to plot from

#         Returns:
#         --------
#         fig: figure handle
#         ax: axes handle
#         '''

#         # Check for valid inputs
#         if isinstance(stats, dict):
#             stats_df = pdTools.dict2df(stats)
#             if any((stat_inds > 0) or (isinstance(stat_inds, list))):
#                 print('WARNING: stat_ind = {} is invalid for a single stats dictionary. Defaulting to stat_inds=0.')
#                 stat_inds = 0
#         elif isinstance(stats, list):
#             stats_df = pdTools.dict2df(stats)
#         elif isinstance(stats, pd.DataFrame):
#             stats_df = stats
#         else:
#             raise TypeError(
#                 'Input stats must be a dictionary, list, or pd.DataFrame containing OpenFAST output statistics.')


#         # Check windspeed length
#         if len(windspeeds) == len(stats_df):
#             ws = windspeeds
#         elif int(len(windspeeds)/len(stats_df.columns.levels[0])) == len(stats_df):
#             ws = windspeeds[0:len(stats_df)]
#         else:
#             raise ValueError('Length of windspeeds is not the correct length for the input statistics')

#         # Get statistical data for desired plot variable
#         if plotvar in stats_df.columns.levels[0]:
#             sdf = stats_df.loc[:, (plotvar, slice(None))].droplevel([0], axis=1)
#         elif plotvar in stats_df.columns.levels[1]:
#             sdf = stats_df.loc[:, (slice(None), plotvar, slice(None))].droplevel([1], axis=1)
#         else:
#             raise ValueError("{} does not exist in the input statistics.".format(plotvar))

#         # Add windspeeds to data
#         sdf['WindSpeeds']= ws
#         # Group by windspeed and average each statistic (for multiple seeds)
#         sdf = sdf.groupby('WindSpeeds').mean()
#         # Final wind speed values
#         pl_windspeeds=sdf.index.values

#         if plottype == 'bar':
#             # Define mean and std dataframes
#             means = sdf.loc[:, (slice(None), 'mean')].droplevel(1, axis=1)
#             std = sdf.loc[:, (slice(None), 'std')].droplevel(1, axis=1)
#             # Plot bar charts
#             fig, ax = plt.subplots()
#             means.plot.bar(yerr=std, ax=ax, title=plotvar, capsize=2)
#             ax.legend(names,loc='upper left')

#         if plottype == 'line':
#             # Define mean, min, max, and std dataframes
#             means = sdf.loc[:, (sdf.columns.levels[0][stat_idx], 'mean')]
#             smax = sdf.loc[:, (sdf.columns.levels[0][stat_idx], 'max')]
#             smin = sdf.loc[:, (sdf.columns.levels[0][stat_idx], 'min')]
#             std = sdf.loc[:, (sdf.columns.levels[0][stat_idx], 'std')]

#             fig, ax = plt.subplots()
#             ax.errorbar(pl_windspeeds, means, [means - smin, smax - means],
#                          fmt='k', ecolor='gray', lw=1, capsize=2)
#             means.plot(yerr=std, ax=ax,
#                         capsize=2, lw=3,
#                         elinewidth=2,
#                         title=names[0] + ' - ' + plotvar)
#             plt.grid(lw=0.5, linestyle='--')

#         return fig, ax


#     def distribution(self, fast_data, channels, caseid, names=None, kde=True):
#         '''
#         Distributions of data from desired fast runs and channels

#         Parameters
#         ----------
#         fast_data: dict, list
#             List or Dictionary containing OpenFAST output data from desired cases to compare
#         channels: list
#             List of strings of OpenFAST output channels e.g. ['RotSpeed','GenTq']
#         caseid: list
#             List of caseid's to compare
#         names: list, optional
#             Names of the runs to compare
#         fignum: ind, (optional)
#             Specified figure number. Useful to plot over previously made plot

#         Returns:
#         --------
#         fig: figure handle
#         ax: axes handle
#         '''
#         # Make sure input types allign
#         if isinstance(fast_data, dict):
#             fd = [fast_data]
#         elif isinstance(fast_data, list):
#             if len(caseid) == 1:
#                 fd = [fast_data[caseid[0]]]
#             else:
#                 fd = [fast_data[case] for case in caseid]
#         else:
#             raise ValueError('fast_data is an improper data type')


#         # if not names:
#         #     names = [[]]*len(fd)

#         for channel in channels:
#             fig, ax = plt.subplots()
#             for idx, data in enumerate(fd):
#                 # sns.kdeplot(data[channel], shade=True, label='case '+ str(idx))
#                 sns.distplot(data[channel], kde=kde, label='case ' + str(idx))  # For a histogram
#                 ax.set_title(channel + ' distribution')

#                 units = data['meta']['attribute_units'][data['meta']['channels'].index(channel)]
#                 ax.set_xlabel('{} [{}]'.format(channel, units))
#                 ax.grid(True)
#             if names:
#                 ax.legend(names)

#         return fig, ax


# def plot_load_ranking(self, load_rankings, case_matrix, classifier_type,
#                     classifier_names=[], n_rankings=10, caseidx_labels=False):
#     '''
#     Plot load rankings

#     Parameters:
#     -----------
#     load_rankings: list, dict, or pd.DataFrame
#         Dict (single case), list(multiple cases), df(single or multiple cases) containing
#         load rankings.
#     case_matrix: dict or pdDataFrame
#         Information mapping classifiers to load_rankings.
#         NOTE: the case matrix must have wind speeds in ('InflowWind','WindSpeeds') if you
#         wish to plot w.r.t. wind speed
#     classifier_type: tuple or str
#         classifier to denote load ranking cases. e.g. classifier_type=('IEC','DLC') will separate
#         the load rankings by DLC type, assuming the case matrix is properly set up to map each
#         DLC to the load ranking case
#     classifier_names: list, optional
#         Naming conventions for each classifier type for plotting purposes
#     n_rankings: int, optional
#         number of load rankings to plot
#     caseidx_labels: bool, optional
#         label x-axis with case index if True. If false, will try to plot with wind speed labels
#         if they exist, then fall abck to case indeces.

#     TODO: Save figs
#     '''

#     # flag_DLC_name = False
#     # n_rankings = 10
#     # fig_ext = '.pdf'
#     # font_size = 10
#     # classifier_type = ('ServoDyn', 'DLL_FileName')
#     # classifiers = list(set(cmw[classifier_type]))
#     # classifier_names = ['ROSCO', 'legacy']

#     # Check for valid inputs
#     if isinstance(load_rankings, dict):
#         load_ranking_df = pdTools.dict2df(load_rankings)
#     elif isinstance(load_rankings, list):
#         load_ranking_df = pdTools.dict2df(load_rankings)
#     elif isinstance(load_rankings, pd.DataFrame):
#         load_ranking_df = load_rankings
#     else:
#         raise TypeError(
#             'Input stats must be a dictionary, list, or pd.DataFrame containing OpenFAST output statistics.')

#     # Check multiindex size
#     if len(load_ranking_df) == 2:
#         load_ranking_df = pd.concat([load_ranking_df], keys=[dataset_0])

#     # Check for classifier_names
#     classifiers = list(set(case_matrix[classifier_type]))
#     if not classifier_names:
#         classifier_names = ['datatset_{}'.format(idx) for idx in range(len(classifiers))]

#     # Check for wind speeds in case_matrix
#     if not caseidx_labels:
#         try:
#             windspeeds = case_matrix[('InflowWind','WindSpeed')]
#         except:
#             print('Unable to find wind speeds in case_matrix, plotting w.r.t case index')
#             caseidx_labels=True


#     # Define a color map
#     clrs = np.array([[127, 60, 141],
#                     [17, 165, 121],
#                     [57, 105, 172],
#                     [242, 183, 1],
#                     [231, 63, 116],
#                     [128, 186, 90],
#                     [230, 131, 16],
#                     [256, 256, 256]]) / 256.

#     # Get channel names
#     channels = load_ranking_df.columns.levels[1]

#     # initialize some variables
#     colors = np.zeros((n_rankings, 3))
#     labels = [''] * n_rankings
#     labels_index = [''] * n_rankings
#     fig_list = []
#     ax_list = []
#     # --- Generate plots ---
#     for cidx, channel in enumerate(channels):
#         # Pull out specific channel
#         cdf = load_ranking_df.loc[:, (slice(None), channel, slice(None))].droplevel(1, axis=1)
#         # put the load ranking from each dataset in a list so we can combine them
#         cdf_list = [cdf[dataset] for dataset in cdf.columns.levels[0]]
#         chan_df = pd.concat(cdf_list) # combine all load rankings
#         chan_stats = chan_df.columns.values # pull out the names of the columns
#         chan_df.sort_values(by=chan_stats[0], ascending=False, inplace=True) # sort
#         chan_df.reset_index(inplace=True, drop=True) # re-index

#         # find colors and labels for plots
#         for i in range(n_rankings):
#             classifier = case_matrix[classifier_type][chan_df[chan_stats[1]][i]]
#             colors[i, :] = clrs[min(len(clrs), classifiers.index(classifier))]

#             if not caseidx_labels:
#                 ws = windspeeds[chan_df[chan_stats[1]][i]]
#                 labels[i] = classifier_names[classifiers.index(classifier)] + ' - ' + str(ws) + ' m/s'
#             else:
#                 labels[i] = classifier_names[classifiers.index(classifier)] + ' - Case ' + str(chan_df[chan_stats[1]][i])
#     #         labels_index = ['case {}'.format(case) for case in chan_df[chan_stats[1]][0:n_rankings]]

#         # make plot
#         fig, ax = plt.subplots()
#         chan_df[chan_stats[0]][0:n_rankings].plot.bar(color=colors)
#         ax.set_ylabel(channel)
#         ax.set_xticklabels(labels, rotation=45, ha='right')
#         plt.draw()

#         fig_list.append(fig)
#         ax_list.append(ax)

#     #     if case_idx_labels:
#     #         ax.set_xlabel('DLC [-]', fontsize=font_size+2, fontweight='bold')
#     # #         ax.set_xticklabels(np.arange(n_rankings), labels=labels)
#     #         ax.set_xticklabels(labels)
#     #     else:
#     #         #         ax.set_xticklabels(np.arange(n_rankings), labels=labels)

#     return fig_list, ax_list
