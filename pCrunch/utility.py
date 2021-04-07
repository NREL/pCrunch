__author__ = ["Nikhar Abbas", "Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ["Nikhar Abbas", "Jake Nunemaker"]
__email__ = ["nikhar.abbas@nrel.gov", "jake.nunemaker@nrel.gov"]


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


# def save_yaml(outdir, fname, data_out):

#     ''' Save yaml file - ripped from WISDEM

#     Parameters:
#     -----------
#     outdir: str
#         directory to save yaml
#     fname: str
#         filename for yaml
#     data_out: dict
#         data to dump to yaml
#     '''
#     fname = os.path.join(outdir, fname)

#     if not os.path.isdir(outdir):
#         os.makedirs(outdir)

#     f = open(fname, "w")
#     yaml = ry.YAML()
#     yaml.default_flow_style = None
#     yaml.width = float("inf")
#     yaml.indent(mapping=4, sequence=6, offset=3)
#     yaml.dump(data_out, f)


# def load_yaml(fname_input, package=0):
#     ''' Import a .yaml file - ripped from WISDEM

#     Parameters:
#     -----------
#     fname_input: str
#         yaml file to load
#     package: bool
#         0 = yaml, 1 = ruamel
#     '''

#     if package == 0:
#         with open(fname_input) as f:
#             data = yaml.safe_load(f)
#         return data

#     elif package == 1:
#         with open(fname_input, 'r') as myfile:
#             text_input = myfile.read()
#         myfile.close()
#         ryaml = ry.YAML()
#         return dict(ryaml.load(text_input))
