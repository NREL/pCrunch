import seaborn as sns
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from wisdem.aeroelasticse.Util import FileTools as filetools

# Need this until you move things to ROSCO or something like that
from BatchAnalysis import post_process as pp

def plot_distribution(fast_data, channels, caseid, names):
    '''
    Distributions of data from desired fast runs and channels

    Parameters
    ----------
    fast_data: dict
        Dictionary containing OpenFAST output data from desired cases to compare
    channels: list
        List of strings of OpenFAST output channels
    caseid: list
        List of caseid's to compare
    names: list
        Names of the runs to compare. 
    fignum: ind, (optional)
        Specified figure number. Useful to plot over previously made plot
    '''
    for case in caseid:
        for channel in channels:
            fig, ax = plt.subplots()
            for name, data in zip(names, fast_data):
                print('Plotting distribution for {} - case {}'.format(channel, str(case)))
                sns.kdeplot(data[channel][case], shade=True, label=name)
                # sns.distplot(data[channel][case], label=name) # For a histogram
                ax.set_title(channel + ' - case ' + str(case))
            ax.legend()
    plt.show()

if __name__=="__main__":


    sim_names = ['ROSCO', 'legacy']
    sum_filenames =['../Processed/NREL5MW/5MW_Land_rosco_stats.yaml',
                    '../Processed/NREL5MW/5MW_Land_legacy_stats.yaml']
    # p_filenames = ['./NREL5MW/NREL5MW_legacy_DLC.p',
    #                './NREL5MW/NREL5MW_rosco_DLC.p']
    
    # Load fast output data statistics
    # all_data = []
    # for pickle in p_filenames:
    #     fast_data = pp.load_pickle(pickle)
    #     all_data.append(fast_data)

    all_stats=[]
    for sumfile in sum_filenames:
        sum_stats = filetools.load_yaml(sumfile, package=1)
        all_stats.append(sum_stats)

    # Build pandas dataframe of statistics
    df = pp.build_dataframe(all_stats,sim_names) 

    # Define channels and statistics to plot
    channels = []
    stats = []
    channels = channels + ['RtTSR', 'RotSpeed', 'RotSpeed']
    stats    = stats    + ['mean', 'std', 'mean']

    channels = channels + ['RotSpeed', 'GenPwr', 'TwrBsFxt', 'TwrBsFxt']
    stats    = stats    + ['max', 'mean', 'mean', 'std']

    # channels = channels + ['PtfmPitch', 'PtfmPitch']
    # stats    = stats    + ['mean', 'max']

    # runs = np.arange(0, 23).tolist()
    runs = np.arange(24, 47).tolist()

    for chan, stat in zip(channels,stats):
        df2 = pp.parse_DLC_stats(df, chan, stat, sim_names, runs)
        ax = df2.plot.bar()
        ax.legend(sim_names)
        plt.title(chan+' - '+stat)
    plt.show()

    # plot_distribution(all_data, ['TwrBsFxt'], [16, 18], sim_names)
    # plt.show()

