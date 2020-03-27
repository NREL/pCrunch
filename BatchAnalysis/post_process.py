'''
Tools for post-processing batch runs.  
'''
import os
import numpy as np
import ruamel_yaml as ry
import pickle
import pandas as pd
from wisdem.aeroelasticse.Util import FileTools as filetools
from ROSCO_toolbox.utilities import FAST_IO 

# Instatiate some things
FAST_IO = FAST_IO()

def save_pickle(outdir, fname, data_out):
    '''
    Save a pickle. 

    Parameters
    ----------
    outdir: str
        directory to write yaml to
    fname: str
        filename for output data
    data_out: dict
        data to write to pickle
    '''
    print('Saving pickle file {}...'.format(fname.split('/')[-1]))
    fname = os.path.join(outdir, fname)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    f = open(fname, "wb")
    pickle.dump(data_out, f)


def load_pickle(fname):
    '''
    Load a pickle.

    Parameters
    ----------
    fname: string
        name of pickle to load

    Returns
    -------
    data: -
        data that lives in the pickle.
    '''
    print('Loading pickle file {}'.format(fname.split('/')[-1]))
    f = open(fname, "rb")
    data = pickle.load(f)

    return data
    

def get_summary_stats(allinfo, alldata, t0=0, tf=100000):
    '''
    Get summary statistics from openfast output data. Particularly of interest for DLC analysis.

    Parameters
    ----------
    allinfo: list
        List of dictionary information for OpenFAST runs (returned from ROSCO_toolbox.FAST_IO.load_output)
    alldata: list
        List of arrays corresponding to allinfo (returned from ROSCO_toolbox.FAST_IO.load_output)
    fname_file_list: list
        list of strings with openfast output filenames to collect summary statistics from
    t0: float, optional
        start time
    tf: float, optional 
        end time

    Returns
    -------
    data_out: dict
        Dictionary containing summary statistics
    fast_outdata: dict, optional
        Dictionary of all OpenFAST output data. Only returned if return_data=true
    '''
    sum_data = {}
    for (fidx, (data, info)) in enumerate(zip(alldata,allinfo)):
        print('Processing data for {}'.format(info['name']))
        data = FAST_IO.trim_output(info, data, t0, tf)

        for channel in info['channels']:
            if channel != 'Time':
                try: 
                    if channel not in sum_data.keys():
                        sum_data[channel] = {}
                        sum_data[channel]['min']  = []
                        sum_data[channel]['max']  = []
                        sum_data[channel]['std']  = []
                        sum_data[channel]['mean'] = []
                        sum_data[channel]['abs']  = []

                    sum_data[channel]['min'].append(
                        float(min(data[:, info['channels'].index(channel)])))
                    sum_data[channel]['max'].append(
                        float(max(data[:, info['channels'].index(channel)])))
                    sum_data[channel]['std'].append(
                        float(np.std(data[:, info['channels'].index(channel)])))
                    sum_data[channel]['mean'].append(
                        float(np.mean(data[:, info['channels'].index(channel)])))
                    sum_data[channel]['abs'].append(
                        float(max(np.abs(data[:, info['channels'].index(channel)]))))
                except ValueError:
                    print('Error loading data from {}.', channel)
                except: 
                    print('{} is not in available OpenFAST output data'.format(channel))


    return sum_data


def parse_batch_data(allinfo, alldata, save_channels=None, t0=0, tf=1000000):
    '''
    Converts openfast output data to a dictionary. 
    Might be easier to parse and share output data this way.

    Parameters
    ----------
    allinfo: list
        List of dictionary information for OpenFAST runs (returned from ROSCO_toolbox.FAST_IO.load_output)
    alldata: list
        List of arrays corresponding to allinfo (returned from ROSCO_toolbox.FAST_IO.load_output)
    save_channels: list (optional)
        List of strings containing OpenFAST output channels to be saved
    t0: float
        start time
    tf: float
        end time

    Returns
    -------
    fast_outdata:
        Dictionary of OpenFAST output data
    '''

    if not save_channels:
        save_channels = allinfo[0]['channels']

    fast_outdata = {}
    for (fidx, (data, info)) in enumerate(zip(alldata, allinfo)):
        print('Parsing batch data for {}'.format(info['name']))
        data = FAST_IO.trim_output(info, data, t0, tf)
        for channel in info['channels']:
            if channel in save_channels:
                if channel not in fast_outdata.keys():
                    fast_outdata[channel] = []

                fast_outdata[channel].append(np.array(data[:, info['channels'].index(channel)]).tolist())

    return fast_outdata


def build_dataframe(sumstats, names=None):
    '''
    Build python datafrom from list of summary statistics

    Inputs:
    -------
    sumstats: list
        List of the dictionaries loaded from post_process.load_yaml
    names: list
        List of names for each run. len(sumstats)=len(names)
    '''
    if names:
        data_dict = {(outerKey, name, innerKey): values for name, sumdata in zip(names, sumstats)
                    for outerKey, innerDict in sumdata.items() for innerKey, values in innerDict.items()}
    else: 
        data_dict = {(outerKey, innerKey): values for sumdata in sumstats
                    for outerKey, innerDict in sumdata.items() for innerKey, values in innerDict.items()}

    # Make dataframe
    df = pd.DataFrame(data_dict)

    return df

def parse_DLC_stats(fast_df, channel, stat, names=None, runs=None):
    '''
    Re-arrange/clean up pandas dataframe to contain desired DLC statistics 

    Parameters
    ----------
    fast_df: DataFrame
        Pandas dataframe containing DLC statistics
    channel: str
        Desired channel ('RotSpeed','GenPwr',etc...)
    stat: str
        Desired statistic ('max','min,'mean',etc...)
    names: list
        Names of the runs to compare. 
        NOTE: len(names) = len(runs)
    runs: list, (optional)
        Specific runs to plot. If none, all runs will be plotted. 
        NOTE: len(names) = len(runs)
    
    '''
    if names: 
        l1 = []
        l2 = []
        l3 = []
        {l1.append(channel) for name in names}
        {l2.append(name) for name in names}
        {l3.append(stat) for name in names}
        a3 = zip(*[l1, l2, l3])
    else:
        l1 = [channel]
        l2 = [stat]
        a3 = zip(*[l1, l2])

    if runs:
        df = fast_df.loc[runs, a3]
    else:
        df = fast_df.loc[:, a3]

    return df

    
if __name__=="__main__":

    out_folder = '/Users/nabbas/Documents/WindEnergyToolbox/BatchAnalysis/Processed/NREL5MW'
    
    outfilename_base = ['5MW_Land_legacy',
                          '5MW_Land_rosco']#,
                        #   '5MW_OC3Spar_legacy',
                        #   '5MW_OC3Spar_rosco']

    
    data_folders = ['/Users/nabbas/Documents/WindEnergyToolbox/BatchAnalysis/BatchOutputs/5MW_Land/5MW_Land_legacy',
                    '/Users/nabbas/Documents/WindEnergyToolbox/BatchAnalysis/BatchOutputs/5MW_Land/5MW_Land_ROSCO']#,
                    # '/Users/nabbas/Documents/WindEnergyToolbox/DLC_Analysis/outputs_NREL5MW/5MW_OC3Spar_legacy',
                    # '/Users/nabbas/Documents/WindEnergyToolbox/DLC_Analysis/outputs_NREL5MW/5MW_OC3Spar_ROSCO']
    
    fio = FAST_IO()
    for stats_outfilename, data_folder in zip(outfilename_base, data_folders):
        outfiles = []
        for file in os.listdir(data_folder):
            if file.endswith('.outb') or file.endswith('.out'):
                outfiles.append(os.path.join(data_folder, file))

        allinfo, alldata = fio.load_output(outfiles)

        summary_stats = get_summary_stats(allinfo, alldata, t0=150)
        filetools.save_yaml(out_folder, stats_outfilename + '_stats.yaml', summary_stats)

        # Save output data
        save_channels = [ "Time",
            # ElastoDyn
            "BldPitch1", "BldPitch2", "BldPitch3", "Azimuth", "RotSpeed", "GenSpeed",
            "TwrClrnc1", "TwrClrnc2", "TwrClrnc3", "NcIMUTAxs", "NcIMUTAys",
            "NcIMUTAzs", "TTDspFA", "TTDspSS", "TTDspTwst", 
            "PtfmSurge", "PtfmSway", "PtfmHeave", "PtfmRoll", "PtfmPitch", "PtfmYaw", "RotThrust", 
            "LSSGagFya", "LSSGagFza", "RotTorq", "LSSGagMya", "LSSGagMza", 
            "TwrBsFxt", "TwrBsFyt", "TwrBs/Fzt", "TwrBsMxt", "TwrBsMyt", "TwrBsMzt",
            # ServoDyn
            "GenPwr", "GenTq",
            # AeroDyn15
            "RtArea", "RtVAvgxh", 
            # InflowWind
            "Wind1VelX", "Wind1VelY", "Wind1VelZ"
        ]

        # fio = FAST_IO()
        # allinfo, alldata = fio.load_output(fname_file_list)
        # fast_outdata = parse_batch_data(allinfo, alldata, save_channels=save_channels)
        # fast_outfilename = "NREL5MW_le_data.p"
        # save_pickle(out_folder, fast_outfilename + '_DLC.p', fast_outdata)
