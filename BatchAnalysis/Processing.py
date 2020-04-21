from __future__ import print_function
import os, sys, time, shutil
from functools import partial
import multiprocessing as mp
import numpy as np
import ruamel_yaml as ry
import matplotlib.pyplot as plt
import pandas as pd

from wisdem.aeroelasticse.Util.ReadFASTout import ReadFASToutFormat
from wisdem.aeroelasticse.Util.FileTools   import get_dlc_label, load_case_matrix, load_file_list, save_yaml
from wisdem.aeroelasticse.Util.spectral    import fft_wrap

from ROSCO_toolbox.utilities import FAST_IO

from BatchAnalysis import Analysis, pdTools
def post_process():
    '''
    Function to run batch post processing. 
    '''

class FAST_Processing(object):
    '''
    A class with tools to post process batch OpenFAST output data
    '''

    def __init__(self, **kwargs):
        # Optional population class attributes from key word arguments
        self.OpenFAST_outfile_list = [[]] # list of lists containing absolute path to fast output files. Each inner list corresponds to a dataset to compare, and should be of equal length
        self.dataset_names = []       # (Optional) N labels that identify each dataset
        # (Optional) AeroelasticSE case matrix text file. Used to generated descriptions of IEC DLCS
        self.fname_case_matrix = ''

        # Analysis Control
        self.parallel_analysis = False  # True/False; Perform post processing in parallel
        self.parallel_cores = None      # None/Int>0; (Only used if parallel_analysis==True) number of parallel cores, if None, multiprocessing will use the maximum available
        self.verbose = False            # True/False; Enable/Disable non-error message outputs to screen

        # Analysis Options
        self.t0 = None       # float>=0    ; start time to include in analysis
        self.tf = None  # float>=0,-1 ; end time to include in analysis

        # Load ranking
        self.ranking_vars = [   ["RotSpeed"], 
                                ["TipDxc1", "TipDxc2", "TipDxc3"], 
                                ["TipDyc1", "TipDyc2", "TipDyc3"], 
                                ['RootMyb1', 'RootMyb2', 'RootMyb3'], 
                                ['RootMxb1', 'RootMxb2', 'RootMxb3'],
                                ['TwrBsFyt'],
                                ]  # List of lists
        self.ranking_stats = [  'max',
                                'max',
                                'max',
                                'max',
                                'max',
                                    ] # should be same length as ranking_vars
        
        # Save settings
        self.results_dir       = 'temp_results'
        self.save_LoadRanking  = False  # NJA - does not exist yet
        self.save_SummaryStats = False

        # Load kwargs
        for k, w in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        # Setup multiprocessing:
        if self.parallel_analysis:
            # Make sure multi-processing cores are valid
            if not self.parallel_cores:
                self.parallel_cores = min(mp.cpu_count(), len(filenames))
            elif self.parallel_cores == 1:
                self.parallel_analysis = False

        # Check for save directory
        if self.save_LoadRanking or self.save_SummaryStats:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)


        super(FAST_Processing, self).__init__()

        '''
        # Analysis Control
        self.plot_TimeSeries = False
        self.plot_FFT = False
        # self.plot_LoadRanking      = False
        
        # [str]       ; (Only used if plot_TimeSeries or plot_FFT = True) list of OpenFAST channels to plot
        self.plot_vars = []
        # [str]       ; (Optional)(Only used if plot_TimeSeries or plot_FFT = True) list axis labels for OpenFAST channels
        self.plot_vars_labels = []
        # float>0.    ; (Only used if plot_FFT = True) maximum frequency on x axis for FFT plots
        self.fft_x_max = 1.
        # True/False  ; (Only used if plot_FFT = True) Include 1P, 3P, 6P bands on FFT plots
        self.fft_show_RtSpeed = True
        # [floats]    ; (Optional)(Only used if plot_FFT = True) Additional frequencies to label on the FFT
        self.fft_include_f = []
        # [str]       ; (Optional)(Only used if plot_FFT = True) Legend names for additional frequencies to label on the FFT
        self.fft_include_f_labels = []
        # [str]       ; (Only used if save_LoadRanking = True) list of OpenFAST channels to perform load ranking on
        self.ranking_vars = []
        # str         ; (Only used if save_LoadRanking = True) statistic to rank loads by (min, max, mean, std, maxabs)
        self.ranking_statistic = 'maxabs'
        # int         ; (Only used if save_LoadRanking = True) number of loads to list per channel
        self.ranking_n_cases = 5
        '''

    def batch_processing(self):
        '''
        Run a full batch processing case!
        '''
        # ------------------ Input consistancy checks ------------------ #
        # Do we have a list of data?
        N = len(self.OpenFAST_outfile_list)
        if N == 0:
            print('Output files not defined! Populate: "FastPost.OpenFAST_outfile_list"')
            print('Quitting FastPost analysis')
            exit()

        # Do all the files exist?
        files_exist = True
        for i, flist in enumerate(self.OpenFAST_outfile_list):
            for fname in flist:
                if not os.path.exists(fname):
                    files_exist = False
                    if len(self.dataset_names) > 0:
                        print('Warning! File "{}" from {} does not exist.'.format(
                            fname, self.dataset_names[i]))
                        flist.remove(fname)
                    else:
                        print('Warning! File "{}" from dataset {} of {} does not exist.'.format(
                              fname, i+1, N))
                        flist.remove(fname)

        # # load case matrix data to get descriptive case naming
        # if self.fname_case_matrix == '':
        #     print('Warning! No case matrix file provided, no case descriptions will be provided.')
        #     self.case_desc = ['Case ID %d' % i for i in range(M)]
        # else:
        #     cases = load_case_matrix(self.fname_case_matrix)
        #     self.case_desc = get_dlc_label(cases, include_seed=True)

        # get unique file namebase for datasets
        self.namebase = []
        if len(self.dataset_names) > 0:
            # use filename safe version of dataset names
            self.namebase = ["".join([c for c in name if c.isalpha() or c.isdigit() or c in [
                                     '_', '-']]).rstrip() for i, name in zip(range(N), self.dataset_names)]
        elif len(self.OpenFAST_outfile_list) > 0:
            # use out file naming
            self.namebase = ['_'.join(os.path.split(flist[0])[1].split('_')[:-1])
                             for flist in self.OpenFAST_outfile_list]
        
        # check that names are unique
        if not len(self.namebase) == len(set(self.namebase)):
            self.namebase = []
        # as last resort, give generic name
        if not self.namebase:
            self.namebase = ['dataset' + ('{}'.format(i)).zfill(len(str(N-1))) for i in range(N)]
        
        # Run design comparison if filenames list has multiple lists
        if (len(self.OpenFAST_outfile_list) > 1) and (isinstance(self.OpenFAST_outfile_list, list)): 
            # Load stats and load rankings for design comparisons
            stats, load_rankings = self.design_comparison(self.OpenFAST_outfile_list, verbose=self.verbose)
        
        else:
            # Initialize Analysis
            loads_analysis = Analysis.Loads_Analysis()
            loads_analysis.verbose = self.verbose
            loads_analysis.t0 = self.t0
            loads_analysis.tf = self.tf

            # run analysis in parallel
            if self.parallel_analysis:
                pool = mp.Pool(self.parallel_cores)
                try:
                    stats_separate = pool.map(
                        partial(loads_analysis.full_loads_analysis, get_load_ranking=False), self.OpenFAST_outfile_list)
                except:
                    stats_separate = pool.map(partial(loads_analysis.full_loads_analysis, get_load_ranking=False), self.OpenFAST_outfile_list[0])
                pool.close()
                pool.join()

                # Re-sort into the more "standard" dictionary/dataframe format we like
                stats_df = pdTools.dict2df(stats_separate)
                stats_df = stats_df.stack().stack()
                stats_df = stats_df.swaplevel(1, 2).T.reset_index(drop=True).sort_index(axis=1)
                stats = pdTools.df2dict(stats_df)

                # Get load rankings after stats are loaded
                load_rankings = loads_analysis.load_ranking(stats, self.ranking_stats, self.ranking_vars,
                                            names=self.dataset_names, get_df=False)
           
            # run analysis in serial
            else:
                stats, load_rankings = loads_analysis.full_loads_analysis(self.OpenFAST_outfile_list, get_load_ranking=True)

        if self.save_SummaryStats:
            if isinstance(stats, dict):
                fname = self.namebase[0] + '_stats.yaml'
                save_yaml(self.results_dir, fname, stats)
            else:
                for namebase, st in zip(self.namebase, stats):
                    fname = namebase + '_stats.yaml'
                    save_yaml(self.results_dir, fname, st)
        if self.save_LoadRanking:
            if isinstance(load_rankings, dict):
                fname = self.namebase[0] + '_LoadRanking.yaml'
                save_yaml(self.results_dir, fname, load_rankings)
            else:
                for namebase, lr in zip(self.namebase, load_rankings):
                    fname = namebase + '_LoadRanking.yaml'
                    save_yaml(self.results_dir, fname, lr)


        return stats, load_rankings

    def design_comparison(self, filenames):
        '''
        Compare design runs

        Parameters:
        ----------
        filenames: list
            list of lists, where the inner lists are of equal length. 

        Returns:
        --------
        stats: dict
            dictionary of summary statistics data
        load_rankings: dict
            dictionary of load rankings
        '''


        # Make sure datasets are the same length
        ds_len = len(filenames[0])
        if any(len(dataset) != ds_len for dataset in filenames):
            raise ValueError('The datasets for filenames corresponding to the design comparison should all be the same size.')

        fnames = np.array(filenames).T.tolist()
        # Setup FAST_Analysis preferences
        loads_analysis = Analysis.Loads_Analysis()
        loads_analysis.verbose=self.verbose
        loads_analysis.t0 = self.t0
        loads_analysis.tf = self.tf

        if self.parallel_analysis: # run analysis in parallel
            # Make sure multi-processing cores are valid
            if not self.parallel_cores:
                self.parallel_cores = min(mp.cpu_count(), len(filenames))
            # run analysis
            pool = mp.Pool(self.parallel_cores)
            stats_separate = pool.map(partial(loads_analysis.full_loads_analysis, get_load_ranking=False), fnames)
            pool.close()
            pool.join()
        
            # Re-sort into the more "standard" dictionary/dataframe format we like
            stats_df = pdTools.dict2df(stats_separate)
            stats_df = stats_df.stack().stack()
            stats_df = stats_df.swaplevel(1, 2).T.reset_index(drop=True).sort_index(axis=1)
            stats = pdTools.df2dict(stats_df) 

            # Get load rankings after stats are loaded
            load_rankings = loads_analysis.load_ranking(stats, self.ranking_stats, self.ranking_vars, 
                                                        names=self.dataset_names, get_df=False)

        else: # run analysis in serial
            stats = []
            for file_sets in filenames:
                stats, load_rankings, stats.append(loads_analysis.full_loads_analysis(file_sets, get_load_ranking=False))



        return stats, load_rankings

def get_windspeeds(case_matrix, return_df=False):
    '''
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
    '''
    if isinstance(case_matrix, dict):
        cmatrix = case_matrix
    elif isinstance(case_matrix, pd.DataFrame):
        cmatrix = case_matrix.to_dict('list')
    else:
        raise TypeError('case_matrix must be a dict or pd.DataFrame.')


    windspeed = []
    seed = []
    IECtype = []
    # loop through and parse each inflow filename text entry to get wind and seed #
    for fname in  cmatrix[('InflowWind','Filename')]:
        if '.bts' in fname:
            obj = fname.split('U')[-1].split('_')
            obj2 = obj[1].split('Seed')[-1].split('.bts')
            windspeed.append(float(obj[0]))
            seed.append(float(obj2[0]))
            if 'NTM' in fname:
                IECtype.append('NTM')
            elif 'ETM' in fname:
                IECtype.append('NTM')
        elif 'ECD' in fname:
            obj = fname.split('U')[-1].split('.wnd')
            windspeed.append(float(obj[0]))
            obj2 = fname.split('ECD_')[-1].split('_U')
            seed.append(float(obj2[0]))
            IECtype.append('ECD')
        elif 'EWS' in fname:
            obj = fname.split('U')[-1].split('.wnd')
            windspeed.append(float(obj[0]))
            obj2 = fname.split('EWS_')[-1].split('_U')
            seed.append(float(obj2[0]))
            IECtype.append('EWS')
        
    if return_df:
        case_matrix = pd.DataFrame(case_matrix)
        case_matrix[('InflowWind','WindSpeed')] = windspeed
        case_matrix[('InflowWind','Seed')] = seed
        case_matrix[('InflowWind','IECtype')] = IECtype
        
        return windspeed, seed, IECtype, case_matrix
    
    else:
        return windspeed, seed, IECtype
