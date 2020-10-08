'''
An example script to run a comparison between two DLC cases 
(sort of strange, but showcases some functionalities)
'''

# Python Modules and instantiation
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time
import os
# ROSCO toolbox modules 
from ROSCO_toolbox import utilities as ROSCO_utilites
fast_io = ROSCO_utilites.FAST_IO()
fast_pl = ROSCO_utilites.FAST_Plots()
# WEIS modules
from weis.aeroelasticse.Util import FileTools
# Batch Analysis
from pCrunch import pdTools
from pCrunch import Processing, Analysis


# Define input files paths
outfile_base      = '../../BatchOutputs/5MW_Land/5MW_Land_ROSCO/'
fname_case_matrix = '../../BatchOutputs/5MW_Land/5MW_Land_ROSCO/case_matrix.yaml'

# Load case matrix into dataframe
case_matrix = FileTools.load_yaml(fname_case_matrix, package=1)
cm = pd.DataFrame(case_matrix)

# Define DLCs
DLCs = [1.1, 1.3]
# Pare down case matrix for desired runs 
cm2 = pd.concat([cm[cm[('IEC', 'DLC')] == dlc].iloc[1:15] for dlc in DLCs]).reset_index()
# Pull desired outfile names from case matrix dataframe
outfiles = []
for dlc in DLCs:
    case_names = cm2[cm2[('IEC', 'DLC')] == dlc]['Case_Name']
    outnames = list(outfile_base + case_names + '.outb')
    outfiles.append(outnames)

# Initialize processing classes
fp = Processing.FAST_Processing()
fa = Analysis.Loads_Analysis()

# Set some processing parameters
fp.OpenFAST_outfile_list = outfiles
fp.dataset_names = ['DLC1.1', 'DLC1.3']
fp.t0 = 30
fp.parallel_analysis = True
fp.parallel_analysis = True
fp.results_dir = 'temp_results'
fp.save_LoadRanking = True
fp.save_SummaryStats = True
fp.verbose=True


# Load and save statistics and load rankings
stats, load_rankings = fp.batch_processing()

# Get wind speeds for processed runs
windspeeds, seed, IECtype, cm_wind = Processing.get_windspeeds(cm2, return_df=True)

# Get AEP
pp = Analysis.Power_Production()
Vavg = 10   # Average wind speed of cite
Vrange = [2,26] # Range of wind speeds being considered
bnums = int(len(windspeeds)/len(fp.dataset_names)) # Number of wind speeds per dataset for binning data
pp.windspeeds = list(set(windspeeds))
p = pp.gen_windPDF(Vavg, bnums, Vrange)
AEP = pp.AEP(stats)
print('AEP = {}'.format(AEP))

# ========== Plotting ==========
an_plts = Analysis.wsPlotting()
#  --- Time domain analysis --- 
filenames = [outfiles[0][2], outfiles[1][2]] # select the 2nd run from each dataset
cases = {'Baseline': ['Wind1VelX', 'GenPwr', 'BldPitch1', 'GenTq', 'RotSpeed']}
fast_dict = fast_io.load_fast_out(filenames, tmin=30)
fast_pl.plot_fast_out(cases, fast_dict)

# Plot some spectral cases
spec_cases = [('RootMyb1', 0), ('TwrBsFyt', 0)]
twrfreq = .0716
fig,ax = fast_pl.plot_spectral(fast_dict, spec_cases, show_RtSpeed=True, 
                        add_freqs=[twrfreq], add_freq_labels=['Tower'],
                        averaging='Welch')
ax.set_title('DLC1.1')

# Plot a data distribution
channels = ['RotSpeed']
caseid = [0, 1]
an_plts.distribution(fast_dict, channels, caseid, names=['DLC 1.1', 'DLC 1.3'])

# --- Batch Statistical analysis ---
# Bar plot
fig,ax = an_plts.stat_curve(windspeeds, stats, 'RotSpeed', 'bar', names=['DLC1.1', 'DLC1.3'])

# Turbulent power curve
fig,ax = an_plts.stat_curve(windspeeds, stats, 'GenPwr', 'line', stat_idx=0, names=['DLC1.1'])

plt.show()