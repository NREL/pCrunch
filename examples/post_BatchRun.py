import os
from fnmatch import fnmatch

#import numpy as np
import pandas as pd

from pCrunch import Crunch, FatigueParams, read
from pCrunch.utility import load_yaml, save_yaml, get_windspeeds, convert_summary_stats


def valid_extension(fp):
    #return any([fnmatch(fp, ext) for ext in ["*.outb", "*.out"]])
    return fnmatch(fp, "*.outb")

if __name__ == '__main__':
    # Define input files paths
    output_dir = "/Users/gbarter/devel/WEIS/examples/03_NREL5MW_OC3_spar/outputs/03_OC3_optimization/openfast_runs"
    results_dir = os.path.join(output_dir, "results")
    save_results = True

    # Find outfiles
    outfiles = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if valid_extension(f)
    ]

    # Configure pCrunch
    magnitude_channels = {
        "RootMc1": ["RootMxc1", "RootMyc1", "RootMzc1"],
        "RootMc2": ["RootMxc2", "RootMyc2", "RootMzc2"],
        "RootMc3": ["RootMxc3", "RootMyc3", "RootMzc3"],
    }

    fatigue_channels = {"RootMc1": FatigueParams(lifetime=25, slope=10),
                        "RootMc2": FatigueParams(lifetime=25, slope=10),
                        "RootMc3": FatigueParams(lifetime=25, slope=10),
                        }

    channel_extremes = [
        "RotSpeed",
        "RotThrust",
        "RotTorq",
        "RootMc1",
        "RootMc2",
        "RootMc3",
    ]

    # Run pCrunch
    outputs = read(outfiles)
    cruncher = Crunch(outputs,
                      magnitude_channels=magnitude_channels,
                      fatigue_channels=fatigue_channels,
                      extreme_channels=channel_extremes,
                      trim_data=(0,),
                      )
    cruncher.process_outputs(cores=3, return_damage=True, goodman=True)

    if save_results:
        save_yaml(
            results_dir,
            "summary_stats.yaml",
            convert_summary_stats(cruncher.summary_stats),
        )

    # Get AEP from average windspeeds in time history
    turbine_class = 1
    windchan = "Wind1VelX"
    pwrchan = "GenPwr"
    loss_factor = 1.0 - 0.15 # Assume 15% loss for soil, farm effects, downtime
    cruncher.set_probability_turbine_class(windchan, turbine_class)
    AEP1, _ = cruncher.compute_aep(pwrchan, loss_factor)

    # Get AEP from case matrix file
    fname_case_matrix = os.path.join(output_dir, "case_matrix_DLC1.1_0.yaml")
    cm = pd.DataFrame( load_yaml(fname_case_matrix) )
    windspeeds, _, _, _ = get_windspeeds(cm, return_df=True)
    cruncher.set_probability_turbine_class(windspeeds, turbine_class)
    AEP2, _ = cruncher.compute_aep(pwrchan, loss_factor)

    print(f"AEP: {AEP1} vs {AEP2}")
    # # ========== Plotting ==========
    # an_plts = Analysis.wsPlotting()
    # #  --- Time domain analysis ---
    # filenames = [outfiles[0][2], outfiles[1][2]] # select the 2nd run from each dataset
    # cases = {'Baseline': ['Wind1VelX', 'GenPwr', 'BldPitch1', 'GenTq', 'RotSpeed']}
    # fast_dict = fast_io.load_FAST_out(filenames, tmin=30)
    # fast_pl.plot_fast_out(cases, fast_dict)

    # # Plot some spectral cases
    # spec_cases = [('RootMyb1', 0), ('TwrBsFyt', 0)]
    # twrfreq = .0716
    # fig,ax = fast_pl.plot_spectral(fast_dict, spec_cases, show_RtSpeed=True,
    #                         add_freqs=[twrfreq], add_freq_labels=['Tower'],
    #                         averaging='Welch')
    # ax.set_title('DLC1.1')

    # # Plot a data distribution
    # channels = ['RotSpeed']
    # caseid = [0, 1]
    # an_plts.distribution(fast_dict, channels, caseid, names=['DLC 1.1', 'DLC 1.3'])

    # # --- Batch Statistical analysis ---
    # # Bar plot
    # fig,ax = an_plts.stat_curve(windspeeds, stats, 'RotSpeed', 'bar', names=['DLC1.1', 'DLC1.3'])

    # # Turbulent power curve
    # fig,ax = an_plts.stat_curve(windspeeds, stats, 'GenPwr', 'line', stat_idx=0, names=['DLC1.1'])

    # plt.show()
