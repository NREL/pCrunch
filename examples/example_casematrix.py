import os
from fnmatch import fnmatch

#import numpy as np
import pandas as pd

from pCrunch import Crunch, FatigueParams, read
from pCrunch.utility import load_yaml, save_yaml, get_windspeeds, convert_summary_stats


def valid_extension(fp):
    return any([fnmatch(fp, ext) for ext in ["*.outb", "*.out"]])

if __name__ == '__main__':
    # Define input files paths
    thisdir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(thisdir, '..', 'pCrunch', 'test', 'data', 'DLC1p1')
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

    fatigue_channels = {"RootMc1": FatigueParams(slope=10, ultimate_stress=6e8),
                        "RootMc2": FatigueParams(slope=10, ultimate_stress=6e8),
                        "RootMc3": FatigueParams(slope=10, ultimate_stress=6e8),
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
    cruncher = Crunch(outputs=outputs,
                      magnitude_channels=magnitude_channels,
                      fatigue_channels=fatigue_channels,
                      extreme_channels=channel_extremes,
                      trim_data=(0,),
                      )
    cruncher.process_outputs(cores=3, goodman=True)

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
