# pCrunch's Crunch class

The `Crunch` class is a general analysis tool for batches of time-series based data across multiple environmental conditions (i.e., a full wind speed and turbulence seed sweep). The methods are agnostic to the aeroelastic multibody simulation tool (OpenFAST or HAWC2 or Bladed or QBlade or in-house equivalents). The `AeroelasticOutput` class provides the data containers for each individual simulation.  The `AeroelasticOutput` class provides many analysis capabilities and the `Crunch` class extends them into their batch versions.

The `Crunch` class supports keeping all time series data in memory and a lean "streaming" version where outputs are processed and then deleted, retaining only the critical statistics and analysis outputs.

This file lays out some workflows and showcases capabilities of the `Crunch` class.  It is probably best to walk through the examples of the `AeroelasticOutput` class first.

## Creating a new class instance

The `Crunch` class can be initialized from a list of AeroelasticOutput instances or none, in order to setup a "streaming" analysis.  Pleaes see the AeroelasticOutput example for the various means to initialize one of its instances.  pCrunch provides a reader for OpenFAST output files (both binary and ascii) and common Python data structures are also supported.  To extend pCrunch for use with other aeroelastic multibody codes, users could simply use the `openfast_readers.py` file as a template.  Here are some examples:


```python
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pCrunch import Crunch, read, FatigueParams

thisdir = os.path.realpath('')
datadir = os.path.join(thisdir, '..', 'pCrunch', 'test', 'data')

# OpenFAST output files
filelist = glob.glob( os.path.join(datadir, '*.out') )
filelist.sort()
print(f"Found {len(filelist)} files.")

# Read all outputs into a list
outputs = [read(m) for m in filelist[1:]]

# Vector magnitudes
mc = {
    "RootMc1": ["RootMxc1", "RootMyc1", "RootMzc1"],
    "RootMc2": ["RootMxc2", "RootMyc2", "RootMzc2"],
    "RootMc3": ["RootMxc3", "RootMyc3", "RootMzc3"],
}

# Channel-specific fatigue properties
fc = {
    "RootMc1": FatigueParams(lifetime=25.0, slope=10.0, ultimate_stress=6e8, load2stress=250.0, S_intercept=5e9),
    "RootMc2": FatigueParams(lifetime=25.0, slope=10.0, ultimate_stress=6e8, load2stress=250.0, S_intercept=5e9),
    "RootMc3": FatigueParams(lifetime=25.0, slope=10.0, ultimate_stress=6e8, load2stress=250.0, S_intercept=5e9),
}

# Channels to focus on for extreme event tabulation
ec = ["RotSpeed", "RotThrust", "RotTorq"]

# Standard use case with all outputs read prior to use of Crunch.
mycruncher = Crunch(outputs)

# Can also add some batch data operations in the constructor (many more available in Batch Processing below)
mycruncher_mc = Crunch(outputs, magnitude_channels=mc, trim_data=[40, 80], fatigue_channels=fc, extreme_channels=ec)

# When planning on adding outputs later, you still need create a Crunch object that is initially empty of data
# The `lean` flag says that the outputs should be processed, but not stored in memory
mycruncher_lean = Crunch(outputs=[], lean=True)

# Can still add the batch operations to be done later when outputs are added
mycruncher_lean_mc = Crunch(outputs=[], lean=True, magnitude_channels=mc, trim_data=[40,80], fatigue_channels=fc, extreme_channels=ec)
```

    Found 4 files.


## Crunching the data

### With full memory storage
The Crunch class can batch process the outputs using one or more processors up to the number available workstation cores.  This computes the essential statistics for each output.


```python
# Process all outputs in parallel
mycruncher.process_outputs(cores=1)

# Process all outputs and override any prior input setting (especially in fatigue calculation)
mycruncher_mc.process_outputs(return_damage=True)
```

The key outputs that are stacked together for each output are:

- Summary statistics
- Load ranking
- Extreme event table
- Damage equivalent loads (DELs)
- Palmgren-Miner damage


```python
# The summary stats per each file are here:
mycruncher.summary_stats
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Time</th>
      <th>WindVxi</th>
      <th>WindVyi</th>
      <th>WindVzi</th>
      <th>WaveElev</th>
      <th>Wave1Vxi</th>
      <th>Wave1Vyi</th>
      <th>Wave1Vzi</th>
      <th>Wave1Axi</th>
      <th>Wave1Ayi</th>
      <th>...</th>
      <th>Fair8Ang</th>
      <th>Anch8Ten</th>
      <th>Anch8Ang</th>
      <th>TipSpdRat</th>
      <th>RotCp</th>
      <th>RotCt</th>
      <th>RotCq</th>
      <th>RootMc1</th>
      <th>RootMc2</th>
      <th>RootMc3</th>
    </tr>
    <tr>
      <th></th>
      <th>min</th>
      <th>min</th>
      <th>min</th>
      <th>min</th>
      <th>min</th>
      <th>min</th>
      <th>min</th>
      <th>min</th>
      <th>min</th>
      <th>min</th>
      <th>...</th>
      <th>integrated</th>
      <th>integrated</th>
      <th>integrated</th>
      <th>integrated</th>
      <th>integrated</th>
      <th>integrated</th>
      <th>integrated</th>
      <th>integrated</th>
      <th>integrated</th>
      <th>integrated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DLC2.3_1.out</th>
      <td>40.0</td>
      <td>8.200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.812</td>
      <td>-0.5420</td>
      <td>0.0</td>
      <td>-0.7480</td>
      <td>-0.7230</td>
      <td>0.0</td>
      <td>...</td>
      <td>3289.97500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>200.575785</td>
      <td>7.928011</td>
      <td>19.910354</td>
      <td>1.038984</td>
      <td>205721.341721</td>
      <td>213361.952123</td>
      <td>222926.620682</td>
    </tr>
    <tr>
      <th>DLC2.3_2.out</th>
      <td>40.0</td>
      <td>8.197</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.055</td>
      <td>-0.5438</td>
      <td>0.0</td>
      <td>-0.5059</td>
      <td>-0.3195</td>
      <td>0.0</td>
      <td>...</td>
      <td>3284.85275</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>202.995350</td>
      <td>8.196020</td>
      <td>20.209087</td>
      <td>1.063160</td>
      <td>209783.673742</td>
      <td>218196.946790</td>
      <td>226368.344817</td>
    </tr>
    <tr>
      <th>DLC2.3_3.out</th>
      <td>40.0</td>
      <td>8.197</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.135</td>
      <td>-0.5403</td>
      <td>0.0</td>
      <td>-0.5156</td>
      <td>-0.2482</td>
      <td>0.0</td>
      <td>...</td>
      <td>3286.98625</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>202.480952</td>
      <td>8.109685</td>
      <td>20.148358</td>
      <td>1.055470</td>
      <td>209258.712717</td>
      <td>216929.128758</td>
      <td>225974.727664</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 952 columns</p>
</div>




```python
# These are indexable by channel, stat:
mycruncher.summary_stats["RootMc1"]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>std</th>
      <th>mean</th>
      <th>median</th>
      <th>abs</th>
      <th>integrated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DLC2.3_1.out</th>
      <td>459.805830</td>
      <td>9134.167593</td>
      <td>2707.224813</td>
      <td>5142.646328</td>
      <td>6147.974788</td>
      <td>9134.167593</td>
      <td>205721.341721</td>
    </tr>
    <tr>
      <th>DLC2.3_2.out</th>
      <td>277.648587</td>
      <td>9079.452302</td>
      <td>2709.503118</td>
      <td>5244.431821</td>
      <td>6430.205456</td>
      <td>9079.452302</td>
      <td>209783.673742</td>
    </tr>
    <tr>
      <th>DLC2.3_3.out</th>
      <td>347.604352</td>
      <td>8986.223847</td>
      <td>2707.345218</td>
      <td>5231.190499</td>
      <td>6669.062227</td>
      <td>8986.223847</td>
      <td>209258.712717</td>
    </tr>
  </tbody>
</table>
</div>




```python
mycruncher.summary_stats["RootMc1"]['min']
```




    DLC2.3_1.out    459.805830
    DLC2.3_2.out    277.648587
    DLC2.3_3.out    347.604352
    Name: min, dtype: float64




```python
# Or by file
mycruncher.summary_stats.loc["DLC2.3_1.out"]
```




    Time      min               40.000000
    WindVxi   min                8.200000
    WindVyi   min                0.000000
    WindVzi   min                0.000000
    WaveElev  min               -0.812000
                                ...      
    RotCt     integrated        19.910354
    RotCq     integrated         1.038984
    RootMc1   integrated    205721.341721
    RootMc2   integrated    213361.952123
    RootMc3   integrated    222926.620682
    Name: DLC2.3_1.out, Length: 952, dtype: float64




```python
# Load rankings are manipulations of the summary statistics table
# All channels and statistics are available
mycruncher.get_load_rankings(['RootMc1'],['max'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>channel</th>
      <th>stat</th>
      <th>val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DLC2.3_1.out</td>
      <td>RootMc1</td>
      <td>max</td>
      <td>9134.167593</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Damage equivalent loads are found here:
mycruncher_mc.dels
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RootMc1</th>
      <th>RootMc2</th>
      <th>RootMc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DLC2.3_1.out</th>
      <td>2557.248362</td>
      <td>5801.906159</td>
      <td>2074.270412</td>
    </tr>
    <tr>
      <th>DLC2.3_2.out</th>
      <td>2759.655817</td>
      <td>4632.573610</td>
      <td>2138.262799</td>
    </tr>
    <tr>
      <th>DLC2.3_3.out</th>
      <td>2791.460474</td>
      <td>5839.945621</td>
      <td>2213.542813</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Palmgren-Miner damage can be viewed with (although it is not computed without a `return_damage=True`
mycruncher_mc.damage
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RootMc1</th>
      <th>RootMc2</th>
      <th>RootMc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DLC2.3_1.out</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>DLC2.3_2.out</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>DLC2.3_3.out</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extreme events table. For each channel, there is a list of the extreme condition for each output case
mycruncher.extremes
```




    {'RotSpeed': [{'Time': 61.8,
       'RotSpeed': 11.1,
       'RotTorq': 844.0,
       'RotThrust': 369.0},
      {'Time': 61.9, 'RotSpeed': 11.28, 'RotTorq': 159.4, 'RotThrust': 367.0},
      {'Time': 61.9, 'RotSpeed': 11.33, 'RotTorq': 140.8, 'RotThrust': 317.0}],
     'RotTorq': [{'Time': 54.45,
       'RotSpeed': 10.6,
       'RotTorq': 2650.0,
       'RotThrust': 546.0},
      {'Time': 54.4, 'RotSpeed': 10.74, 'RotTorq': 2701.0, 'RotThrust': 554.0},
      {'Time': 54.4, 'RotSpeed': 10.61, 'RotTorq': 2638.0, 'RotThrust': 575.1}],
     'RotThrust': [{'Time': 51.6,
       'RotSpeed': 10.1,
       'RotTorq': 2410.0,
       'RotThrust': 759.0},
      {'Time': 60.35, 'RotSpeed': 10.41, 'RotTorq': -1041.0, 'RotThrust': 786.9},
      {'Time': 60.35, 'RotSpeed': 10.48, 'RotTorq': -1046.0, 'RotThrust': 746.2}]}



### Crunching in "lean / streaming" mode

If operating in "lean / streaming" mode, the outputs can either be processed one at a time, or even more lean, the summary statistics themselves can be passed to the `cruncher` object to append to the running list.


```python
# Adding AeroelasticOutput objects in lean / streaming mode
for iout in outputs:
    mycruncher_lean.add_output( iout ) # Each output is processed without retaining the full time series

# Adding statistics incrementally
results_pool = []
for iout in outputs:
    iresults = mycruncher_lean_mc.process_single( iout ) # This could be the result of parallelized function
    results_pool.append( iresults )

# After parallel processing is complete, assemble all the statistic for batch analysis
for iresults in results_pool:
    fname, stats, extremes, dels, damage =  iresults
    mycruncher_lean_mc.add_output_stats(fname, stats, extremes, dels, damage)
```


```python
# Results are the same as the full-memory approach above
mycruncher_lean.summary_stats["RootMc1"]['min']
```




    DLC2.3_1.out    459.805830
    DLC2.3_2.out    277.648587
    DLC2.3_3.out    347.604352
    Name: min, dtype: float64




```python
mycruncher_lean_mc.dels
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RootMc1</th>
      <th>RootMc2</th>
      <th>RootMc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DLC2.3_1.out</th>
      <td>2557.248362</td>
      <td>5801.906159</td>
      <td>2074.270412</td>
    </tr>
    <tr>
      <th>DLC2.3_2.out</th>
      <td>2759.655817</td>
      <td>4632.573610</td>
      <td>2138.262799</td>
    </tr>
    <tr>
      <th>DLC2.3_3.out</th>
      <td>2791.460474</td>
      <td>5839.945621</td>
      <td>2213.542813</td>
    </tr>
  </tbody>
</table>
</div>



## Integrating outputs with a probability weighting (AEP, Damage, etc)

When running design load cases, not all windspeeds, or other environmental condition, occur with equal likelihood.  pCrunch provides a way to assign a probability to each output.  This probability can then weight a summation to compute annual energy production (AEP), or sum all Palmgren-Miner damages together.  Using a subset of the outputs is also a provided capability.

pCrunch provides a couple different ways to set the probabilities, either:
- Inflow wind speed using a Weibull or Rayleigh distribution for the site
- IEC turbine class with different average wind speeds that define a Weibull distribution
- Users can set the probability values directly.


```python
# Set probability based on wind speed channel name, Weibull distribution average of 7.5 m/s (shape factor input optional)
mycruncher.set_probability_distribution('WindVxi', 7.5, kind='weibull', weibull_k=2.0)

# Or Rayleigh distribution using the same distribution average of 7.5 m/s
mycruncher.set_probability_distribution('WindVxi', 7.5, kind='rayleigh')

# If you only want to use some of the outputs, but not all of them
mycruncher.set_probability_distribution('WindVxi', 7.5, kind='weibull', idx=[0,2])

# If you would rather specify the inflow wind speed directly to use in the probability distribution
mycruncher.set_probability_distribution([8,10,12], 7.5, kind='weibull')

# Can also set the probability based on IEC turbine class, again using a channel name of user input of wind speeds
mycruncher.set_probability_turbine_class('WindVxi', 2)
mycruncher.set_probability_turbine_class([8,10,12], 2)

# A savvy user can set the probability values directly (they will be rescaled to sum to one no matter what)
mycruncher.prob = np.array([0.1, 0.5, 0.4])
```

Once the probabilities are set, the user can use them to calculate AEP or total fatigue accumulation across the scenarios represented by each output.  For the AEP calculation, the user must specify the channel name.  Additional loss factors or restriction to certain indices are optional inputs.


```python
# Probability weighted and unweighted AEP values are returned
mycruncher.compute_aep('GenPwr')
```




    (39945405528.0, 39675356190.000015)




```python
# Or with loss factors and restricted by select outputs
mycruncher.compute_aep('GenPwr', loss_factor=0.15, idx=[0,2])
```




    (33707934777.6, 33486451832.250023)




```python
# Damage calculation does not require a channel name, as it uses the previously computed case-specific and channel-specific values.
dels_tot, dams_tot = mycruncher_mc.compute_total_fatigue()
```


```python
dels_tot
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RootMc1</th>
      <th>RootMc2</th>
      <th>RootMc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Weighted</th>
      <td>2702.788217</td>
      <td>5424.808463</td>
      <td>2142.025341</td>
    </tr>
    <tr>
      <th>Unweighted</th>
      <td>2702.788217</td>
      <td>5424.808463</td>
      <td>2142.025341</td>
    </tr>
  </tbody>
</table>
</div>




```python
dams_tot
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RootMc1</th>
      <th>RootMc2</th>
      <th>RootMc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Weighted</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Unweighted</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select indices are also available to restrict the summation
dels_tot, dams_tot = mycruncher_mc.compute_total_fatigue(idx=[0,2])
```

## Other Batch Procressing

The Crunch class provides batch extensions of nearly all of the operations offered in the AeroelasticOutputs class.  This includes the add channel or drop channel utilities and all statistical functions.  For the statistics, unlike the AeroelasticOutput class, these batch versions are functions, not data properties.  The result is returned as a list, with each index corresponding to the output list.  Many of these statistics also vary by channel, so there are likely to be nested lists.  Also, some are unavailable in "lean / streaming" mode.


```python
# Adding channel
mycruncher.calculate_channel('LSSGagMya + LSSGagMza', 'Test')

# Adding Load Roses
lr = {'TwrBs': ['TwrBsFxt', 'TwrBsFyt']}
mycruncher.add_load_rose(lr, nsec=6)

# Dropping channels by string wildcard
mycruncher.drop_channel('Fair*')
mycruncher.drop_channel('Anch*')
mycruncher.drop_channel('Spn*')
mycruncher.drop_channel('Root*')
mycruncher.drop_channel('Wave*')
mycruncher.drop_channel('Ptfm*')
mycruncher.drop_channel('Tw*')
mycruncher.drop_channel('Yaw*')
```

    Added channel, TwrBs0
    Added channel, TwrBs60
    Added channel, TwrBs120
    Added channel, TwrBs180
    Added channel, TwrBs240
    Added channel, TwrBs300
    Added channel, TwrBs0
    Added channel, TwrBs60
    Added channel, TwrBs120
    Added channel, TwrBs180
    Added channel, TwrBs240
    Added channel, TwrBs300
    Added channel, TwrBs0
    Added channel, TwrBs60
    Added channel, TwrBs120
    Added channel, TwrBs180
    Added channel, TwrBs240
    Added channel, TwrBs300



```python
# Indices to the minimum value for each channel
mycruncher.idxmins()
```




    array([[  0, 448,   0,   0, 401, 401,   0,   0,   0,   0, 258, 798, 798,
            490, 262, 505, 284, 486, 247, 473, 788, 653, 552, 209, 553, 222,
              0, 522, 286, 137, 406, 507, 489, 798, 453, 555, 406, 492],
           [  0, 450,   0,   0, 401, 401,   0,   0,   0,   0, 373, 798, 798,
            490, 224, 507, 412, 487, 249, 474, 789, 656, 617, 720, 785,  18,
              0, 522, 288, 260, 406, 508, 489, 798, 453, 555, 406, 493],
           [  0, 450,   0,   0, 401, 401,   0,   0,   0,   0, 260, 798, 798,
            490, 431, 505, 412, 486, 249, 474, 788, 654, 617, 721, 553,  19,
              0, 522, 288, 260, 406, 506, 488, 798, 453, 555, 406, 492]])




```python
# Indices to the maximum value for each channel
mycruncher.idxmaxs()
```




    array([[800, 502,   0,   0, 313, 315,   0, 630, 630, 630, 487, 436, 438,
            364, 677, 257, 465, 234, 563, 284, 512, 555, 403, 298, 655, 551,
              0, 232, 109, 314, 289, 567, 515, 448, 292, 231, 289, 555],
           [800, 505,   0,   0, 307, 313,   0, 630, 630, 630, 260, 438, 438,
            363, 683, 259, 466, 235, 565, 285, 511, 783, 403, 552, 657, 618,
              0, 407, 114, 316, 288, 561, 515, 449, 291, 407, 289, 554],
           [800, 505,   0,   0, 307, 311,   0, 630, 630, 630, 486, 438, 438,
            600, 686, 262, 466, 238, 564, 288, 511, 782, 403, 552, 655, 617,
              0, 407, 113, 316, 288, 590, 515, 449, 291, 407, 289, 553]])




```python
# Minimum value of each channel
mycruncher.minima()
```




    array([[ 4.00000000e+01,  8.20000000e+00,  0.00000000e+00,
             0.00000000e+00, -8.12000000e-01, -5.42000000e-01,
             0.00000000e+00, -7.48000000e-01, -7.23000000e-01,
             0.00000000e+00, -8.32000000e-01,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  7.56000000e-02,
            -3.20000000e-02, -3.14000000e+00, -2.13000000e-02,
            -3.07000000e-01, -2.09000000e+00, -9.38000000e-01,
             0.00000000e+00, -3.08000000e+00, -8.94000000e-01,
             0.00000000e+00, -3.44000000e+00, -1.22000000e+00,
             0.00000000e+00,  9.61000000e+00,  9.39000000e+00,
             9.52000000e+00, -2.01000000e+00, -1.32000000e-01,
            -4.79000000e-01, -8.20000000e-01, -4.85000000e-02,
             0.00000000e+00,  1.95000000e+01, -1.29000000e-01,
            -1.01000000e+00, -1.07000000e-01, -5.44000000e+00,
            -1.29000000e+00, -3.51000000e-01, -1.92000000e-02,
            -3.52000000e-01, -1.27000000e+02, -2.12000000e+02,
             1.13000000e+02, -3.77000000e+03, -5.17000000e+03,
            -8.00000000e+01, -1.46000000e+02, -2.05000000e+02,
            -1.68000000e+02, -4.04000000e+03, -6.63000000e+03,
            -6.75000000e+01, -1.06000000e+02, -2.08000000e+02,
            -1.14000000e+02, -3.77000000e+03, -5.69000000e+03,
            -6.67000000e+01, -5.72000000e+02, -1.11000000e+03,
            -5.46000000e+01, -9.23000000e+02, -1.60000000e+03,
            -5.01000000e+01, -5.99000000e+02, -1.93000000e+03,
            -4.83000000e+01, -3.53000000e+02, -1.14000000e+03,
            -1.10000000e+03, -1.16000000e+03, -3.84000000e+03,
            -3.36000000e+03, -1.20000000e+03, -6.53000000e+01,
            -3.62000000e+03, -1.31000000e+03, -5.21000000e+03,
            -3.36000000e+03, -1.68000000e+03, -1.37000000e+02,
            -7.16000000e+03, -4.94000000e+03, -1.35000000e+05,
            -3.36000000e+03, -2.73000000e+03, -6.75000000e+04,
            -3.36000000e+03,  2.00000000e+02,  7.79000000e+01,
             0.00000000e+00,  0.00000000e+00,  2.34000000e+02,
             7.28000000e+01,  0.00000000e+00,  0.00000000e+00,
             3.00000000e+02,  5.97000000e+01,  0.00000000e+00,
             0.00000000e+00,  3.70000000e+02,  4.51000000e+01,
             0.00000000e+00,  0.00000000e+00,  3.66000000e+02,
             4.50000000e+01,  0.00000000e+00,  0.00000000e+00,
             3.01000000e+02,  5.98000000e+01,  0.00000000e+00,
             0.00000000e+00,  2.34000000e+02,  7.31000000e+01,
             0.00000000e+00,  0.00000000e+00,  2.00000000e+02,
             7.77000000e+01,  0.00000000e+00,  0.00000000e+00,
            -2.25000000e-02, -2.15000000e-01, -6.15000000e-01,
            -2.74000000e-02,  4.59805830e+02,  6.62020574e+02,
             1.87380339e+03],
           [ 4.00000000e+01,  8.19700000e+00,  0.00000000e+00,
             0.00000000e+00, -1.05500000e+00, -5.43800000e-01,
             0.00000000e+00, -5.05900000e-01, -3.19500000e-01,
             0.00000000e+00, -3.15600000e-01,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.06600000e-01,
            -2.88300000e-02, -2.83500000e+00, -2.33700000e-02,
            -2.56000000e-01, -2.20200000e+00, -9.36000000e-01,
             0.00000000e+00, -3.28700000e+00, -9.15600000e-01,
             0.00000000e+00, -3.54500000e+00, -1.21300000e+00,
             0.00000000e+00,  9.44200000e+00,  9.32700000e+00,
             9.36200000e+00, -2.36600000e+00, -1.64600000e-01,
            -5.90300000e-01, -9.16400000e-01, -5.22300000e-02,
             0.00000000e+00,  1.66100000e+01, -1.46700000e-01,
            -1.19200000e+00, -1.46400000e-01, -6.83600000e+00,
            -1.41600000e+00, -4.08900000e-01, -1.87900000e-02,
            -2.33100000e-01, -1.41800000e+02, -2.09400000e+02,
             1.14400000e+02, -3.69900000e+03, -5.61300000e+03,
            -8.10700000e+01, -1.58600000e+02, -2.04500000e+02,
            -1.74600000e+02, -4.15000000e+03, -7.08600000e+03,
            -6.97200000e+01, -1.07500000e+02, -2.07400000e+02,
            -1.08200000e+02, -3.81500000e+03, -5.79000000e+03,
            -6.83400000e+01, -5.70200000e+02, -1.16000000e+03,
            -5.60000000e+01, -9.29000000e+02, -1.72600000e+03,
            -5.14700000e+01, -6.06300000e+02, -1.98300000e+03,
            -5.02600000e+01, -4.06700000e+02, -1.12800000e+03,
            -1.09700000e+03, -1.17700000e+03, -3.96900000e+03,
            -3.72900000e+03, -1.32700000e+03, -6.11600000e+01,
            -3.60000000e+03, -1.33600000e+03, -5.76900000e+03,
            -3.68200000e+03, -1.92200000e+03, -1.29600000e+02,
            -7.12000000e+03, -6.76100000e+03, -1.51400000e+05,
            -3.68100000e+03, -3.54200000e+03, -7.52000000e+04,
            -3.68200000e+03,  1.98800000e+02,  7.66900000e+01,
             0.00000000e+00,  0.00000000e+00,  2.33100000e+02,
             7.21800000e+01,  0.00000000e+00,  0.00000000e+00,
             2.91900000e+02,  5.95600000e+01,  0.00000000e+00,
             0.00000000e+00,  3.46800000e+02,  4.43400000e+01,
             0.00000000e+00,  0.00000000e+00,  3.43300000e+02,
             4.42600000e+01,  0.00000000e+00,  0.00000000e+00,
             2.92400000e+02,  5.96200000e+01,  0.00000000e+00,
             0.00000000e+00,  2.33100000e+02,  7.25200000e+01,
             0.00000000e+00,  0.00000000e+00,  1.98500000e+02,
             7.65700000e+01,  0.00000000e+00,  0.00000000e+00,
            -2.02100000e-02, -2.19300000e-01, -6.87500000e-01,
            -2.77800000e-02,  2.77648587e+02,  8.44238852e+02,
             2.09476388e+03],
           [ 4.00000000e+01,  8.19700000e+00,  0.00000000e+00,
             0.00000000e+00, -1.13500000e+00, -5.40300000e-01,
             0.00000000e+00, -5.15600000e-01, -2.48200000e-01,
             0.00000000e+00, -1.83600000e-01,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  7.13700000e-01,
            -2.93900000e-02, -2.90500000e+00, -2.32000000e-02,
            -2.46200000e-01, -2.20700000e+00, -9.38800000e-01,
             0.00000000e+00, -3.26500000e+00, -9.06200000e-01,
             0.00000000e+00, -3.59400000e+00, -1.21400000e+00,
             0.00000000e+00,  9.49300000e+00,  9.42900000e+00,
             9.50200000e+00, -2.18800000e+00, -1.65800000e-01,
            -4.39900000e-01, -8.25200000e-01, -5.21200000e-02,
             0.00000000e+00,  1.66200000e+01, -1.37400000e-01,
            -1.30300000e+00, -1.56000000e-01, -5.86500000e+00,
            -1.49400000e+00, -4.19300000e-01, -1.91400000e-02,
            -1.90000000e-01, -1.34600000e+02, -2.07900000e+02,
             1.20900000e+02, -3.71400000e+03, -5.44900000e+03,
            -8.25400000e+01, -1.58500000e+02, -2.03000000e+02,
            -1.71800000e+02, -4.04300000e+03, -7.07900000e+03,
            -7.03800000e+01, -1.14600000e+02, -2.04800000e+02,
            -1.10000000e+02, -3.82100000e+03, -5.99400000e+03,
            -6.88700000e+01, -5.66500000e+02, -1.17600000e+03,
            -5.65000000e+01, -9.31000000e+02, -1.69700000e+03,
            -5.22700000e+01, -6.17600000e+02, -2.00000000e+03,
            -5.01400000e+01, -3.73300000e+02, -1.11700000e+03,
            -1.09200000e+03, -1.18200000e+03, -4.00800000e+03,
            -3.67600000e+03, -1.20500000e+03, -6.56300000e+01,
            -3.57700000e+03, -1.36000000e+03, -5.50400000e+03,
            -3.65600000e+03, -1.68700000e+03, -1.33100000e+02,
            -7.07700000e+03, -6.94200000e+03, -1.35300000e+05,
            -3.65500000e+03, -3.65400000e+03, -6.79500000e+04,
            -3.65600000e+03,  1.98700000e+02,  7.66700000e+01,
             0.00000000e+00,  0.00000000e+00,  2.32700000e+02,
             7.21500000e+01,  0.00000000e+00,  0.00000000e+00,
             2.95000000e+02,  5.96400000e+01,  0.00000000e+00,
             0.00000000e+00,  3.50600000e+02,  4.47600000e+01,
             0.00000000e+00,  0.00000000e+00,  3.46900000e+02,
             4.47000000e+01,  0.00000000e+00,  0.00000000e+00,
             2.95800000e+02,  5.97400000e+01,  0.00000000e+00,
             0.00000000e+00,  2.32700000e+02,  7.25000000e+01,
             0.00000000e+00,  0.00000000e+00,  1.98400000e+02,
             7.65300000e+01,  0.00000000e+00,  0.00000000e+00,
            -2.06100000e-02, -2.26500000e-01, -6.20100000e-01,
            -2.78900000e-02,  3.47604352e+02,  6.24889950e+02,
             1.96820440e+03]])




```python
# Maximum value of each channel
mycruncher.maxima()
```




    array([[ 8.00000000e+01,  1.27000000e+01,  0.00000000e+00,
             0.00000000e+00,  1.11000000e+00,  8.75000000e-01,
             0.00000000e+00,  6.70000000e-01,  6.80000000e-01,
             0.00000000e+00,  5.37000000e-01,  2.79000000e+03,
             2.73000000e+01,  0.00000000e+00,  9.00000000e+01,
             9.00000000e+01,  9.00000000e+01,  3.60000000e+02,
             1.11000000e+01,  1.09000000e+03,  6.38000000e-03,
             1.29000000e+00,  4.91000000e+00,  1.23000000e+00,
             0.00000000e+00,  4.58000000e+00,  1.57000000e+00,
             0.00000000e+00,  4.72000000e+00,  2.10000000e+00,
             0.00000000e+00,  6.55000000e+01,  6.54000000e+01,
             6.54000000e+01,  2.19000000e+00,  1.76000000e-01,
             5.19000000e-01,  6.94000000e-01,  2.72000000e-02,
             0.00000000e+00,  3.72000000e+01,  3.93000000e-02,
             6.39000000e-01,  1.85000000e-01,  5.02000000e+00,
             3.11000000e-01,  2.67000000e-01,  1.03000000e-02,
             3.83000000e-01,  2.50000000e+02,  1.70000000e+02,
             6.75000000e+02,  4.70000000e+03,  8.87000000e+03,
             5.82000000e+01,  2.32000000e+02,  1.84000000e+02,
             6.37000000e+02,  4.50000000e+03,  8.19000000e+03,
             5.95000000e+01,  2.38000000e+02,  1.56000000e+02,
             6.23000000e+02,  4.59000000e+03,  8.55000000e+03,
             6.67000000e+01,  1.17000000e+03,  2.39000000e+03,
             1.25000000e+01,  1.51000000e+03,  2.25000000e+03,
             1.84000000e+01,  8.08000000e+02,  2.28000000e+03,
             1.42000000e+01,  7.59000000e+02,  1.09000000e+03,
             1.13000000e+03,  2.65000000e+03,  2.95000000e+03,
             3.59000000e+03,  1.09000000e+03,  4.76000000e+01,
            -3.27000000e+03,  2.69000000e+03,  2.09000000e+03,
             1.01000000e+03,  1.51000000e+03,  5.36000000e+01,
            -6.61000000e+03,  7.22000000e+03,  1.17000000e+05,
             1.00000000e+03,  4.75000000e+03,  5.68000000e+04,
             1.01000000e+03,  2.29000000e+02,  8.37000000e+01,
             0.00000000e+00,  0.00000000e+00,  2.57000000e+02,
             7.61000000e+01,  0.00000000e+00,  0.00000000e+00,
             3.65000000e+02,  6.57000000e+01,  0.00000000e+00,
             0.00000000e+00,  6.16000000e+02,  5.85000000e+01,
             2.90000000e+02,  0.00000000e+00,  6.16000000e+02,
             5.88000000e+01,  2.91000000e+02,  0.00000000e+00,
             3.64000000e+02,  6.56000000e+01,  0.00000000e+00,
             0.00000000e+00,  2.55000000e+02,  7.61000000e+01,
             0.00000000e+00,  0.00000000e+00,  2.30000000e+02,
             8.38000000e+01,  0.00000000e+00,  0.00000000e+00,
             8.83000000e+00,  4.66000000e-01,  1.13000000e+00,
             6.24000000e-02,  9.13416759e+03,  9.09380137e+03,
             8.93835680e+03],
           [ 8.00000000e+01,  1.27200000e+01,  0.00000000e+00,
             0.00000000e+00,  8.63300000e-01,  4.58900000e-01,
             0.00000000e+00,  5.32900000e-01,  3.01700000e-01,
             0.00000000e+00,  2.94900000e-01,  2.86600000e+03,
             2.78000000e+01,  0.00000000e+00,  9.00000000e+01,
             9.00000000e+01,  9.00000000e+01,  3.59500000e+02,
             1.12800000e+01,  1.10300000e+03,  6.03800000e-03,
             1.41800000e+00,  4.85200000e+00,  1.15200000e+00,
             0.00000000e+00,  4.70100000e+00,  1.65800000e+00,
             0.00000000e+00,  4.52600000e+00,  2.18300000e+00,
             0.00000000e+00,  6.54900000e+01,  6.53600000e+01,
             6.54300000e+01,  2.47200000e+00,  1.51900000e-01,
             4.82900000e-01,  8.36800000e-01,  3.55900000e-02,
             0.00000000e+00,  3.82300000e+01,  5.32600000e-02,
             7.31100000e-01,  2.19000000e-01,  6.28400000e+00,
             2.58100000e-01,  3.56100000e-01,  1.42900000e-02,
             3.23600000e-01,  2.41200000e+02,  1.65700000e+02,
             6.87500000e+02,  4.63900000e+03,  8.68000000e+03,
             6.28700000e+01,  2.34500000e+02,  1.87200000e+02,
             6.50900000e+02,  4.51000000e+03,  8.39800000e+03,
             5.47100000e+01,  2.31700000e+02,  1.55300000e+02,
             6.39300000e+02,  4.56700000e+03,  8.12600000e+03,
             6.14600000e+01,  1.25700000e+03,  2.37400000e+03,
             1.55700000e+01,  1.55500000e+03,  2.29500000e+03,
             1.25800000e+01,  8.09000000e+02,  2.20600000e+03,
             1.52900000e+01,  7.86900000e+02,  1.08000000e+03,
             1.11700000e+03,  2.70100000e+03,  3.09400000e+03,
             3.93000000e+03,  1.25700000e+03,  6.39200000e+01,
            -3.24200000e+03,  2.76800000e+03,  2.32700000e+03,
             9.51600000e+02,  1.82700000e+03,  7.76300000e+01,
            -6.54600000e+03,  7.64500000e+03,  1.40800000e+05,
             9.50800000e+02,  5.13200000e+03,  6.84600000e+04,
             9.52300000e+02,  2.35600000e+02,  8.39700000e+01,
             0.00000000e+00,  0.00000000e+00,  2.61400000e+02,
             7.62900000e+01,  0.00000000e+00,  0.00000000e+00,
             3.65300000e+02,  6.65300000e+01,  0.00000000e+00,
             0.00000000e+00,  6.30700000e+02,  6.04300000e+01,
             3.11100000e+02,  0.00000000e+00,  6.32100000e+02,
             6.07800000e+01,  3.13000000e+02,  0.00000000e+00,
             3.64300000e+02,  6.64900000e+01,  0.00000000e+00,
             0.00000000e+00,  2.59400000e+02,  7.62600000e+01,
             0.00000000e+00,  0.00000000e+00,  2.36400000e+02,
             8.40200000e+01,  0.00000000e+00,  0.00000000e+00,
             8.98800000e+00,  4.80100000e-01,  1.17100000e+00,
             6.36500000e-02,  9.07945230e+03,  9.29242629e+03,
             8.86344339e+03],
           [ 8.00000000e+01,  1.27200000e+01,  0.00000000e+00,
             0.00000000e+00,  1.12200000e+00,  3.93700000e-01,
             0.00000000e+00,  5.56300000e-01,  2.99000000e-01,
             0.00000000e+00,  3.11200000e-01,  2.76500000e+03,
             2.71400000e+01,  0.00000000e+00,  9.00000000e+01,
             9.00000000e+01,  9.00000000e+01,  3.58700000e+02,
             1.13300000e+01,  1.10900000e+03,  5.58900000e-03,
             1.49600000e+00,  4.61600000e+00,  1.23500000e+00,
             0.00000000e+00,  4.54500000e+00,  1.61600000e+00,
             0.00000000e+00,  4.34900000e+00,  2.16200000e+00,
             0.00000000e+00,  6.55000000e+01,  6.53800000e+01,
             6.54200000e+01,  2.20000000e+00,  1.76400000e-01,
             4.48600000e-01,  7.61700000e-01,  3.66900000e-02,
             0.00000000e+00,  3.77900000e+01,  5.59000000e-02,
             1.03000000e+00,  2.15700000e-01,  5.65000000e+00,
             2.49800000e-01,  2.98300000e-01,  1.25800000e-02,
             2.63700000e-01,  2.30200000e+02,  1.66600000e+02,
             6.94500000e+02,  4.59700000e+03,  8.19800000e+03,
             5.66800000e+01,  2.24100000e+02,  1.85100000e+02,
             6.39200000e+02,  4.48200000e+03,  8.06600000e+03,
             5.62800000e+01,  2.28100000e+02,  1.54000000e+02,
             6.39300000e+02,  4.51000000e+03,  8.04100000e+03,
             5.90000000e+01,  1.20800000e+03,  2.26400000e+03,
             1.32400000e+01,  1.56200000e+03,  2.22300000e+03,
             1.29200000e+01,  8.41900000e+02,  2.13000000e+03,
             1.68100000e+01,  7.46200000e+02,  1.07100000e+03,
             1.10400000e+03,  2.63800000e+03,  3.02100000e+03,
             3.81100000e+03,  1.14700000e+03,  6.56000000e+01,
            -3.27600000e+03,  2.71700000e+03,  1.97200000e+03,
             8.80400000e+02,  1.66200000e+03,  7.91300000e+01,
            -6.61400000e+03,  7.63600000e+03,  1.28200000e+05,
             8.80500000e+02,  5.12400000e+03,  6.23300000e+04,
             8.80400000e+02,  2.35800000e+02,  8.39100000e+01,
             0.00000000e+00,  0.00000000e+00,  2.61600000e+02,
             7.62900000e+01,  0.00000000e+00,  0.00000000e+00,
             3.67500000e+02,  6.63400000e+01,  0.00000000e+00,
             0.00000000e+00,  6.17600000e+02,  6.02500000e+01,
             2.92600000e+02,  0.00000000e+00,  6.18500000e+02,
             6.06300000e+01,  2.93900000e+02,  0.00000000e+00,
             3.66200000e+02,  6.62600000e+01,  0.00000000e+00,
             0.00000000e+00,  2.59600000e+02,  7.62700000e+01,
             0.00000000e+00,  0.00000000e+00,  2.36700000e+02,
             8.39500000e+01,  0.00000000e+00,  0.00000000e+00,
             9.01200000e+00,  4.63400000e-01,  1.11000000e+00,
             6.21700000e-02,  8.98622385e+03,  9.11128469e+03,
             8.78134426e+03]])




```python
# Maximum value of absolute values of each channel
mycruncher.absmaxima()
```




    array([[8.000e+01, 1.270e+01, 0.000e+00, 0.000e+00, 2.790e+03, 2.730e+01,
            0.000e+00, 9.000e+01, 9.000e+01, 9.000e+01, 3.600e+02, 1.110e+01,
            1.090e+03, 2.130e-02, 1.290e+00, 4.910e+00, 1.230e+00, 4.580e+00,
            1.570e+00, 4.720e+00, 2.100e+00, 2.190e+00, 1.760e-01, 5.190e-01,
            8.200e-01, 4.850e-02, 0.000e+00, 7.590e+02, 1.140e+03, 1.130e+03,
            2.650e+03, 3.840e+03, 3.590e+03, 8.830e+00, 4.660e-01, 1.130e+00,
            6.240e-02, 4.930e+03],
           [8.000e+01, 1.272e+01, 0.000e+00, 0.000e+00, 2.866e+03, 2.780e+01,
            0.000e+00, 9.000e+01, 9.000e+01, 9.000e+01, 3.595e+02, 1.128e+01,
            1.103e+03, 2.337e-02, 1.418e+00, 4.852e+00, 1.152e+00, 4.701e+00,
            1.658e+00, 4.526e+00, 2.183e+00, 2.472e+00, 1.646e-01, 5.903e-01,
            9.164e-01, 5.223e-02, 0.000e+00, 7.869e+02, 1.128e+03, 1.117e+03,
            2.701e+03, 3.969e+03, 3.930e+03, 8.988e+00, 4.801e-01, 1.171e+00,
            6.365e-02, 5.283e+03],
           [8.000e+01, 1.272e+01, 0.000e+00, 0.000e+00, 2.765e+03, 2.714e+01,
            0.000e+00, 9.000e+01, 9.000e+01, 9.000e+01, 3.587e+02, 1.133e+01,
            1.109e+03, 2.320e-02, 1.496e+00, 4.616e+00, 1.235e+00, 4.545e+00,
            1.616e+00, 4.349e+00, 2.162e+00, 2.200e+00, 1.764e-01, 4.486e-01,
            8.252e-01, 5.212e-02, 0.000e+00, 7.462e+02, 1.117e+03, 1.104e+03,
            2.638e+03, 4.008e+03, 3.811e+03, 9.012e+00, 4.634e-01, 1.110e+00,
            6.217e-02, 5.112e+03]])




```python
# The range of data values (max - min)
mycruncher.ranges()
```




    array([[4.00000000e+01, 4.50000000e+00, 0.00000000e+00, 0.00000000e+00,
            1.92200000e+00, 1.41700000e+00, 0.00000000e+00, 1.41800000e+00,
            1.40300000e+00, 0.00000000e+00, 1.36900000e+00, 2.79000000e+03,
            2.73000000e+01, 0.00000000e+00, 9.00000000e+01, 9.00000000e+01,
            9.00000000e+01, 3.59924400e+02, 1.11320000e+01, 1.09314000e+03,
            2.76800000e-02, 1.59700000e+00, 7.00000000e+00, 2.16800000e+00,
            0.00000000e+00, 7.66000000e+00, 2.46400000e+00, 0.00000000e+00,
            8.16000000e+00, 3.32000000e+00, 0.00000000e+00, 5.58900000e+01,
            5.60100000e+01, 5.58800000e+01, 4.20000000e+00, 3.08000000e-01,
            9.98000000e-01, 1.51400000e+00, 7.57000000e-02, 0.00000000e+00,
            1.77000000e+01, 1.68300000e-01, 1.64900000e+00, 2.92000000e-01,
            1.04600000e+01, 1.60100000e+00, 6.18000000e-01, 2.95000000e-02,
            7.35000000e-01, 3.77000000e+02, 3.82000000e+02, 5.62000000e+02,
            8.47000000e+03, 1.40400000e+04, 1.38200000e+02, 3.78000000e+02,
            3.89000000e+02, 8.05000000e+02, 8.54000000e+03, 1.48200000e+04,
            1.27000000e+02, 3.44000000e+02, 3.64000000e+02, 7.37000000e+02,
            8.36000000e+03, 1.42400000e+04, 1.33400000e+02, 1.74200000e+03,
            3.50000000e+03, 6.71000000e+01, 2.43300000e+03, 3.85000000e+03,
            6.85000000e+01, 1.40700000e+03, 4.21000000e+03, 6.25000000e+01,
            1.11200000e+03, 2.23000000e+03, 2.23000000e+03, 3.81000000e+03,
            6.79000000e+03, 6.95000000e+03, 2.29000000e+03, 1.12900000e+02,
            3.50000000e+02, 4.00000000e+03, 7.30000000e+03, 4.37000000e+03,
            3.19000000e+03, 1.90600000e+02, 5.50000000e+02, 1.21600000e+04,
            2.52000000e+05, 4.36000000e+03, 7.48000000e+03, 1.24300000e+05,
            4.37000000e+03, 2.90000000e+01, 5.80000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.30000000e+01, 3.30000000e+00, 0.00000000e+00,
            0.00000000e+00, 6.50000000e+01, 6.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.46000000e+02, 1.34000000e+01, 2.90000000e+02,
            0.00000000e+00, 2.50000000e+02, 1.38000000e+01, 2.91000000e+02,
            0.00000000e+00, 6.30000000e+01, 5.80000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.10000000e+01, 3.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 3.00000000e+01, 6.10000000e+00, 0.00000000e+00,
            0.00000000e+00, 8.85250000e+00, 6.81000000e-01, 1.74500000e+00,
            8.98000000e-02, 8.67436176e+03, 8.43178079e+03, 7.06455341e+03],
           [4.00000000e+01, 4.52300000e+00, 0.00000000e+00, 0.00000000e+00,
            1.91830000e+00, 1.00270000e+00, 0.00000000e+00, 1.03880000e+00,
            6.21200000e-01, 0.00000000e+00, 6.10500000e-01, 2.86600000e+03,
            2.78000000e+01, 0.00000000e+00, 9.00000000e+01, 9.00000000e+01,
            9.00000000e+01, 3.59393400e+02, 1.13088300e+01, 1.10583500e+03,
            2.94080000e-02, 1.67400000e+00, 7.05400000e+00, 2.08800000e+00,
            0.00000000e+00, 7.98800000e+00, 2.57360000e+00, 0.00000000e+00,
            8.07100000e+00, 3.39600000e+00, 0.00000000e+00, 5.60480000e+01,
            5.60330000e+01, 5.60680000e+01, 4.83800000e+00, 3.16500000e-01,
            1.07320000e+00, 1.75320000e+00, 8.78200000e-02, 0.00000000e+00,
            2.16200000e+01, 1.99960000e-01, 1.92310000e+00, 3.65400000e-01,
            1.31200000e+01, 1.67410000e+00, 7.65000000e-01, 3.30800000e-02,
            5.56700000e-01, 3.83000000e+02, 3.75100000e+02, 5.73100000e+02,
            8.33800000e+03, 1.42930000e+04, 1.43940000e+02, 3.93100000e+02,
            3.91700000e+02, 8.25500000e+02, 8.66000000e+03, 1.54840000e+04,
            1.24430000e+02, 3.39200000e+02, 3.62700000e+02, 7.47500000e+02,
            8.38200000e+03, 1.39160000e+04, 1.29800000e+02, 1.82720000e+03,
            3.53400000e+03, 7.15700000e+01, 2.48400000e+03, 4.02100000e+03,
            6.40500000e+01, 1.41530000e+03, 4.18900000e+03, 6.55500000e+01,
            1.19360000e+03, 2.20800000e+03, 2.21400000e+03, 3.87800000e+03,
            7.06300000e+03, 7.65900000e+03, 2.58400000e+03, 1.25080000e+02,
            3.58000000e+02, 4.10400000e+03, 8.09600000e+03, 4.63360000e+03,
            3.74900000e+03, 2.07230000e+02, 5.74000000e+02, 1.44060000e+04,
            2.92200000e+05, 4.63180000e+03, 8.67400000e+03, 1.43660000e+05,
            4.63430000e+03, 3.68000000e+01, 7.28000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.83000000e+01, 4.11000000e+00, 0.00000000e+00,
            0.00000000e+00, 7.34000000e+01, 6.97000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.83900000e+02, 1.60900000e+01, 3.11100000e+02,
            0.00000000e+00, 2.88800000e+02, 1.65200000e+01, 3.13000000e+02,
            0.00000000e+00, 7.19000000e+01, 6.87000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.63000000e+01, 3.74000000e+00, 0.00000000e+00,
            0.00000000e+00, 3.79000000e+01, 7.45000000e+00, 0.00000000e+00,
            0.00000000e+00, 9.00821000e+00, 6.99400000e-01, 1.85850000e+00,
            9.14300000e-02, 8.80180372e+03, 8.44818744e+03, 6.76867951e+03],
           [4.00000000e+01, 4.52300000e+00, 0.00000000e+00, 0.00000000e+00,
            2.25700000e+00, 9.34000000e-01, 0.00000000e+00, 1.07190000e+00,
            5.47200000e-01, 0.00000000e+00, 4.94800000e-01, 2.76500000e+03,
            2.71400000e+01, 0.00000000e+00, 9.00000000e+01, 9.00000000e+01,
            9.00000000e+01, 3.57986300e+02, 1.13593900e+01, 1.11190500e+03,
            2.87890000e-02, 1.74220000e+00, 6.82300000e+00, 2.17380000e+00,
            0.00000000e+00, 7.81000000e+00, 2.52220000e+00, 0.00000000e+00,
            7.94300000e+00, 3.37600000e+00, 0.00000000e+00, 5.60070000e+01,
            5.59510000e+01, 5.59180000e+01, 4.38800000e+00, 3.42200000e-01,
            8.88500000e-01, 1.58690000e+00, 8.88100000e-02, 0.00000000e+00,
            2.11700000e+01, 1.93300000e-01, 2.33300000e+00, 3.71700000e-01,
            1.15150000e+01, 1.74380000e+00, 7.17600000e-01, 3.17200000e-02,
            4.53700000e-01, 3.64800000e+02, 3.74500000e+02, 5.73600000e+02,
            8.31100000e+03, 1.36470000e+04, 1.39220000e+02, 3.82600000e+02,
            3.88100000e+02, 8.11000000e+02, 8.52500000e+03, 1.51450000e+04,
            1.26660000e+02, 3.42700000e+02, 3.58800000e+02, 7.49300000e+02,
            8.33100000e+03, 1.40350000e+04, 1.27870000e+02, 1.77450000e+03,
            3.44000000e+03, 6.97400000e+01, 2.49300000e+03, 3.92000000e+03,
            6.51900000e+01, 1.45950000e+03, 4.13000000e+03, 6.69500000e+01,
            1.11950000e+03, 2.18800000e+03, 2.19600000e+03, 3.82000000e+03,
            7.02900000e+03, 7.48700000e+03, 2.35200000e+03, 1.31230000e+02,
            3.01000000e+02, 4.07700000e+03, 7.47600000e+03, 4.53640000e+03,
            3.34900000e+03, 2.12230000e+02, 4.63000000e+02, 1.45780000e+04,
            2.63500000e+05, 4.53550000e+03, 8.77800000e+03, 1.30280000e+05,
            4.53640000e+03, 3.71000000e+01, 7.24000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.89000000e+01, 4.14000000e+00, 0.00000000e+00,
            0.00000000e+00, 7.25000000e+01, 6.70000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.67000000e+02, 1.54900000e+01, 2.92600000e+02,
            0.00000000e+00, 2.71600000e+02, 1.59300000e+01, 2.93900000e+02,
            0.00000000e+00, 7.04000000e+01, 6.52000000e+00, 0.00000000e+00,
            0.00000000e+00, 2.69000000e+01, 3.77000000e+00, 0.00000000e+00,
            0.00000000e+00, 3.83000000e+01, 7.42000000e+00, 0.00000000e+00,
            0.00000000e+00, 9.03261000e+00, 6.89900000e-01, 1.73010000e+00,
            9.00600000e-02, 8.63861950e+03, 8.48639474e+03, 6.81313986e+03]])




```python
# Channel indices which vary in time
mycruncher.variable()
```




    array([[ 0,  1,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37],
           [ 0,  1,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37],
           [ 0,  1,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37]])




```python
# Channel indices which are constant in time
mycruncher.constant()
```




    array([[ 2,  3,  6, 26],
           [ 2,  3,  6, 26],
           [ 2,  3,  6, 26]])




```python
# Sum of channel values over time
mycruncher.sums()
```




    array([[ 4.80600000e+04,  7.58900000e+03,  0.00000000e+00,
             0.00000000e+00,  9.89530000e+05,  1.00830000e+04,
             0.00000000e+00,  2.55375000e+04,  2.55375000e+04,
             2.55375000e+04,  1.44128285e+05,  5.74874039e+03,
             5.57390046e+05, -4.14189357e-01,  2.00572446e+02,
             1.55066591e+03, -3.54332400e+01,  1.52463849e+03,
             9.82605120e+01,  1.48320427e+03, -2.38651510e+02,
            -1.95322500e+01, -1.30349040e+00, -4.35471540e+01,
             1.17890310e+02, -1.10558919e+01,  0.00000000e+00,
             2.62518510e+05,  1.95963330e+04,  1.82839544e+05,
             8.71292238e+05,  4.83505880e+05, -5.12910800e+04,
             4.01501454e+03,  1.58754217e+02,  3.98490080e+02,
             2.08073147e+01,  4.32214800e+05],
           [ 4.80600000e+04,  7.58870100e+03,  0.00000000e+00,
             0.00000000e+00,  1.02185900e+06,  1.03015900e+04,
             0.00000000e+00,  2.55375000e+04,  2.55375000e+04,
             2.55375000e+04,  1.46322747e+05,  5.81471523e+03,
             5.64074042e+05, -4.29132560e-01,  2.31461687e+02,
             1.58449982e+03, -1.92210633e+01,  1.55236547e+03,
             8.14886831e+01,  1.50997165e+03, -2.40795873e+02,
            -1.14118070e+01,  5.92582160e-01, -5.98549325e+01,
             1.18416339e+02, -1.16533660e+01,  0.00000000e+00,
             2.65750469e+05,  4.45438676e+04,  1.75280669e+05,
             8.91249956e+05,  4.59429342e+05, -7.62801380e+04,
             4.06345349e+03,  1.64121847e+02,  4.04393590e+02,
             2.12915326e+01,  3.83149204e+05],
           [ 4.80600000e+04,  7.58870100e+03,  0.00000000e+00,
             0.00000000e+00,  1.01158900e+06,  1.02322000e+04,
             0.00000000e+00,  2.55375000e+04,  2.55375000e+04,
             2.55375000e+04,  1.45494185e+05,  5.79919548e+03,
             5.62571542e+05, -4.70458914e-01,  2.49827552e+02,
             1.57556414e+03, -2.24069898e+01,  1.54333543e+03,
             8.94640447e+01,  1.50694153e+03, -2.42992410e+02,
            -1.76334419e+01,  1.15344480e+00, -4.68919519e+01,
             1.19248747e+02, -1.15551107e+01,  0.00000000e+00,
             2.65406862e+05,  3.66434530e+04,  1.79283001e+05,
             8.84936482e+05,  4.63947772e+05, -6.89581070e+04,
             4.05315083e+03,  1.62392546e+02,  4.03220058e+02,
             2.11374915e+01,  3.94989665e+05]])




```python
# Sum of channel values over time to the second power
mycruncher.sums_squared()
```




    array([[2.99066700e+06, 7.23318632e+04, 0.00000000e+00, 0.00000000e+00,
            2.45104090e+09, 2.53956520e+05, 0.00000000e+00, 1.99462425e+06,
            1.99462425e+06, 1.99462425e+06, 3.26158758e+07, 5.51322865e+04,
            5.18293747e+08, 1.52310134e-02, 3.39176067e+02, 6.81489156e+03,
            2.66962590e+02, 6.77778710e+03, 3.74944460e+02, 6.92880700e+03,
            5.28852071e+02, 8.93030853e+02, 2.49956593e+00, 5.45906269e+01,
            1.66439403e+02, 4.62125266e-01, 0.00000000e+00, 1.84694760e+08,
            4.01064942e+08, 5.08073920e+08, 2.49756762e+09, 2.46152297e+09,
            1.85959911e+09, 2.71241855e+04, 7.06899358e+01, 4.08180563e+02,
            1.37977617e+00, 3.49188164e+09],
           [2.99066700e+06, 7.23262441e+04, 0.00000000e+00, 0.00000000e+00,
            2.61382196e+09, 2.65084568e+05, 0.00000000e+00, 1.99462425e+06,
            1.99462425e+06, 1.99462425e+06, 3.34249703e+07, 5.64178526e+04,
            5.30965262e+08, 1.81228244e-02, 4.09763475e+02, 7.04301256e+03,
            2.74098853e+02, 7.07116127e+03, 3.85430977e+02, 7.19330845e+03,
            5.40419542e+02, 1.08371235e+03, 2.95931019e+00, 6.45379099e+01,
            2.03904667e+02, 5.11556206e-01, 0.00000000e+00, 1.99404699e+08,
            4.14198717e+08, 4.89508882e+08, 2.60675743e+09, 2.42516652e+09,
            2.03201446e+09, 2.77923977e+04, 7.53970809e+01, 4.39973036e+02,
            1.44009614e+00, 3.47276714e+09],
           [2.99066700e+06, 7.23262441e+04, 0.00000000e+00, 0.00000000e+00,
            2.55846131e+09, 2.61390686e+05, 0.00000000e+00, 1.99462425e+06,
            1.99462425e+06, 1.99462425e+06, 3.31168578e+07, 5.61496845e+04,
            5.28448721e+08, 1.55653909e-02, 4.54157780e+02, 7.04486461e+03,
            2.77771540e+02, 7.00787799e+03, 3.80998288e+02, 7.16263392e+03,
            5.37869143e+02, 8.21733248e+02, 3.07988112e+00, 4.18266526e+01,
            1.69619926e+02, 5.11603552e-01, 0.00000000e+00, 1.91745474e+08,
            4.09224175e+08, 4.97844436e+08, 2.57841571e+09, 2.37670182e+09,
            1.90309777e+09, 2.76736531e+04, 7.40735980e+01, 4.22012404e+02,
            1.42446832e+00, 3.36627034e+09]])




```python
# Sum of channel values over time to the third power
mycruncher.sums_cubed()
```




    array([[ 1.92288060e+08,  6.94298326e+05,  0.00000000e+00,
             0.00000000e+00,  6.09512315e+12,  6.40731260e+06,
             0.00000000e+00,  1.65847398e+08,  1.65847398e+08,
             1.65847398e+08,  8.23177046e+09,  5.53178222e+05,
             5.04221025e+11, -1.55543032e-04,  3.65344809e+02,
             2.58163535e+04,  1.20947939e+01,  2.46568356e+04,
             2.05625057e+02,  2.45975817e+04,  7.10796755e+01,
             9.96182890e+01, -1.48520740e-02, -4.35589968e+00,
             1.53278176e+01, -1.34159296e-02,  0.00000000e+00,
             1.01753538e+11, -3.22542353e+10,  1.49847668e+11,
             5.84157948e+12,  1.43604884e+12,  5.65690138e+11,
             1.93195077e+05,  2.86981674e+01,  3.38936129e+02,
             7.67175294e-02,  3.45922125e+12],
           [ 1.92288060e+08,  6.94219888e+05,  0.00000000e+00,
             0.00000000e+00,  6.71230136e+12,  6.83293668e+06,
             0.00000000e+00,  1.65847398e+08,  1.65847398e+08,
             1.65847398e+08,  8.47749243e+09,  5.72455267e+05,
             5.22686792e+11, -2.15949030e-04,  4.91595938e+02,
             2.71626676e+04,  1.59472299e+01,  2.60249519e+04,
             2.30437505e+02,  2.57402611e+04,  1.09502185e+02,
             1.77166444e+02, -1.23906496e-02, -1.00046839e+01,
             3.95596489e+00, -1.53204480e-02,  0.00000000e+00,
             1.07198521e+11, -8.07906243e+09,  1.33418617e+11,
             6.22978579e+12,  1.37991105e+12,  2.11274187e+11,
             2.00370633e+05,  3.16235461e+01,  3.58246493e+02,
             8.18518941e-02,  3.47182961e+12],
           [ 1.92288060e+08,  6.94219888e+05,  0.00000000e+00,
             0.00000000e+00,  6.48782731e+12,  6.68524869e+06,
             0.00000000e+00,  1.65847398e+08,  1.65847398e+08,
             1.65847398e+08,  8.38161564e+09,  5.68591824e+05,
             5.19172832e+11, -2.03925755e-04,  5.81324273e+02,
             2.68583738e+04,  2.43883094e+01,  2.56430480e+04,
             2.16015743e+02,  2.55127871e+04,  8.86497887e+01,
             8.67920936e+01, -2.82224593e-02, -3.91787274e+00,
             4.61963408e+00, -1.47915829e-02,  0.00000000e+00,
             1.03774269e+11, -1.63234993e+10,  1.42749896e+11,
             6.11488921e+12,  1.49365291e+12,  3.20648189e+11,
             1.99215329e+05,  3.06624163e+01,  3.45734171e+02,
             8.03371253e-02,  3.37793176e+12]])




```python
# Sum of channel values over time to the fourth power
mycruncher.sums_fourth()
```




    array([[1.27193675e+10, 6.72058716e+06, 0.00000000e+00, 0.00000000e+00,
            1.52191817e+16, 1.61942556e+08, 0.00000000e+00, 1.41881502e+10,
            1.41881502e+10, 1.41881502e+10, 2.23468362e+12, 5.63005878e+06,
            4.97577410e+14, 2.75297219e-06, 4.38130767e+02, 1.06767943e+05,
            1.74151600e+02, 1.01872083e+05, 4.17986394e+02, 1.04873960e+05,
            9.00199729e+02, 2.22800798e+03, 2.07158246e-02, 7.26447975e+00,
            6.04853833e+01, 5.39994955e-04, 0.00000000e+00, 6.62801332e+13,
            3.18222414e+14, 4.30309506e+14, 1.45026648e+16, 1.32816255e+16,
            1.00332283e+16, 1.40368241e+06, 1.20486157e+01, 3.26802571e+02,
            4.46744190e-03, 3.85369085e+16],
           [1.27193675e+10, 6.71962967e+06, 0.00000000e+00, 0.00000000e+00,
            1.73077680e+16, 1.76438826e+08, 0.00000000e+00, 1.41881502e+10,
            1.41881502e+10, 1.41881502e+10, 2.30428942e+12, 5.89126197e+06,
            5.21876656e+14, 4.09519763e-06, 6.40737567e+02, 1.13515183e+05,
            1.71822081e+02, 1.10169337e+05, 4.85685686e+02, 1.11323508e+05,
            9.79741027e+02, 3.99063936e+03, 2.96466358e-02, 1.03558992e+01,
            9.78748918e+01, 7.05342834e-04, 0.00000000e+00, 7.18301735e+13,
            3.29207326e+14, 4.06090981e+14, 1.57982336e+16, 1.34969948e+16,
            1.21547740e+16, 1.47386976e+06, 1.37061298e+01, 3.54137028e+02,
            4.86815847e-03, 3.96254617e+16],
           [1.27193675e+10, 6.71962967e+06, 0.00000000e+00, 0.00000000e+00,
            1.64965054e+16, 1.71182977e+08, 0.00000000e+00, 1.41881502e+10,
            1.41881502e+10, 1.41881502e+10, 2.27616268e+12, 5.84004675e+06,
            5.17362253e+14, 3.78177162e-06, 8.00439475e+02, 1.11706745e+05,
            1.86324902e+02, 1.07836380e+05, 4.50472627e+02, 1.10040069e+05,
            9.54738092e+02, 2.50292316e+03, 3.37720417e-02, 4.30915266e+00,
            6.32818249e+01, 6.91903169e-04, 0.00000000e+00, 6.69559817e+13,
            3.24116649e+14, 4.16535980e+14, 1.54023817e+16, 1.27874548e+16,
            1.05403788e+16, 1.46350347e+06, 1.31312447e+01, 3.28398422e+02,
            4.74606571e-03, 3.72049225e+16]])




```python
# Second moment of the timeseries for each channel
mycruncher.second_moments()
```




    array([[1.33666667e+02, 5.37563724e-01, 0.00000000e+00, 0.00000000e+00,
            1.53384134e+06, 1.58591217e+02, 0.00000000e+00, 1.47370425e+03,
            1.47370425e+03, 1.47370425e+03, 8.34218479e+03, 1.73206485e+01,
            1.62826473e+05, 1.87476156e-05, 3.60739343e-01, 4.76022226e+00,
            3.31329783e-01, 4.83865350e+00, 4.53046962e-01, 5.22143747e+00,
            5.71470377e-01, 1.11430033e+00, 3.11790852e-03, 6.51974319e-02,
            1.86127885e-01, 3.86423328e-04, 0.00000000e+00, 1.23167724e+05,
            5.00106768e+05, 5.82195026e+05, 1.93484970e+06, 2.70869585e+06,
            2.31749656e+06, 8.73772149e+00, 4.89708356e-02, 2.62091685e-01,
            1.04777949e-03, 4.06824110e+06],
           [1.33666667e+02, 5.37621765e-01, 0.00000000e+00, 0.00000000e+00,
            1.63571375e+06, 1.65538992e+02, 0.00000000e+00, 1.47370425e+03,
            1.47370425e+03, 1.47370425e+03, 8.35886306e+03, 1.77365476e+01,
            1.66963034e+05, 2.23382251e-05, 4.28063595e-01, 4.87968906e+00,
            3.41619997e-01, 5.07193978e+00, 4.70837494e-01, 5.42677721e+00,
            5.84309253e-01, 1.35274627e+00, 3.69397228e-03, 7.49878085e-02,
            2.32707258e-01, 4.26987461e-04, 0.00000000e+00, 1.38871124e+05,
            5.14009511e+05, 5.63236811e+05, 2.01634071e+06, 2.69869134e+06,
            2.52777804e+06, 8.96204384e+00, 5.21462422e-02, 2.94395155e-01,
            1.09131321e-03, 4.10673170e+06],
           [1.33666667e+02, 5.37621765e-01, 0.00000000e+00, 0.00000000e+00,
            1.59914839e+06, 1.63148160e+02, 0.00000000e+00, 1.47370425e+03,
            1.47370425e+03, 1.47370425e+03, 8.35105509e+03, 1.76826861e+01,
            1.66459663e+05, 1.90874804e-05, 4.69710266e-01, 4.92601194e+00,
            3.45998417e-01, 5.03650401e+00, 4.63178539e-01, 5.40273005e+00,
            5.79468973e-01, 1.02539958e+00, 3.84297147e-03, 4.87909052e-02,
            1.89596489e-01, 4.30600735e-04, 0.00000000e+00, 1.29593504e+05,
            5.08798804e+05, 5.71431464e+05, 1.99843580e+06, 2.63168328e+06,
            2.36849084e+06, 8.94413264e+00, 5.13740049e-02, 2.73449573e-01,
            1.08198954e-03, 3.95941669e+06]])




```python
# Third moment of the timeseries for each channel
mycruncher.third_moments()
```




    array([[ 0.00000000e+00,  1.04578434e+00,  0.00000000e+00,
             0.00000000e+00,  3.94767725e+07,  1.54227646e+01,
             0.00000000e+00,  3.36895113e+04,  3.36895113e+04,
             3.36895113e+04, -5.20236650e+04, -5.19943844e+01,
            -4.73889823e+07, -1.64965164e-07,  1.69420113e-01,
            -2.67129717e+00,  5.91565473e-02, -3.74347926e+00,
             8.81354619e-02, -4.64587100e+00,  6.25981817e-01,
             2.05897979e-01, -3.31601828e-06,  5.35617821e-03,
            -6.62345831e-02,  1.88154176e-06,  0.00000000e+00,
            -2.92704786e+07, -7.69871917e+07, -2.23500517e+08,
            -3.08118295e+08, -3.33225423e+09,  1.15168683e+09,
            -1.61412350e+01, -1.07475623e-03, -9.11508028e-02,
            -3.40501759e-06, -2.42407639e+09],
           [ 0.00000000e+00,  1.04733005e+00,  0.00000000e+00,
             0.00000000e+00,  4.34894071e+07,  1.63269772e+01,
             0.00000000e+00,  3.36895113e+04,  3.36895113e+04,
             3.36895113e+04, -9.31343873e+04, -5.41397262e+01,
            -4.94188476e+07, -2.33542671e-07,  2.18511388e-01,
            -2.78803397e+00,  4.45158510e-02, -4.27741500e+00,
             1.42934253e-01, -5.25401120e+00,  6.90837849e-01,
             2.79001996e-01, -2.36678155e-05,  4.73746389e-03,
            -1.01499492e-01,  2.58879026e-06,  0.00000000e+00,
            -4.09098387e+07, -9.60108982e+07, -2.13668341e+08,
            -3.30594643e+08, -3.10962171e+09,  9.86796183e+08,
            -1.67955312e+01, -1.17565997e-03, -1.27318005e-01,
            -3.61927012e-06, -1.66830321e+09],
           [ 0.00000000e+00,  1.04733005e+00,  0.00000000e+00,
             0.00000000e+00,  2.66731465e+07,  9.28978962e+00,
             0.00000000e+00,  3.36895113e+04,  3.36895113e+04,
             3.36895113e+04, -7.96664510e+04, -5.37072130e+01,
            -4.90223915e+07, -2.20753852e-07,  2.55907360e-01,
            -3.14774696e+00,  5.95058583e-02, -4.25154440e+00,
             1.13091416e-01, -5.30044132e+00,  6.65957055e-01,
             1.76085656e-01, -5.18387234e-05,  3.87832060e-03,
            -8.22107277e-02,  3.17105444e-06,  0.00000000e+00,
            -3.56423594e+07, -9.03028987e+07, -2.16698524e+08,
            -3.37946545e+08, -2.90247866e+09,  1.01265819e+09,
            -1.66296266e+01, -1.29909239e-03, -1.08896152e-01,
            -3.73793431e-06, -1.76017867e+09]])




```python
# Fourth moment of the timeseries for each channel
mycruncher.fourth_moments()
```




    array([[3.21601332e+04, 3.44360404e+00, 0.00000000e+00, 0.00000000e+00,
            2.43097390e+12, 2.55095341e+04, 0.00000000e+00, 3.39569151e+06,
            3.39569151e+06, 3.39569151e+06, 1.58498572e+08, 5.15307667e+02,
            4.55459785e+10, 3.06556334e-09, 2.37642092e-01, 3.28922739e+01,
            2.23991186e-01, 3.73738647e+01, 4.37451084e-01, 4.61649792e+01,
            1.55761531e+00, 2.79764035e+00, 2.57913196e-05, 9.06909402e-03,
            8.98454452e-02, 3.00027025e-07, 0.00000000e+00, 3.02030150e+10,
            4.03019005e+11, 5.56559180e+11, 4.31031003e+12, 1.85725589e+13,
            1.27638339e+13, 1.27548618e+02, 2.80917505e-03, 1.38923674e-01,
            1.23362554e-06, 4.61511943e+13],
           [3.21601332e+04, 3.45256143e+00, 0.00000000e+00, 0.00000000e+00,
            2.76447638e+12, 2.77911904e+04, 0.00000000e+00, 3.39569151e+06,
            3.39569151e+06, 3.39569151e+06, 1.57628652e+08, 5.41849680e+02,
            4.80084271e+10, 4.57357607e-09, 3.25917050e-01, 3.38973659e+01,
            2.17601724e-01, 4.22909594e+01, 5.18838858e-01, 5.02612284e+01,
            1.72886585e+00, 4.99632383e+00, 3.70699371e-05, 1.18012360e-02,
            1.51218760e-01, 4.44174684e-07, 0.00000000e+00, 4.01343707e+10,
            4.22805166e+11, 5.29887978e+11, 4.68393307e+12, 1.85493620e+13,
            1.54127659e+13, 1.34722411e+02, 3.17693880e-03, 1.84043401e-01,
            1.33672552e-06, 4.69717784e+13],
           [3.21601332e+04, 3.45256143e+00, 0.00000000e+00, 0.00000000e+00,
            2.61309287e+12, 2.68711060e+04, 0.00000000e+00, 3.39569151e+06,
            3.39569151e+06, 3.39569151e+06, 1.57798668e+08, 5.37554869e+02,
            4.76306262e+10, 4.16305670e-09, 3.96417218e-01, 3.49011375e+01,
            2.37648630e-01, 4.14267338e+01, 4.77039025e-01, 5.00044956e+01,
            1.67160143e+00, 3.13727175e+00, 4.24131240e-05, 5.27286681e-03,
            1.02255753e-01, 4.65810260e-07, 0.00000000e+00, 3.34086922e+10,
            4.14771141e+11, 5.39756925e+11, 4.59734736e+12, 1.72790574e+13,
            1.34023646e+13, 1.34010005e+02, 3.08808609e-03, 1.49276924e-01,
            1.31399086e-06, 4.40840636e+13]])




```python
# Mean of channel values over time
mycruncher.means()
```




    array([[ 6.00000000e+01,  9.47440699e+00,  0.00000000e+00,
             0.00000000e+00,  7.65198252e-03,  3.38654182e-03,
             0.00000000e+00,  2.86660924e-03,  1.07333071e-02,
             0.00000000e+00, -1.19349938e-03,  1.23536829e+03,
             1.25880150e+01,  0.00000000e+00,  3.18820225e+01,
             3.18820225e+01,  3.18820225e+01,  1.79935436e+02,
             7.17695430e+00,  6.95867723e+02, -5.17090333e-04,
             2.50402554e-01,  1.93591250e+00, -4.42362547e-02,
             0.00000000e+00,  1.90341885e+00,  1.22672300e-01,
             0.00000000e+00,  1.85169072e+00, -2.97941960e-01,
             0.00000000e+00,  4.78828340e+01,  5.53099001e+01,
             5.65999376e+01, -2.43848315e-02, -1.62732884e-03,
            -5.43659850e-02,  1.47178914e-01, -1.38026115e-02,
             0.00000000e+00,  3.25186017e+01, -4.99510314e-02,
            -1.34598809e-01,  4.10221463e-02,  4.49336192e-01,
            -2.49922965e-01, -6.27209320e-02, -4.76854682e-05,
             3.68551061e-03,  1.06168428e+02, -9.16578027e+00,
             3.21034956e+02,  2.87685393e+02,  3.54242197e+03,
            -6.29183059e+00,  1.00946222e+02,  1.74133090e+01,
             2.62917305e+02, -2.67731323e+02,  3.43179089e+03,
            -4.13698340e+00,  1.02345858e+02, -4.56916679e+01,
             2.68011596e+02,  1.00910487e+03,  3.44064831e+03,
            -1.69243071e+00,  4.61886292e+01,  8.79323795e+02,
            -8.09610262e+00,  3.51814145e+01,  8.09106467e+02,
            -5.59885120e+00, -6.77125094e+00,  9.92334819e+02,
            -6.77470911e+00,  3.27738464e+02,  2.44648352e+01,
             2.28264100e+02,  1.08775560e+03,  6.03627815e+02,
            -6.40338077e+01,  2.67143059e+02, -5.22003720e+00,
            -3.42626717e+03,  1.09641897e+03, -3.73226055e+02,
            -8.14319600e+01,  3.01164744e+02, -6.26703471e+00,
            -6.83057428e+03,  1.58272072e+03,  2.43849233e+04,
            -8.26205406e+01,  1.34110462e+03,  1.22598340e+04,
            -8.10144020e+01,  2.06850187e+02,  8.22606742e+01,
             0.00000000e+00,  0.00000000e+00,  2.39981273e+02,
             7.52605493e+01,  0.00000000e+00,  0.00000000e+00,
             3.44184769e+02,  6.14156055e+01,  0.00000000e+00,
             0.00000000e+00,  5.34882647e+02,  4.87066167e+01,
             1.76122295e+02,  0.00000000e+00,  5.33192260e+02,
             4.88169788e+01,  1.74341287e+02,  0.00000000e+00,
             3.43925094e+02,  6.14287141e+01,  0.00000000e+00,
             0.00000000e+00,  2.39377029e+02,  7.53580524e+01,
             0.00000000e+00,  0.00000000e+00,  2.06938826e+02,
             8.22459426e+01,  0.00000000e+00,  0.00000000e+00,
             5.01250255e+00,  1.98195028e-01,  4.97490737e-01,
             2.59766725e-02,  5.14264633e+03,  5.33348392e+03,
             5.57251262e+03],
           [ 6.00000000e+01,  9.47403371e+00,  0.00000000e+00,
             0.00000000e+00,  3.28332734e-02,  1.31223617e-02,
             0.00000000e+00, -1.28483273e-02, -2.87691511e-03,
             0.00000000e+00, -4.70230220e-03,  1.27572909e+03,
             1.28609114e+01,  0.00000000e+00,  3.18820225e+01,
             3.18820225e+01,  3.18820225e+01,  1.82675090e+02,
             7.25931989e+00,  7.04212287e+02, -5.35746017e-04,
             2.88965901e-01,  1.97815208e+00, -2.39963337e-02,
             0.00000000e+00,  1.93803429e+00,  1.01733687e-01,
             0.00000000e+00,  1.88510818e+00, -3.00619067e-01,
             0.00000000e+00,  4.86517303e+01,  5.55030774e+01,
             5.61701199e+01, -1.42469501e-02,  7.39802946e-04,
            -7.47252591e-02,  1.47835629e-01, -1.45485218e-02,
             0.00000000e+00,  3.22991760e+01, -4.64842420e-02,
            -1.09192322e-01,  4.54848147e-02,  4.29511104e-01,
            -2.88429482e-01, -6.10315805e-02, -8.63179001e-05,
             3.24080799e-03,  1.07522113e+02, -4.55984207e+00,
             3.26600999e+02,  2.00882432e+02,  3.60478099e+03,
            -6.94366563e+00,  1.04167161e+02,  1.30650287e+01,
             2.66257326e+02, -1.80796401e+02,  3.52082633e+03,
            -2.52692800e+00,  1.03480896e+02, -4.69617381e+01,
             2.79989649e+02,  1.03239385e+03,  3.49048891e+03,
            -1.18223893e+00,  4.43068906e+01,  8.94499953e+02,
            -7.40038445e+00,  3.75845190e+01,  8.28543188e+02,
            -7.02820061e+00, -4.67112547e+00,  1.01323698e+03,
            -5.50352356e+00,  3.31773370e+02,  5.56103216e+01,
             2.18827302e+02,  1.11267161e+03,  5.73569715e+02,
            -9.52311336e+01,  2.68942825e+02, -6.09547497e+00,
            -3.41671036e+03,  1.12281668e+03, -3.50316764e+02,
            -8.44249150e+01,  2.99610922e+02, -7.54803458e+00,
            -6.81807740e+03,  1.70562042e+03,  2.44228373e+04,
            -8.52681126e+01,  1.41383326e+03,  1.23270051e+04,
            -8.40415149e+01,  2.07519600e+02,  8.21396130e+01,
             0.00000000e+00,  0.00000000e+00,  2.40354931e+02,
             7.51978402e+01,  0.00000000e+00,  0.00000000e+00,
             3.44118102e+02,  6.14480899e+01,  0.00000000e+00,
             0.00000000e+00,  5.37217603e+02,  4.87864045e+01,
             1.83459510e+02,  0.00000000e+00,  5.35746442e+02,
             4.89000749e+01,  1.82537130e+02,  0.00000000e+00,
             3.43965793e+02,  6.14486267e+01,  0.00000000e+00,
             0.00000000e+00,  2.39694757e+02,  7.53025468e+01,
             0.00000000e+00,  0.00000000e+00,  2.07586891e+02,
             8.21180275e+01,  0.00000000e+00,  0.00000000e+00,
             5.07297564e+00,  2.04896189e-01,  5.04860911e-01,
             2.65811893e-02,  5.24443182e+03,  5.45419705e+03,
             5.65868218e+03],
           [ 6.00000000e+01,  9.47403371e+00,  0.00000000e+00,
             0.00000000e+00,  8.66469313e-03,  6.19838502e-03,
             0.00000000e+00, -1.01186900e-02, -5.85548851e-03,
             0.00000000e+00, -3.95656991e-03,  1.26290762e+03,
             1.27742821e+01,  0.00000000e+00,  3.18820225e+01,
             3.18820225e+01,  3.18820225e+01,  1.81640680e+02,
             7.23994442e+00,  7.02336506e+02, -5.87339468e-04,
             3.11894571e-01,  1.96699643e+00, -2.79737700e-02,
             0.00000000e+00,  1.92676084e+00,  1.11690443e-01,
             0.00000000e+00,  1.88132525e+00, -3.03361311e-01,
             0.00000000e+00,  4.83419438e+01,  5.54414994e+01,
             5.62864769e+01, -2.20142845e-02,  1.44000599e-03,
            -5.85417627e-02,  1.48874840e-01, -1.44258560e-02,
             0.00000000e+00,  3.24405243e+01, -4.17046104e-02,
            -1.33542798e-01,  4.49572618e-02,  4.45284494e-01,
            -3.11304365e-01, -6.12945167e-02, -7.97915506e-05,
             2.56493014e-03,  1.07031718e+02, -5.90752747e+00,
             3.25879151e+02,  2.25565166e+02,  3.58619913e+03,
            -6.02077745e+00,  1.03560120e+02,  1.47049788e+01,
             2.65860130e+02, -2.13714608e+02,  3.50165393e+03,
            -3.30314305e+00,  1.03314955e+02, -4.68971352e+01,
             2.76778813e+02,  1.03330074e+03,  3.48522632e+03,
            -4.79660724e-01,  4.59014501e+01,  8.92593811e+02,
            -7.01157207e+00,  3.73897167e+01,  8.18963075e+02,
            -7.02111411e+00, -6.00307303e+00,  1.01298512e+03,
            -5.27346948e+00,  3.31344398e+02,  4.57471323e+01,
             2.23823971e+02,  1.10478962e+03,  5.79210702e+02,
            -8.60900212e+01,  2.70258800e+02, -6.02324387e+00,
            -3.42392260e+03,  1.11399275e+03, -3.51095939e+02,
            -9.25901634e+01,  3.02540994e+02, -7.47930797e+00,
            -6.82704245e+03,  1.69083532e+03,  2.46142802e+04,
            -9.35909788e+01,  1.40175927e+03,  1.24089184e+04,
            -9.21719689e+01,  2.07211985e+02,  8.21967541e+01,
             0.00000000e+00,  0.00000000e+00,  2.40123471e+02,
             7.52279775e+01,  0.00000000e+00,  0.00000000e+00,
             3.44333084e+02,  6.14203870e+01,  0.00000000e+00,
             0.00000000e+00,  5.38761298e+02,  4.86901248e+01,
             1.85574897e+02,  0.00000000e+00,  5.37176904e+02,
             4.88091760e+01,  1.84526633e+02,  0.00000000e+00,
             3.44276404e+02,  6.14105493e+01,  0.00000000e+00,
             0.00000000e+00,  2.39467416e+02,  7.53328464e+01,
             0.00000000e+00,  0.00000000e+00,  2.07297503e+02,
             8.21710612e+01,  0.00000000e+00,  0.00000000e+00,
             5.06011340e+00,  2.02737261e-01,  5.03395828e-01,
             2.63888783e-02,  5.23119050e+03,  5.42248129e+03,
             5.64879224e+03]])




```python
# Median of channel values over time
mycruncher.medians()
```




    array([[ 6.00000000e+01,  9.40000000e+00,  0.00000000e+00,
             0.00000000e+00, -1.52000000e-02, -2.76000000e-02,
             0.00000000e+00, -1.29000000e-02, -1.86000000e-03,
             0.00000000e+00,  6.05000000e-02,  2.30000000e+03,
             2.40000000e+01,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.92000000e+02,
             1.00000000e+01,  9.74000000e+02,  9.87000000e-06,
            -8.82000000e-02,  3.34000000e+00, -8.95000000e-02,
             0.00000000e+00,  3.40000000e+00,  5.27000000e-02,
             0.00000000e+00,  3.41000000e+00, -4.79000000e-01,
             0.00000000e+00,  5.38000000e+01,  6.38000000e+01,
             6.34000000e+01, -7.60000000e-02,  1.91000000e-03,
            -5.33000000e-02,  2.78000000e-01, -1.54000000e-02,
             0.00000000e+00,  3.41000000e+01, -3.82000000e-02,
            -9.87000000e-02,  5.17000000e-02,  1.07000000e+00,
             9.09000000e-02, -4.90000000e-02, -2.78000000e-05,
             1.08000000e-02,  1.57000000e+02,  1.12000000e+01,
             2.81000000e+02, -1.62000000e+02,  5.85000000e+03,
            -2.82000000e+00,  1.67000000e+02,  6.05000000e+01,
             3.43000000e+02, -1.17000000e+03,  6.22000000e+03,
            -8.73000000e-01,  1.70000000e+02, -8.57000000e+01,
             3.00000000e+02,  1.78000000e+03,  6.19000000e+03,
             3.76000000e+00, -6.56000000e+00,  1.62000000e+03,
            -2.40000000e+00,  3.82000000e+01,  1.65000000e+03,
            -2.35000000e+00, -1.22000000e+01,  1.65000000e+03,
            -2.56000000e+00,  4.63000000e+02,  1.45000000e+02,
             5.64000000e+02,  2.32000000e+03,  1.16000000e+03,
            -3.12000000e+02,  4.72000000e+02, -7.19000000e+00,
            -3.41000000e+03,  2.34000000e+03,  1.21000000e+02,
             8.42000000e-01,  5.72000000e+02, -5.34000000e+00,
            -6.82000000e+03,  1.72000000e+03,  4.59000000e+04,
            -3.89000000e-01,  1.46000000e+03,  2.33000000e+04,
             1.31000000e+00,  2.05000000e+02,  8.27000000e+01,
             0.00000000e+00,  0.00000000e+00,  2.38000000e+02,
             7.55000000e+01,  0.00000000e+00,  0.00000000e+00,
             3.44000000e+02,  6.11000000e+01,  0.00000000e+00,
             0.00000000e+00,  5.47000000e+02,  4.76000000e+01,
             1.92000000e+02,  0.00000000e+00,  5.45000000e+02,
             4.77000000e+01,  1.89000000e+02,  0.00000000e+00,
             3.44000000e+02,  6.12000000e+01,  0.00000000e+00,
             0.00000000e+00,  2.38000000e+02,  7.55000000e+01,
             0.00000000e+00,  0.00000000e+00,  2.05000000e+02,
             8.27000000e+01,  0.00000000e+00,  0.00000000e+00,
             7.04000000e+00,  3.83000000e-01,  6.93000000e-01,
             5.47000000e-02,  6.14797479e+03,  6.44359923e+03,
             6.35244106e+03],
           [ 6.00000000e+01,  9.40000000e+00,  0.00000000e+00,
             0.00000000e+00,  1.17000000e-01,  2.55200000e-02,
             0.00000000e+00, -3.23100000e-02,  1.41800000e-02,
             0.00000000e+00,  4.88600000e-03,  2.39800000e+03,
             2.46500000e+01,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.96900000e+02,
             1.01700000e+01,  9.86400000e+02,  1.54000000e-04,
            -1.40800000e-01,  3.45700000e+00, -8.41600000e-02,
             0.00000000e+00,  3.51900000e+00,  3.99700000e-02,
             0.00000000e+00,  3.51300000e+00, -5.13500000e-01,
             0.00000000e+00,  5.37200000e+01,  6.38200000e+01,
             6.32000000e+01, -3.05000000e-02, -3.63700000e-04,
            -7.13900000e-02,  2.92000000e-01, -1.59200000e-02,
             0.00000000e+00,  3.56600000e+01, -3.48500000e-02,
            -6.20800000e-02,  4.55800000e-02,  9.62200000e-01,
             1.40100000e-01, -6.27200000e-02,  4.55200000e-05,
             2.53500000e-03,  1.66000000e+02,  1.93500000e+01,
             2.91000000e+02, -3.59300000e+02,  6.09500000e+03,
            -2.79100000e+00,  1.74100000e+02,  4.56800000e+01,
             3.51700000e+02, -8.55800000e+02,  6.49500000e+03,
            -4.67700000e-01,  1.84200000e+02, -8.89500000e+01,
             3.11200000e+02,  1.89800000e+03,  6.46700000e+03,
             3.63700000e+00, -9.07700000e+00,  1.67600000e+03,
            -1.92800000e+00,  5.66500000e+01,  1.69800000e+03,
            -3.58200000e+00, -8.16500000e+00,  1.70500000e+03,
            -1.60500000e+00,  4.98000000e+02,  1.85500000e+02,
             5.81200000e+02,  2.39100000e+03,  1.09100000e+03,
            -3.31500000e+02,  5.13900000e+02, -6.54100000e+00,
            -3.41300000e+03,  2.37000000e+03,  2.47600000e+02,
             2.43800000e+01,  5.63100000e+02, -7.61200000e+00,
            -6.80000000e+03,  1.96800000e+03,  4.71900000e+04,
             2.27400000e+01,  1.52900000e+03,  2.44600000e+04,
             2.40600000e+01,  2.03400000e+02,  8.31100000e+01,
             0.00000000e+00,  0.00000000e+00,  2.37400000e+02,
             7.57100000e+01,  0.00000000e+00,  0.00000000e+00,
             3.52700000e+02,  6.05000000e+01,  0.00000000e+00,
             0.00000000e+00,  5.75900000e+02,  4.64000000e+01,
             2.33400000e+02,  0.00000000e+00,  5.76900000e+02,
             4.63700000e+01,  2.34800000e+02,  0.00000000e+00,
             3.51900000e+02,  6.05500000e+01,  0.00000000e+00,
             0.00000000e+00,  2.37500000e+02,  7.56800000e+01,
             0.00000000e+00,  0.00000000e+00,  2.03200000e+02,
             8.31300000e+01,  0.00000000e+00,  0.00000000e+00,
             7.13000000e+00,  4.01200000e-01,  7.46900000e-01,
             5.63400000e-02,  6.43020546e+03,  6.65293663e+03,
             6.54591179e+03],
           [ 6.00000000e+01,  9.40000000e+00,  0.00000000e+00,
             0.00000000e+00,  6.22700000e-02,  5.86300000e-02,
             0.00000000e+00, -3.54100000e-02, -2.97100000e-03,
             0.00000000e+00, -1.39700000e-02,  2.37100000e+03,
             2.44800000e+01,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.95400000e+02,
             1.01600000e+01,  9.85900000e+02,  1.69500000e-04,
            -1.48800000e-01,  3.56100000e+00, -8.27000000e-02,
             0.00000000e+00,  3.52300000e+00,  4.94000000e-02,
             0.00000000e+00,  3.50600000e+00, -4.92000000e-01,
             0.00000000e+00,  5.40200000e+01,  6.37900000e+01,
             6.32600000e+01, -1.25500000e-02,  3.15900000e-03,
            -5.28000000e-02,  3.08800000e-01, -1.57000000e-02,
             0.00000000e+00,  3.58800000e+01, -3.18600000e-02,
            -5.20500000e-02,  4.77400000e-02,  1.06300000e+00,
             1.50200000e-01, -6.37000000e-02,  1.75500000e-05,
            -6.53800000e-03,  1.78700000e+02,  1.56500000e+01,
             2.87200000e+02, -2.96600000e+02,  6.47000000e+03,
            -2.42500000e+00,  1.81900000e+02,  5.17700000e+01,
             3.48600000e+02, -9.86700000e+02,  6.59400000e+03,
            -8.53600000e-01,  1.86600000e+02, -8.66800000e+01,
             3.09600000e+02,  1.82600000e+03,  6.49100000e+03,
             3.89000000e+00, -1.10500000e+01,  1.72900000e+03,
            -1.84200000e+00,  4.71700000e+01,  1.70000000e+03,
            -3.31000000e+00, -1.48600000e+01,  1.70300000e+03,
            -1.42800000e+00,  5.41000000e+02,  1.83800000e+02,
             5.68300000e+02,  2.37900000e+03,  1.05600000e+03,
            -3.11200000e+02,  5.40600000e+02, -6.47100000e+00,
            -3.43300000e+03,  2.35500000e+03,  3.55700000e+02,
             2.67900000e+01,  5.95800000e+02, -6.57600000e+00,
            -6.82900000e+03,  1.97200000e+03,  4.99400000e+04,
             2.42600000e+01,  1.51100000e+03,  2.58600000e+04,
             2.73400000e+01,  2.04100000e+02,  8.31100000e+01,
             0.00000000e+00,  0.00000000e+00,  2.38600000e+02,
             7.56300000e+01,  0.00000000e+00,  0.00000000e+00,
             3.53300000e+02,  6.04500000e+01,  0.00000000e+00,
             0.00000000e+00,  5.79900000e+02,  4.62100000e+01,
             2.39100000e+02,  0.00000000e+00,  5.80300000e+02,
             4.62000000e+01,  2.39700000e+02,  0.00000000e+00,
             3.52500000e+02,  6.04900000e+01,  0.00000000e+00,
             0.00000000e+00,  2.37900000e+02,  7.56100000e+01,
             0.00000000e+00,  0.00000000e+00,  2.04000000e+02,
             8.31300000e+01,  0.00000000e+00,  0.00000000e+00,
             7.12700000e+00,  3.97600000e-01,  8.07500000e-01,
             5.60700000e-02,  6.66906223e+03,  6.75352172e+03,
             6.56674887e+03]])




```python
# Standard deviation of channel values over time
mycruncher.stddevs()
```




    array([[1.15614301e+01, 7.33187373e-01, 0.00000000e+00, 0.00000000e+00,
            4.22167785e-01, 2.98854589e-01, 0.00000000e+00, 2.96800509e-01,
            2.80448935e-01, 0.00000000e+00, 2.83252233e-01, 1.23848349e+03,
            1.25933005e+01, 0.00000000e+00, 3.83888558e+01, 3.83888558e+01,
            3.83888558e+01, 9.13355615e+01, 4.16180832e+00, 4.03517624e+02,
            4.32985169e-03, 6.00615803e-01, 2.18179336e+00, 5.75612528e-01,
            0.00000000e+00, 2.19969396e+00, 6.73087633e-01, 0.00000000e+00,
            2.28504649e+00, 7.55956598e-01, 0.00000000e+00, 1.76632092e+01,
            1.57823044e+01, 1.45667500e+01, 1.05560425e+00, 5.58382353e-02,
            2.55337878e-01, 4.31425410e-01, 1.96576532e-02, 0.00000000e+00,
            5.00503028e+00, 5.03460156e-02, 3.75609084e-01, 7.12951653e-02,
            2.70958397e+00, 6.01353472e-01, 1.51377511e-01, 4.02569647e-03,
            1.55848509e-01, 1.04680766e+02, 1.17135641e+02, 1.70983981e+02,
            2.38836378e+03, 3.92927918e+03, 2.83745885e+01, 1.10355235e+02,
            1.21393533e+02, 2.66205462e+02, 2.54577129e+03, 4.06591591e+03,
            2.77034358e+01, 1.09071609e+02, 1.21465918e+02, 2.40994245e+02,
            2.50596846e+03, 4.07038526e+03, 2.87813073e+01, 3.69104515e+02,
            1.12810515e+03, 1.41563313e+01, 4.10414418e+02, 1.19507409e+03,
            1.21169729e+01, 3.38349829e+02, 1.08347111e+03, 1.32067635e+01,
            3.50952595e+02, 7.07182273e+02, 7.63017055e+02, 1.39098875e+03,
            1.64581161e+03, 1.52233260e+03, 6.56972075e+02, 2.13342813e+01,
            7.92768410e+01, 1.42497379e+03, 1.91441496e+03, 6.82149150e+02,
            8.95890852e+02, 2.56648387e+01, 1.29744471e+02, 2.73834418e+03,
            7.12070648e+04, 6.82028784e+02, 1.92834279e+03, 3.55444297e+04,
            6.82240330e+02, 7.12506631e+00, 1.52400322e+00, 0.00000000e+00,
            0.00000000e+00, 5.66531369e+00, 8.62937351e-01, 0.00000000e+00,
            0.00000000e+00, 1.59001509e+01, 1.54376638e+00, 0.00000000e+00,
            0.00000000e+00, 7.05258027e+01, 3.72638715e+00, 9.69645976e+01,
            0.00000000e+00, 7.29972926e+01, 3.89279707e+00, 9.94377700e+01,
            0.00000000e+00, 1.51435703e+01, 1.46917541e+00, 0.00000000e+00,
            0.00000000e+00, 5.00338678e+00, 7.40086847e-01, 0.00000000e+00,
            0.00000000e+00, 7.37069157e+00, 1.58104775e+00, 0.00000000e+00,
            0.00000000e+00, 2.95596372e+00, 2.21293551e-01, 5.11948909e-01,
            3.23694221e-02, 2.70722481e+03, 2.53302710e+03, 2.15689636e+03],
           [1.15614301e+01, 7.33226954e-01, 0.00000000e+00, 0.00000000e+00,
            5.55708165e-01, 2.75050237e-01, 0.00000000e+00, 2.78235040e-01,
            1.65753372e-01, 0.00000000e+00, 1.64493300e-01, 1.27895025e+03,
            1.28661957e+01, 0.00000000e+00, 3.83888558e+01, 3.83888558e+01,
            3.83888558e+01, 9.14268181e+01, 4.21147808e+00, 4.08611104e+02,
            4.72633316e-03, 6.54265692e-01, 2.20900183e+00, 5.84482675e-01,
            0.00000000e+00, 2.25209675e+00, 6.86175993e-01, 0.00000000e+00,
            2.32954442e+00, 7.64401238e-01, 0.00000000e+00, 1.72182508e+01,
            1.56984347e+01, 1.47877492e+01, 1.16307621e+00, 6.07780576e-02,
            2.73839019e-01, 4.82397407e-01, 2.06636749e-02, 0.00000000e+00,
            6.37628491e+00, 5.73379086e-02, 5.40220059e-01, 8.94745637e-02,
            3.22810397e+00, 6.55020119e-01, 1.89646304e-01, 5.25981532e-03,
            1.36142771e-01, 1.07054870e+02, 1.18911217e+02, 1.75373005e+02,
            2.42161724e+03, 3.99317774e+03, 3.04265124e+01, 1.16744355e+02,
            1.20481102e+02, 2.72713407e+02, 2.53346528e+03, 4.22319601e+03,
            2.80843472e+01, 1.11664054e+02, 1.20265714e+02, 2.41518437e+02,
            2.48495088e+03, 4.15339614e+03, 2.89344031e+01, 3.79906280e+02,
            1.15432231e+03, 1.48623994e+01, 4.17600529e+02, 1.20742514e+03,
            1.23995715e+01, 3.43616159e+02, 1.11037881e+03, 1.35295247e+01,
            3.72654161e+02, 7.16944566e+02, 7.50491047e+02, 1.41997912e+03,
            1.64276941e+03, 1.58989875e+03, 7.31709700e+02, 2.54977356e+01,
            8.30525217e+01, 1.45385592e+03, 2.11369872e+03, 7.44764209e+02,
            1.00542513e+03, 3.29685428e+01, 1.27998211e+02, 3.08235983e+03,
            7.97126196e+04, 7.44710748e+02, 2.02555160e+03, 3.97152630e+04,
            7.44813457e+02, 9.45192067e+00, 2.00175129e+00, 0.00000000e+00,
            0.00000000e+00, 7.25086615e+00, 1.11297678e+00, 0.00000000e+00,
            0.00000000e+00, 1.95397711e+01, 1.92194521e+00, 0.00000000e+00,
            0.00000000e+00, 8.42369042e+01, 4.65915522e+00, 1.09783238e+02,
            0.00000000e+00, 8.71225291e+01, 4.86228331e+00, 1.12084307e+02,
            0.00000000e+00, 1.87780340e+01, 1.84664290e+00, 0.00000000e+00,
            0.00000000e+00, 6.45528923e+00, 9.60415019e-01, 0.00000000e+00,
            0.00000000e+00, 9.76487325e+00, 2.06908035e+00, 0.00000000e+00,
            0.00000000e+00, 2.99366729e+00, 2.28355517e-01, 5.42581934e-01,
            3.30350300e-02, 2.70950312e+03, 2.63348932e+03, 2.15758773e+03],
           [1.15614301e+01, 7.33226954e-01, 0.00000000e+00, 0.00000000e+00,
            6.46525960e-01, 2.64543430e-01, 0.00000000e+00, 2.56099742e-01,
            1.28625632e-01, 0.00000000e+00, 1.24944726e-01, 1.26457439e+03,
            1.27729464e+01, 0.00000000e+00, 3.83888558e+01, 3.83888558e+01,
            3.83888558e+01, 9.13841074e+01, 4.20507861e+00, 4.07994685e+02,
            4.36892211e-03, 6.85354117e-01, 2.21946208e+00, 5.88216301e-01,
            0.00000000e+00, 2.24421568e+00, 6.80572214e-01, 0.00000000e+00,
            2.32437735e+00, 7.61228594e-01, 0.00000000e+00, 1.73801113e+01,
            1.57256666e+01, 1.47980806e+01, 1.01262015e+00, 6.19917049e-02,
            2.20886634e-01, 4.35426790e-01, 2.07509213e-02, 0.00000000e+00,
            6.25468591e+00, 5.55646761e-02, 6.37185656e-01, 9.35379222e-02,
            2.89462036e+00, 6.86051309e-01, 1.71536622e-01, 4.98143763e-03,
            1.13290778e-01, 1.06188126e+02, 1.18161461e+02, 1.74558260e+02,
            2.40770100e+03, 3.99817940e+03, 2.94860525e+01, 1.14648330e+02,
            1.20896435e+02, 2.71182011e+02, 2.53964224e+03, 4.18886292e+03,
            2.78487073e+01, 1.10861376e+02, 1.20802274e+02, 2.42498408e+02,
            2.49353104e+03, 4.14249297e+03, 2.91253603e+01, 3.74815949e+02,
            1.15760840e+03, 1.45352805e+01, 4.16151168e+02, 1.20648201e+03,
            1.24195290e+01, 3.42643081e+02, 1.10595809e+03, 1.34553404e+01,
            3.59990978e+02, 7.13301341e+02, 7.55930859e+02, 1.41366042e+03,
            1.62224637e+03, 1.53899020e+03, 6.64672542e+02, 2.59132445e+01,
            6.96094079e+01, 1.44627967e+03, 1.92944998e+03, 6.88419076e+02,
            9.01201881e+02, 3.33695721e+01, 1.04792909e+02, 3.11180798e+03,
            7.17814696e+04, 6.88217636e+02, 2.03543402e+03, 3.58916052e+04,
            6.88500506e+02, 9.07536519e+00, 1.93878135e+00, 0.00000000e+00,
            0.00000000e+00, 6.92953284e+00, 1.06933697e+00, 0.00000000e+00,
            0.00000000e+00, 1.93375545e+01, 1.90212319e+00, 0.00000000e+00,
            0.00000000e+00, 8.29475774e+01, 4.58651116e+00, 1.07963246e+02,
            0.00000000e+00, 8.58814261e+01, 4.79338444e+00, 1.10213772e+02,
            0.00000000e+00, 1.84612333e+01, 1.81480937e+00, 0.00000000e+00,
            0.00000000e+00, 6.18675033e+00, 9.21361359e-01, 0.00000000e+00,
            0.00000000e+00, 9.40146182e+00, 2.00924971e+00, 0.00000000e+00,
            0.00000000e+00, 2.99067428e+00, 2.26658344e-01, 5.22924061e-01,
            3.28936095e-02, 2.70734522e+03, 2.62699307e+03, 2.16435966e+03]])




```python
# Skew of channel values over time
mycruncher.skews()
```

    /Users/gbarter/devel/pCrunch/pCrunch/aeroelastic_output.py:416: RuntimeWarning: invalid value encountered in divide
      return self.third_moments / np.sqrt(self.second_moments) ** 3





    array([[ 0.        ,  2.65336626,         nan,         nan,  0.02078122,
             0.00772224,         nan,  0.59549649,  0.59549649,  0.59549649,
            -0.06827807, -0.72129051, -0.7212567 , -2.03223151,  0.78194229,
            -0.25720616,  0.3101788 , -0.35171318,  0.28902529, -0.38938752,
             1.44900917,  0.17504459, -0.01904679,  0.32174313, -0.82483607,
             0.24769592,         nan, -0.67714933, -0.21768293, -0.50312488,
            -0.11448448, -0.74747688,  0.32644179, -0.62494164, -0.09917534,
            -0.67932975, -0.10039556, -0.29541754],
           [ 0.        ,  2.65685773,         nan,         nan,  0.02078847,
             0.00766576,         nan,  0.59549649,  0.59549649,  0.59549649,
            -0.12186788, -0.72479038, -0.72437278, -2.21204139,  0.78021022,
            -0.25864842,  0.22294603, -0.37447278,  0.44241491, -0.41560236,
             1.54672098,  0.17733024, -0.10541872,  0.23070655, -0.90416767,
             0.29340951,         nan, -0.7905145 , -0.26053364, -0.5054796 ,
            -0.11546489, -0.70141934,  0.24553819, -0.62601271, -0.09872956,
            -0.79706518, -0.10039149, -0.20046128],
           [ 0.        ,  2.65685773,         nan,         nan,  0.01318989,
             0.00445792,         nan,  0.59549649,  0.59549649,  0.59549649,
            -0.10439109, -0.72228777, -0.72182343, -2.64719164,  0.79494613,
            -0.28790992,  0.2923806 , -0.37614296,  0.35876246, -0.42207743,
             1.50973577,  0.16958377, -0.21759733,  0.35986151, -0.99582507,
             0.35488819,         nan, -0.7639969 , -0.24881844, -0.5016603 ,
            -0.11962245, -0.67985861,  0.27781476, -0.62169182, -0.11156421,
            -0.76154691, -0.10502606, -0.22341377]])




```python
# Kurtosis of channel values over time
mycruncher.kurtosis()
```

    /Users/gbarter/devel/pCrunch/pCrunch/aeroelastic_output.py:420: RuntimeWarning: invalid value encountered in divide
      return self.fourth_moments / self.second_moments ** 2





    array([[ 1.79999626, 11.91662575,         nan,         nan,  1.03328332,
             1.01424824,         nan,  1.56353482,  1.56353482,  1.56353482,
             2.27753858,  1.71766437,  1.71790841,  8.72204274,  1.82614957,
             1.45157486,  2.04037492,  1.59631635,  2.13129248,  1.69329444,
             4.76949898,  2.25313579,  2.65305849,  2.13355049,  2.59342195,
             2.00924873,         nan,  1.99093218,  1.61138777,  1.6420046 ,
             1.15136769,  2.53134453,  2.37652806,  1.67062648,  1.1713961 ,
             2.02241265,  1.12368238,  2.78849312],
           [ 1.79999626, 11.94504332,         nan,         nan,  1.03323302,
             1.01416014,         nan,  1.56353482,  1.56353482,  1.56353482,
             2.25600854,  1.72242627,  1.72217318,  9.16555105,  1.77864998,
             1.42357926,  1.86455742,  1.64399072,  2.34040176,  1.70666827,
             5.06379094,  2.73034727,  2.71665559,  2.09867975,  2.79245263,
             2.43625957,         nan,  2.08109812,  1.60028736,  1.67032893,
             1.15208055,  2.54696248,  2.4121411 ,  1.67735782,  1.1683227 ,
             2.12353261,  1.12238904,  2.78512275],
           [ 1.79999626, 11.94504332,         nan,         nan,  1.02182686,
             1.00953455,         nan,  1.56353482,  1.56353482,  1.56353482,
             2.26266696,  1.71919969,  1.71896989, 11.42654978,  1.79676966,
             1.43829729,  1.98512149,  1.6331359 ,  2.22360198,  1.71309921,
             4.97820023,  2.98377363,  2.8718777 ,  2.21497613,  2.84463906,
             2.51223068,         nan,  1.98926669,  1.60219868,  1.65298885,
             1.15113675,  2.49489807,  2.38912024,  1.67517723,  1.17004497,
             1.99635686,  1.12239627,  2.81202515]])




```python
# Integration of channel values over time
mycruncher.integrated()
```




    array([[ 2.40000000e+03,  3.78980000e+02,  0.00000000e+00,
             0.00000000e+00,  3.18586900e-01,  1.45506000e-01,
             0.00000000e+00,  1.23832700e-01,  4.33616450e-01,
             0.00000000e+00, -5.73996500e-02,  4.94185000e+04,
             5.03547500e+02,  0.00000000e+00,  1.27462500e+03,
             1.27462500e+03,  1.27462500e+03,  7.19341423e+03,
             2.87187817e+02,  2.78452803e+04, -2.07112429e-02,
             1.00038398e+01,  7.74383080e+01, -1.78914950e+00,
             0.00000000e+00,  7.61321998e+01,  4.92485060e+00,
             0.00000000e+00,  7.40774884e+01, -1.18943255e+01,
             0.00000000e+00,  1.91505500e+03,  2.21192900e+03,
             2.26502450e+03, -1.01791250e+00, -6.63082700e-02,
            -2.18417020e+00,  5.90296550e+00, -5.51826343e-01,
             0.00000000e+00,  1.30115750e+03, -1.99829631e+00,
            -5.37945730e+00,  1.64098696e+00,  1.80961645e+01,
            -9.98462475e+00, -2.50321332e+00, -1.74655300e-03,
             1.45464700e-01,  4.24779055e+03, -3.73289500e+02,
             1.28457750e+04,  1.16388000e+04,  1.41710675e+05,
            -2.52957815e+02,  4.03917620e+03,  7.00238025e+02,
             1.05259130e+04, -1.07893895e+04,  1.37291475e+05,
            -1.64612185e+02,  4.09503410e+03, -1.82504630e+03,
             1.07184519e+04,  4.03079500e+04,  1.37653190e+05,
            -6.76038500e+01,  1.86006210e+03,  3.51787430e+04,
            -3.24398660e+02,  1.39404065e+03,  3.23577640e+04,
            -2.23766491e+02, -2.70643600e+02,  3.96872095e+04,
            -2.71061350e+02,  1.31164005e+04,  9.38341650e+02,
             9.13400220e+03,  4.35060431e+04,  2.41420690e+04,
            -2.48330400e+03,  1.07087795e+04, -2.08373240e+02,
            -1.37049250e+05,  4.38513871e+04, -1.48698285e+04,
            -3.26162500e+03,  1.20814230e+04, -2.50149990e+02,
            -2.73220500e+05,  6.32594150e+04,  9.78053680e+05,
            -3.30919015e+03,  5.36168650e+04,  4.91683850e+05,
            -3.24491930e+03,  8.27337500e+03,  3.29056000e+03,
             0.00000000e+00,  0.00000000e+00,  9.59877500e+03,
             3.01049250e+03,  0.00000000e+00,  0.00000000e+00,
             1.37687500e+04,  2.45649000e+03,  0.00000000e+00,
             0.00000000e+00,  2.14009250e+04,  1.94794750e+03,
             7.05154040e+03,  0.00000000e+00,  2.13333500e+04,
             1.95236000e+03,  6.98023355e+03,  0.00000000e+00,
             1.37583250e+04,  2.45702000e+03,  0.00000000e+00,
             0.00000000e+00,  9.57465000e+03,  3.01439000e+03,
             0.00000000e+00,  0.00000000e+00,  8.27690000e+03,
             3.28997500e+03,  0.00000000e+00,  0.00000000e+00,
             2.00575785e+02,  7.92801089e+00,  1.99103540e+01,
             1.03898412e+00,  2.05721342e+05,  2.13361952e+05,
             2.22926621e+05],
           [ 2.40000000e+03,  3.78965050e+02,  0.00000000e+00,
             0.00000000e+00,  1.31916760e+00,  5.30118835e-01,
             0.00000000e+00, -5.03973010e-01, -1.10075700e-01,
             0.00000000e+00, -1.92521953e-01,  5.10330000e+04,
             5.14463250e+02,  0.00000000e+00,  1.27462500e+03,
             1.27462500e+03,  1.27462500e+03,  7.30340486e+03,
             2.90482975e+02,  2.81791722e+04, -2.14295280e-02,
             1.15507844e+01,  7.91295274e+01, -9.82005665e-01,
             0.00000000e+00,  7.75179458e+01,  4.08734665e+00,
             0.00000000e+00,  7.54128224e+01, -1.19980136e+01,
             0.00000000e+00,  1.94575680e+03,  2.21965450e+03,
             2.24758980e+03, -6.28927351e-01,  2.91517080e-02,
            -3.00176620e+00,  5.93510694e+00, -5.81702432e-01,
             0.00000000e+00,  1.29234575e+03, -1.85935789e+00,
            -4.36580501e+00,  1.81985183e+00,  1.73450497e+01,
            -1.15293208e+01, -2.43362005e+00, -3.28393190e-03,
             1.27467860e-01,  4.30226137e+03, -1.89611675e+02,
             1.30675675e+04,  8.17894140e+03,  1.44211861e+05,
            -2.79201383e+02,  4.16857930e+03,  5.25766650e+02,
             1.06604859e+04, -7.30153835e+03,  1.40863244e+05,
            -1.00339691e+02,  4.14075637e+03, -1.87475261e+03,
             1.11976677e+04,  4.12166485e+04,  1.39651316e+05,
            -4.69602440e+01,  1.78513347e+03,  3.57868481e+04,
            -2.96602692e+02,  1.49081749e+03,  3.31339669e+04,
            -2.81048910e+02, -1.90664325e+02,  4.05219010e+04,
            -2.20171043e+02,  1.32803884e+04,  2.18092588e+03,
             8.75080595e+03,  4.45023895e+04,  2.29268446e+04,
            -3.71875690e+03,  1.07893827e+04, -2.43477898e+02,
            -1.36666750e+05,  4.49075778e+04, -1.39308814e+04,
            -3.37697085e+03,  1.20317524e+04, -3.01436035e+02,
            -2.72720300e+05,  6.81819999e+04,  9.80548635e+05,
            -3.41068566e+03,  5.65301670e+04,  4.94850555e+05,
            -3.36163667e+03,  8.30011000e+03,  3.28571525e+03,
             0.00000000e+00,  0.00000000e+00,  9.61369250e+03,
             3.00798500e+03,  0.00000000e+00,  0.00000000e+00,
             1.37659700e+04,  2.45780175e+03,  0.00000000e+00,
             0.00000000e+00,  2.14936025e+04,  1.95117100e+03,
             7.34331086e+03,  0.00000000e+00,  2.14347600e+04,
             1.95571575e+03,  7.30635704e+03,  0.00000000e+00,
             1.37598650e+04,  2.45782450e+03,  0.00000000e+00,
             0.00000000e+00,  9.58730250e+03,  3.01217100e+03,
             0.00000000e+00,  0.00000000e+00,  8.30279250e+03,
             3.28485275e+03,  0.00000000e+00,  0.00000000e+00,
             2.02995350e+02,  8.19601990e+00,  2.02090870e+01,
             1.06315992e+00,  2.09783674e+05,  2.18196947e+05,
             2.26368345e+05],
           [ 2.40000000e+03,  3.78965050e+02,  0.00000000e+00,
             0.00000000e+00,  3.34628710e-01,  2.46483070e-01,
             0.00000000e+00, -3.99726285e-01, -2.29721065e-01,
             0.00000000e+00, -1.60163375e-01,  5.05201750e+04,
             5.10998000e+02,  0.00000000e+00,  1.27462500e+03,
             1.27462500e+03,  1.27462500e+03,  7.26197924e+03,
             2.89708001e+02,  2.81041488e+04, -2.34987582e-02,
             1.24655976e+01,  7.86829572e+01, -1.14007624e+00,
             0.00000000e+00,  7.70664916e+01,  4.48505223e+00,
             0.00000000e+00,  7.52618314e+01, -1.21084805e+01,
             0.00000000e+00,  1.93339185e+03,  2.21719030e+03,
             2.25228365e+03, -9.32077845e-01,  5.71803150e-02,
            -2.35032210e+00,  5.97337985e+00, -5.76806826e-01,
             0.00000000e+00,  1.29803350e+03, -1.66823214e+00,
            -5.35651080e+00,  1.79904584e+00,  1.79427690e+01,
            -1.24419773e+01, -2.44726990e+00, -3.05770160e-03,
             1.02723452e-01,  4.28246182e+03, -2.43353975e+02,
             1.30387950e+04,  9.16258490e+03,  1.43465193e+05,
            -2.42199612e+02,  4.14402155e+03,  5.91369650e+02,
             1.06444532e+04, -8.61814505e+03,  1.40090940e+05,
            -1.31392254e+02,  4.13396171e+03, -1.87233577e+03,
             1.10693439e+04,  4.12565945e+04,  1.39438522e+05,
            -1.88802370e+01,  1.84924333e+03,  3.57102747e+04,
            -2.81014837e+02,  1.48358066e+03,  3.27512811e+04,
            -2.80757270e+02, -2.43115825e+02,  4.05123265e+04,
            -2.10968753e+02,  1.32618281e+04,  1.78732515e+03,
             8.95121255e+03,  4.41872190e+04,  2.31541261e+04,
            -3.35790535e+03,  1.08371899e+04, -2.40599717e+02,
            -1.36956225e+05,  4.45546691e+04, -1.39746673e+04,
            -3.70445230e+03,  1.21419043e+04, -2.98740409e+02,
            -2.73080700e+05,  6.75934186e+04,  9.87655174e+05,
            -3.74446795e+03,  5.60481715e+04,  4.97857680e+05,
            -3.68772535e+03,  8.28774750e+03,  3.28801250e+03,
             0.00000000e+00,  0.00000000e+00,  9.60439000e+03,
             3.00919700e+03,  0.00000000e+00,  0.00000000e+00,
             1.37745500e+04,  2.45669050e+03,  0.00000000e+00,
             0.00000000e+00,  2.15556975e+04,  1.94730000e+03,
             7.42856462e+03,  0.00000000e+00,  2.14923300e+04,
             1.95205950e+03,  7.38657666e+03,  0.00000000e+00,
             1.37722650e+04,  2.45629900e+03,  0.00000000e+00,
             0.00000000e+00,  9.57816500e+03,  3.01338950e+03,
             0.00000000e+00,  0.00000000e+00,  8.29115750e+03,
             3.28698625e+03,  0.00000000e+00,  0.00000000e+00,
             2.02480952e+02,  8.10968486e+00,  2.01483579e+01,
             1.05546960e+00,  2.09258713e+05,  2.16929129e+05,
             2.25974728e+05]])




```python
# Special instance of the integration that specifically uses
# the Power channel string to integrate over time and calculate energy
mycruncher.compute_energy('GenPwr')
```




    DLC2.3_1.out    49418.500
    DLC2.3_2.out    51033.000
    DLC2.3_3.out    50520.175
    Name: integrated, dtype: float64




```python
# Total travel across simulation- useful for pitch drives and yaw drivers
mycruncher.total_travel('BldPitch1')
```




    array([90., 90., 90.])


