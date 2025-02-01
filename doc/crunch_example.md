# pCrunch's Crunch class

The Crunch class is a general analysis tool for batches of time-series based data across multiple environmental conditions (i.e., a full wind speed and turbulence seed sweep). The methods are agnostic to the aeroelastic multibody simulation tool (OpenFAST or HAWC2 or Bladed or QBlade or in-house equivalents). The AeroelasticOutput class provides the data containers for each individual simulation.  The AeroelasticOutput class provides many analysis capabilities and the Crunch class extends them into their batch versions.

The Crunch class supports keeping all time series data in memory and a lean "streaming" version where outputs are processed and then deleted, retaining only the critical statistics and analysis outputs.

This file lays out some workflows and showcases capabilities of the Crunch class.  It is probably best to walk through the examples of the AeroelasticOutput class first.

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
ec = [
    "RotSpeed",
    "RotThrust",
    "RotTorq",
]

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
- Extreme event table
- Damage equivalent loads (DELs)
- Palmgren-Miner damage


```python
# The summary stats per each file are here:
mycruncher.summary_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
# Damage equivalent loads are found here:
mycruncher_mc.dels
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
# Palmgren-Miner damage can be viewed with:
mycruncher_mc.damage
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>9.208221e-31</td>
      <td>3.327773e-27</td>
      <td>1.146698e-31</td>
    </tr>
    <tr>
      <th>DLC2.3_2.out</th>
      <td>1.972404e-30</td>
      <td>3.504854e-28</td>
      <td>1.537632e-31</td>
    </tr>
    <tr>
      <th>DLC2.3_3.out</th>
      <td>2.211880e-30</td>
      <td>3.552505e-27</td>
      <td>2.174332e-31</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extreme events table. For each channel, there is a list of the extreme condition for each output case
mycruncher_mc.extremes
```




    {'RotSpeed': [{'Time': 61.8,
       'RotSpeed': 11.1,
       'RotThrust': 369.0,
       'RotTorq': 844.0},
      {'Time': 61.9, 'RotSpeed': 11.28, 'RotThrust': 367.0, 'RotTorq': 159.4},
      {'Time': 61.9, 'RotSpeed': 11.33, 'RotThrust': 317.0, 'RotTorq': 140.8}],
     'RotThrust': [{'Time': 51.6,
       'RotSpeed': 10.1,
       'RotThrust': 759.0,
       'RotTorq': 2410.0},
      {'Time': 60.35, 'RotSpeed': 10.41, 'RotThrust': 786.9, 'RotTorq': -1041.0},
      {'Time': 60.35, 'RotSpeed': 10.48, 'RotThrust': 746.2, 'RotTorq': -1046.0}],
     'RotTorq': [{'Time': 54.45,
       'RotSpeed': 10.6,
       'RotThrust': 546.0,
       'RotTorq': 2650.0},
      {'Time': 54.4, 'RotSpeed': 10.74, 'RotThrust': 554.0, 'RotTorq': 2701.0},
      {'Time': 54.4, 'RotSpeed': 10.61, 'RotThrust': 575.1, 'RotTorq': 2638.0}]}



### Crunching in "lean / streaming" mode

If operating in "lean / streaming" mode, the outputs can either be processed one at a time, or even more lean, the summary statistics themselves can be passed to the `cruncher` object to append to the running list.


```python
# Adding AeroelasticOutput objects in lean / streaming mode
for iout in outputs:
    mycruncher_lean.add_output( iout ) # Each output is processed without retaining the full time series

# Adding statistics incrementally, which is especially helpful when using parallel processing
# and finaly assembly of the full pool of outputs can still strain memory resources
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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




    (33953594698.8, 33724052761.50001)




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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>1.701702e-30</td>
      <td>2.410254e-27</td>
      <td>1.619554e-31</td>
    </tr>
    <tr>
      <th>Unweighted</th>
      <td>1.701702e-30</td>
      <td>2.410254e-27</td>
      <td>1.619554e-31</td>
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




    [array([  0, 448,   0,   0, 401, 401,   0,   0,   0,   0, 258, 798, 798,
            490, 262, 505, 284, 486, 247, 473, 788, 653, 552, 209, 553, 222,
              0, 522, 286, 137, 406, 507, 489, 798, 453, 555, 406, 492]),
     array([  0, 450,   0,   0, 401, 401,   0,   0,   0,   0, 373, 798, 798,
            490, 224, 507, 412, 487, 249, 474, 789, 656, 617, 720, 785,  18,
              0, 522, 288, 260, 406, 508, 489, 798, 453, 555, 406, 493]),
     array([  0, 450,   0,   0, 401, 401,   0,   0,   0,   0, 260, 798, 798,
            490, 431, 505, 412, 486, 249, 474, 788, 654, 617, 721, 553,  19,
              0, 522, 288, 260, 406, 506, 488, 798, 453, 555, 406, 492])]




```python
# Indices to the maximum value for each channel
mycruncher.idxmaxs()
```




    [array([800, 502,   0,   0, 313, 315,   0, 630, 630, 630, 487, 436, 438,
            364, 677, 257, 465, 234, 563, 284, 512, 555, 403, 298, 655, 551,
              0, 232, 109, 314, 289, 567, 515, 448, 292, 231, 289, 555]),
     array([800, 505,   0,   0, 307, 313,   0, 630, 630, 630, 260, 438, 438,
            363, 683, 259, 466, 235, 565, 285, 511, 783, 403, 552, 657, 618,
              0, 407, 114, 316, 288, 561, 515, 449, 291, 407, 289, 554]),
     array([800, 505,   0,   0, 307, 311,   0, 630, 630, 630, 486, 438, 438,
            600, 686, 262, 466, 238, 564, 288, 511, 782, 403, 552, 655, 617,
              0, 407, 113, 316, 288, 590, 515, 449, 291, 407, 289, 553])]




```python
# Minimum value of each channel
mycruncher.minima()
```




    [[40.0,
      8.2,
      0.0,
      0.0,
      -0.812,
      -0.542,
      0.0,
      -0.748,
      -0.723,
      0.0,
      -0.832,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0756,
      -0.032,
      -3.14,
      -0.0213,
      -0.307,
      -2.09,
      -0.938,
      0.0,
      -3.08,
      -0.894,
      0.0,
      -3.44,
      -1.22,
      0.0,
      9.61,
      9.39,
      9.52,
      -2.01,
      -0.132,
      -0.479,
      -0.82,
      -0.0485,
      0.0,
      19.5,
      -0.129,
      -1.01,
      -0.107,
      -5.44,
      -1.29,
      -0.351,
      -0.0192,
      -0.352,
      -127.0,
      -212.0,
      113.0,
      -3770.0,
      -5170.0,
      -80.0,
      -146.0,
      -205.0,
      -168.0,
      -4040.0,
      -6630.0,
      -67.5,
      -106.0,
      -208.0,
      -114.0,
      -3770.0,
      -5690.0,
      -66.7,
      -572.0,
      -1110.0,
      -54.6,
      -923.0,
      -1600.0,
      -50.1,
      -599.0,
      -1930.0,
      -48.3,
      -353.0,
      -1140.0,
      -1100.0,
      -1160.0,
      -3840.0,
      -3360.0,
      -1200.0,
      -65.3,
      -3620.0,
      -1310.0,
      -5210.0,
      -3360.0,
      -1680.0,
      -137.0,
      -7160.0,
      -4940.0,
      -135000.0,
      -3360.0,
      -2730.0,
      -67500.0,
      -3360.0,
      200.0,
      77.9,
      0.0,
      0.0,
      234.0,
      72.8,
      0.0,
      0.0,
      300.0,
      59.7,
      0.0,
      0.0,
      370.0,
      45.1,
      0.0,
      0.0,
      366.0,
      45.0,
      0.0,
      0.0,
      301.0,
      59.8,
      0.0,
      0.0,
      234.0,
      73.1,
      0.0,
      0.0,
      200.0,
      77.7,
      0.0,
      0.0,
      -0.0225,
      -0.215,
      -0.615,
      -0.0274,
      459.8058303240619,
      662.0205736984312,
      1873.8033881920483],
     [40.0,
      8.197,
      0.0,
      0.0,
      -1.055,
      -0.5438,
      0.0,
      -0.5059,
      -0.3195,
      0.0,
      -0.3156,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.1066,
      -0.02883,
      -2.835,
      -0.02337,
      -0.256,
      -2.202,
      -0.936,
      0.0,
      -3.287,
      -0.9156,
      0.0,
      -3.545,
      -1.213,
      0.0,
      9.442,
      9.327,
      9.362,
      -2.366,
      -0.1646,
      -0.5903,
      -0.9164,
      -0.05223,
      0.0,
      16.61,
      -0.1467,
      -1.192,
      -0.1464,
      -6.836,
      -1.416,
      -0.4089,
      -0.01879,
      -0.2331,
      -141.8,
      -209.4,
      114.4,
      -3699.0,
      -5613.0,
      -81.07,
      -158.6,
      -204.5,
      -174.6,
      -4150.0,
      -7086.0,
      -69.72,
      -107.5,
      -207.4,
      -108.2,
      -3815.0,
      -5790.0,
      -68.34,
      -570.2,
      -1160.0,
      -56.0,
      -929.0,
      -1726.0,
      -51.47,
      -606.3,
      -1983.0,
      -50.26,
      -406.7,
      -1128.0,
      -1097.0,
      -1177.0,
      -3969.0,
      -3729.0,
      -1327.0,
      -61.16,
      -3600.0,
      -1336.0,
      -5769.0,
      -3682.0,
      -1922.0,
      -129.6,
      -7120.0,
      -6761.0,
      -151400.0,
      -3681.0,
      -3542.0,
      -75200.0,
      -3682.0,
      198.8,
      76.69,
      0.0,
      0.0,
      233.1,
      72.18,
      0.0,
      0.0,
      291.9,
      59.56,
      0.0,
      0.0,
      346.8,
      44.34,
      0.0,
      0.0,
      343.3,
      44.26,
      0.0,
      0.0,
      292.4,
      59.62,
      0.0,
      0.0,
      233.1,
      72.52,
      0.0,
      0.0,
      198.5,
      76.57,
      0.0,
      0.0,
      -0.02021,
      -0.2193,
      -0.6875,
      -0.02778,
      277.64858720332074,
      844.2388524582366,
      2094.763881367301],
     [40.0,
      8.197,
      0.0,
      0.0,
      -1.135,
      -0.5403,
      0.0,
      -0.5156,
      -0.2482,
      0.0,
      -0.1836,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.7137,
      -0.02939,
      -2.905,
      -0.0232,
      -0.2462,
      -2.207,
      -0.9388,
      0.0,
      -3.265,
      -0.9062,
      0.0,
      -3.594,
      -1.214,
      0.0,
      9.493,
      9.429,
      9.502,
      -2.188,
      -0.1658,
      -0.4399,
      -0.8252,
      -0.05212,
      0.0,
      16.62,
      -0.1374,
      -1.303,
      -0.156,
      -5.865,
      -1.494,
      -0.4193,
      -0.01914,
      -0.19,
      -134.6,
      -207.9,
      120.9,
      -3714.0,
      -5449.0,
      -82.54,
      -158.5,
      -203.0,
      -171.8,
      -4043.0,
      -7079.0,
      -70.38,
      -114.6,
      -204.8,
      -110.0,
      -3821.0,
      -5994.0,
      -68.87,
      -566.5,
      -1176.0,
      -56.5,
      -931.0,
      -1697.0,
      -52.27,
      -617.6,
      -2000.0,
      -50.14,
      -373.3,
      -1117.0,
      -1092.0,
      -1182.0,
      -4008.0,
      -3676.0,
      -1205.0,
      -65.63,
      -3577.0,
      -1360.0,
      -5504.0,
      -3656.0,
      -1687.0,
      -133.1,
      -7077.0,
      -6942.0,
      -135300.0,
      -3655.0,
      -3654.0,
      -67950.0,
      -3656.0,
      198.7,
      76.67,
      0.0,
      0.0,
      232.7,
      72.15,
      0.0,
      0.0,
      295.0,
      59.64,
      0.0,
      0.0,
      350.6,
      44.76,
      0.0,
      0.0,
      346.9,
      44.7,
      0.0,
      0.0,
      295.8,
      59.74,
      0.0,
      0.0,
      232.7,
      72.5,
      0.0,
      0.0,
      198.4,
      76.53,
      0.0,
      0.0,
      -0.02061,
      -0.2265,
      -0.6201,
      -0.02789,
      347.604351913206,
      624.8899503112528,
      1968.2043958799097]]




```python
# Maximum value of each channel
mycruncher.maxima()
```




    [[80.0,
      12.7,
      0.0,
      0.0,
      1.11,
      0.875,
      0.0,
      0.67,
      0.68,
      0.0,
      0.537,
      2790.0,
      27.3,
      0.0,
      90.0,
      90.0,
      90.0,
      360.0,
      11.1,
      1090.0,
      0.00638,
      1.29,
      4.91,
      1.23,
      0.0,
      4.58,
      1.57,
      0.0,
      4.72,
      2.1,
      0.0,
      65.5,
      65.4,
      65.4,
      2.19,
      0.176,
      0.519,
      0.694,
      0.0272,
      0.0,
      37.2,
      0.0393,
      0.639,
      0.185,
      5.02,
      0.311,
      0.267,
      0.0103,
      0.383,
      250.0,
      170.0,
      675.0,
      4700.0,
      8870.0,
      58.2,
      232.0,
      184.0,
      637.0,
      4500.0,
      8190.0,
      59.5,
      238.0,
      156.0,
      623.0,
      4590.0,
      8550.0,
      66.7,
      1170.0,
      2390.0,
      12.5,
      1510.0,
      2250.0,
      18.4,
      808.0,
      2280.0,
      14.2,
      759.0,
      1090.0,
      1130.0,
      2650.0,
      2950.0,
      3590.0,
      1090.0,
      47.6,
      -3270.0,
      2690.0,
      2090.0,
      1010.0,
      1510.0,
      53.6,
      -6610.0,
      7220.0,
      117000.0,
      1000.0,
      4750.0,
      56800.0,
      1010.0,
      229.0,
      83.7,
      0.0,
      0.0,
      257.0,
      76.1,
      0.0,
      0.0,
      365.0,
      65.7,
      0.0,
      0.0,
      616.0,
      58.5,
      290.0,
      0.0,
      616.0,
      58.8,
      291.0,
      0.0,
      364.0,
      65.6,
      0.0,
      0.0,
      255.0,
      76.1,
      0.0,
      0.0,
      230.0,
      83.8,
      0.0,
      0.0,
      8.83,
      0.466,
      1.13,
      0.0624,
      9134.167592616199,
      9093.801366315409,
      8938.356798092142],
     [80.0,
      12.72,
      0.0,
      0.0,
      0.8633,
      0.4589,
      0.0,
      0.5329,
      0.3017,
      0.0,
      0.2949,
      2866.0,
      27.8,
      0.0,
      90.0,
      90.0,
      90.0,
      359.5,
      11.28,
      1103.0,
      0.006038,
      1.418,
      4.852,
      1.152,
      0.0,
      4.701,
      1.658,
      0.0,
      4.526,
      2.183,
      0.0,
      65.49,
      65.36,
      65.43,
      2.472,
      0.1519,
      0.4829,
      0.8368,
      0.03559,
      0.0,
      38.23,
      0.05326,
      0.7311,
      0.219,
      6.284,
      0.2581,
      0.3561,
      0.01429,
      0.3236,
      241.2,
      165.7,
      687.5,
      4639.0,
      8680.0,
      62.87,
      234.5,
      187.2,
      650.9,
      4510.0,
      8398.0,
      54.71,
      231.7,
      155.3,
      639.3,
      4567.0,
      8126.0,
      61.46,
      1257.0,
      2374.0,
      15.57,
      1555.0,
      2295.0,
      12.58,
      809.0,
      2206.0,
      15.29,
      786.9,
      1080.0,
      1117.0,
      2701.0,
      3094.0,
      3930.0,
      1257.0,
      63.92,
      -3242.0,
      2768.0,
      2327.0,
      951.6,
      1827.0,
      77.63,
      -6546.0,
      7645.0,
      140800.0,
      950.8,
      5132.0,
      68460.0,
      952.3,
      235.6,
      83.97,
      0.0,
      0.0,
      261.4,
      76.29,
      0.0,
      0.0,
      365.3,
      66.53,
      0.0,
      0.0,
      630.7,
      60.43,
      311.1,
      0.0,
      632.1,
      60.78,
      313.0,
      0.0,
      364.3,
      66.49,
      0.0,
      0.0,
      259.4,
      76.26,
      0.0,
      0.0,
      236.4,
      84.02,
      0.0,
      0.0,
      8.988,
      0.4801,
      1.171,
      0.06365,
      9079.4523022537,
      9292.426290485171,
      8863.443388711861],
     [80.0,
      12.72,
      0.0,
      0.0,
      1.122,
      0.3937,
      0.0,
      0.5563,
      0.299,
      0.0,
      0.3112,
      2765.0,
      27.14,
      0.0,
      90.0,
      90.0,
      90.0,
      358.7,
      11.33,
      1109.0,
      0.005589,
      1.496,
      4.616,
      1.235,
      0.0,
      4.545,
      1.616,
      0.0,
      4.349,
      2.162,
      0.0,
      65.5,
      65.38,
      65.42,
      2.2,
      0.1764,
      0.4486,
      0.7617,
      0.03669,
      0.0,
      37.79,
      0.0559,
      1.03,
      0.2157,
      5.65,
      0.2498,
      0.2983,
      0.01258,
      0.2637,
      230.2,
      166.6,
      694.5,
      4597.0,
      8198.0,
      56.68,
      224.1,
      185.1,
      639.2,
      4482.0,
      8066.0,
      56.28,
      228.1,
      154.0,
      639.3,
      4510.0,
      8041.0,
      59.0,
      1208.0,
      2264.0,
      13.24,
      1562.0,
      2223.0,
      12.92,
      841.9,
      2130.0,
      16.81,
      746.2,
      1071.0,
      1104.0,
      2638.0,
      3021.0,
      3811.0,
      1147.0,
      65.6,
      -3276.0,
      2717.0,
      1972.0,
      880.4,
      1662.0,
      79.13,
      -6614.0,
      7636.0,
      128200.0,
      880.5,
      5124.0,
      62330.0,
      880.4,
      235.8,
      83.91,
      0.0,
      0.0,
      261.6,
      76.29,
      0.0,
      0.0,
      367.5,
      66.34,
      0.0,
      0.0,
      617.6,
      60.25,
      292.6,
      0.0,
      618.5,
      60.63,
      293.9,
      0.0,
      366.2,
      66.26,
      0.0,
      0.0,
      259.6,
      76.27,
      0.0,
      0.0,
      236.7,
      83.95,
      0.0,
      0.0,
      9.012,
      0.4634,
      1.11,
      0.06217,
      8986.223846955962,
      9111.284688867976,
      8781.344257663515]]




```python
# Maximum value of absolute values of each channel
mycruncher.absmaxima()
```




    [array([8.00e+01, 1.27e+01, 0.00e+00, 0.00e+00, 2.79e+03, 2.73e+01,
            0.00e+00, 9.00e+01, 9.00e+01, 9.00e+01, 3.60e+02, 1.11e+01,
            1.09e+03, 2.13e-02, 1.29e+00, 4.91e+00, 1.23e+00, 4.58e+00,
            1.57e+00, 4.72e+00, 2.10e+00, 2.19e+00, 1.76e-01, 5.19e-01,
            8.20e-01, 4.85e-02, 0.00e+00, 7.59e+02, 1.14e+03, 1.13e+03,
            2.65e+03, 3.84e+03, 3.59e+03, 8.83e+00, 4.66e-01, 1.13e+00,
            6.24e-02, 4.93e+03]),
     array([8.000e+01, 1.272e+01, 0.000e+00, 0.000e+00, 2.866e+03, 2.780e+01,
            0.000e+00, 9.000e+01, 9.000e+01, 9.000e+01, 3.595e+02, 1.128e+01,
            1.103e+03, 2.337e-02, 1.418e+00, 4.852e+00, 1.152e+00, 4.701e+00,
            1.658e+00, 4.526e+00, 2.183e+00, 2.472e+00, 1.646e-01, 5.903e-01,
            9.164e-01, 5.223e-02, 0.000e+00, 7.869e+02, 1.128e+03, 1.117e+03,
            2.701e+03, 3.969e+03, 3.930e+03, 8.988e+00, 4.801e-01, 1.171e+00,
            6.365e-02, 5.283e+03]),
     array([8.000e+01, 1.272e+01, 0.000e+00, 0.000e+00, 2.765e+03, 2.714e+01,
            0.000e+00, 9.000e+01, 9.000e+01, 9.000e+01, 3.587e+02, 1.133e+01,
            1.109e+03, 2.320e-02, 1.496e+00, 4.616e+00, 1.235e+00, 4.545e+00,
            1.616e+00, 4.349e+00, 2.162e+00, 2.200e+00, 1.764e-01, 4.486e-01,
            8.252e-01, 5.212e-02, 0.000e+00, 7.462e+02, 1.117e+03, 1.104e+03,
            2.638e+03, 4.008e+03, 3.811e+03, 9.012e+00, 4.634e-01, 1.110e+00,
            6.217e-02, 5.112e+03])]




```python
# The range of data values (max - min)
mycruncher.ranges()
```




    [[40.0,
      4.5,
      0.0,
      0.0,
      1.9220000000000002,
      1.417,
      0.0,
      1.4180000000000001,
      1.403,
      0.0,
      1.369,
      2790.0,
      27.3,
      0.0,
      90.0,
      90.0,
      90.0,
      359.9244,
      11.132,
      1093.14,
      0.02768,
      1.597,
      7.0,
      2.168,
      0.0,
      7.66,
      2.464,
      0.0,
      8.16,
      3.3200000000000003,
      0.0,
      55.89,
      56.010000000000005,
      55.88000000000001,
      4.199999999999999,
      0.308,
      0.998,
      1.5139999999999998,
      0.0757,
      0.0,
      17.700000000000003,
      0.1683,
      1.649,
      0.292,
      10.46,
      1.601,
      0.618,
      0.0295,
      0.735,
      377.0,
      382.0,
      562.0,
      8470.0,
      14040.0,
      138.2,
      378.0,
      389.0,
      805.0,
      8540.0,
      14820.0,
      127.0,
      344.0,
      364.0,
      737.0,
      8360.0,
      14240.0,
      133.4,
      1742.0,
      3500.0,
      67.1,
      2433.0,
      3850.0,
      68.5,
      1407.0,
      4210.0,
      62.5,
      1112.0,
      2230.0,
      2230.0,
      3810.0,
      6790.0,
      6950.0,
      2290.0,
      112.9,
      350.0,
      4000.0,
      7300.0,
      4370.0,
      3190.0,
      190.6,
      550.0,
      12160.0,
      252000.0,
      4360.0,
      7480.0,
      124300.0,
      4370.0,
      29.0,
      5.799999999999997,
      0.0,
      0.0,
      23.0,
      3.299999999999997,
      0.0,
      0.0,
      65.0,
      6.0,
      0.0,
      0.0,
      246.0,
      13.399999999999999,
      290.0,
      0.0,
      250.0,
      13.799999999999997,
      291.0,
      0.0,
      63.0,
      5.799999999999997,
      0.0,
      0.0,
      21.0,
      3.0,
      0.0,
      0.0,
      30.0,
      6.099999999999994,
      0.0,
      0.0,
      8.852500000000001,
      0.681,
      1.7449999999999999,
      0.08979999999999999,
      8674.361762292137,
      8431.780792616977,
      7064.553409900093],
     [40.0,
      4.5230000000000015,
      0.0,
      0.0,
      1.9183,
      1.0027,
      0.0,
      1.0388000000000002,
      0.6212,
      0.0,
      0.6105,
      2866.0,
      27.8,
      0.0,
      90.0,
      90.0,
      90.0,
      359.3934,
      11.308829999999999,
      1105.835,
      0.029407999999999997,
      1.674,
      7.054,
      2.088,
      0.0,
      7.9879999999999995,
      2.5736,
      0.0,
      8.071,
      3.396,
      0.0,
      56.047999999999995,
      56.033,
      56.068000000000005,
      4.838,
      0.3165,
      1.0732,
      1.7532,
      0.08782,
      0.0,
      21.619999999999997,
      0.19996,
      1.9230999999999998,
      0.3654,
      13.120000000000001,
      1.6741,
      0.765,
      0.03308,
      0.5567,
      383.0,
      375.1,
      573.1,
      8338.0,
      14293.0,
      143.94,
      393.1,
      391.7,
      825.5,
      8660.0,
      15484.0,
      124.43,
      339.2,
      362.70000000000005,
      747.5,
      8382.0,
      13916.0,
      129.8,
      1827.2,
      3534.0,
      71.57,
      2484.0,
      4021.0,
      64.05,
      1415.3,
      4189.0,
      65.55,
      1193.6,
      2208.0,
      2214.0,
      3878.0,
      7063.0,
      7659.0,
      2584.0,
      125.08,
      358.0,
      4104.0,
      8096.0,
      4633.6,
      3749.0,
      207.23,
      574.0,
      14406.0,
      292200.0,
      4631.8,
      8674.0,
      143660.0,
      4634.3,
      36.79999999999998,
      7.280000000000001,
      0.0,
      0.0,
      28.299999999999983,
      4.109999999999999,
      0.0,
      0.0,
      73.40000000000003,
      6.969999999999999,
      0.0,
      0.0,
      283.90000000000003,
      16.089999999999996,
      311.1,
      0.0,
      288.8,
      16.520000000000003,
      313.0,
      0.0,
      71.90000000000003,
      6.869999999999997,
      0.0,
      0.0,
      26.299999999999983,
      3.740000000000009,
      0.0,
      0.0,
      37.900000000000006,
      7.450000000000003,
      0.0,
      0.0,
      9.00821,
      0.6994,
      1.8585,
      0.09143,
      8801.803715050379,
      8448.187438026935,
      6768.67950734456],
     [40.0,
      4.5230000000000015,
      0.0,
      0.0,
      2.257,
      0.9339999999999999,
      0.0,
      1.0718999999999999,
      0.5472,
      0.0,
      0.4948,
      2765.0,
      27.14,
      0.0,
      90.0,
      90.0,
      90.0,
      357.98629999999997,
      11.35939,
      1111.905,
      0.028789,
      1.7422,
      6.8229999999999995,
      2.1738,
      0.0,
      7.8100000000000005,
      2.5222,
      0.0,
      7.943,
      3.376,
      0.0,
      56.007,
      55.95099999999999,
      55.918,
      4.388,
      0.3422,
      0.8885000000000001,
      1.5869,
      0.08881,
      0.0,
      21.169999999999998,
      0.1933,
      2.333,
      0.37170000000000003,
      11.515,
      1.7438,
      0.7176,
      0.03172,
      0.4537,
      364.79999999999995,
      374.5,
      573.6,
      8311.0,
      13647.0,
      139.22,
      382.6,
      388.1,
      811.0,
      8525.0,
      15145.0,
      126.66,
      342.7,
      358.8,
      749.3,
      8331.0,
      14035.0,
      127.87,
      1774.5,
      3440.0,
      69.74,
      2493.0,
      3920.0,
      65.19,
      1459.5,
      4130.0,
      66.95,
      1119.5,
      2188.0,
      2196.0,
      3820.0,
      7029.0,
      7487.0,
      2352.0,
      131.23,
      301.0,
      4077.0,
      7476.0,
      4536.4,
      3349.0,
      212.23,
      463.0,
      14578.0,
      263500.0,
      4535.5,
      8778.0,
      130280.0,
      4536.4,
      37.10000000000002,
      7.239999999999995,
      0.0,
      0.0,
      28.900000000000034,
      4.140000000000001,
      0.0,
      0.0,
      72.5,
      6.700000000000003,
      0.0,
      0.0,
      267.0,
      15.490000000000002,
      292.6,
      0.0,
      271.6,
      15.93,
      293.9,
      0.0,
      70.39999999999998,
      6.520000000000003,
      0.0,
      0.0,
      26.900000000000034,
      3.769999999999996,
      0.0,
      0.0,
      38.29999999999998,
      7.420000000000002,
      0.0,
      0.0,
      9.03261,
      0.6899,
      1.7301000000000002,
      0.09006,
      8638.619495042756,
      8486.394738556723,
      6813.139861783606]]




```python
# Channel indices which vary in time
mycruncher.variable()
```




    [array([ 0,  1,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]),
     array([ 0,  1,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]),
     array([ 0,  1,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])]




```python
# Channel indices which are constant in time
mycruncher.constant()
```




    [array([ 2,  3,  6, 26]), array([ 2,  3,  6, 26]), array([ 2,  3,  6, 26])]




```python
# Sum of channel values over time
mycruncher.sums()
```




    [array([ 4.80600000e+04,  7.58900000e+03,  0.00000000e+00,  0.00000000e+00,
             9.89530000e+05,  1.00830000e+04,  0.00000000e+00,  2.55375000e+04,
             2.55375000e+04,  2.55375000e+04,  1.44128285e+05,  5.74874039e+03,
             5.57390046e+05, -4.14189357e-01,  2.00572446e+02,  1.55066591e+03,
            -3.54332400e+01,  1.52463849e+03,  9.82605120e+01,  1.48320427e+03,
            -2.38651510e+02, -1.95322500e+01, -1.30349040e+00, -4.35471540e+01,
             1.17890310e+02, -1.10558919e+01,  0.00000000e+00,  2.62518510e+05,
             1.95963330e+04,  1.82839544e+05,  8.71292238e+05,  4.83505880e+05,
            -5.12910800e+04,  4.01501454e+03,  1.58754217e+02,  3.98490080e+02,
             2.08073147e+01,  4.32214800e+05]),
     array([ 4.80600000e+04,  7.58870100e+03,  0.00000000e+00,  0.00000000e+00,
             1.02185900e+06,  1.03015900e+04,  0.00000000e+00,  2.55375000e+04,
             2.55375000e+04,  2.55375000e+04,  1.46322747e+05,  5.81471523e+03,
             5.64074042e+05, -4.29132560e-01,  2.31461687e+02,  1.58449982e+03,
            -1.92210633e+01,  1.55236547e+03,  8.14886831e+01,  1.50997165e+03,
            -2.40795873e+02, -1.14118070e+01,  5.92582160e-01, -5.98549325e+01,
             1.18416339e+02, -1.16533660e+01,  0.00000000e+00,  2.65750469e+05,
             4.45438676e+04,  1.75280669e+05,  8.91249956e+05,  4.59429342e+05,
            -7.62801380e+04,  4.06345349e+03,  1.64121847e+02,  4.04393590e+02,
             2.12915326e+01,  3.83149204e+05]),
     array([ 4.80600000e+04,  7.58870100e+03,  0.00000000e+00,  0.00000000e+00,
             1.01158900e+06,  1.02322000e+04,  0.00000000e+00,  2.55375000e+04,
             2.55375000e+04,  2.55375000e+04,  1.45494185e+05,  5.79919548e+03,
             5.62571542e+05, -4.70458914e-01,  2.49827552e+02,  1.57556414e+03,
            -2.24069898e+01,  1.54333543e+03,  8.94640447e+01,  1.50694153e+03,
            -2.42992410e+02, -1.76334419e+01,  1.15344480e+00, -4.68919519e+01,
             1.19248747e+02, -1.15551107e+01,  0.00000000e+00,  2.65406862e+05,
             3.66434530e+04,  1.79283001e+05,  8.84936482e+05,  4.63947772e+05,
            -6.89581070e+04,  4.05315083e+03,  1.62392546e+02,  4.03220058e+02,
             2.11374915e+01,  3.94989665e+05])]




```python
# Sum of channel values over time to the second power
mycruncher.sums_squared()
```




    [array([2.99066700e+06, 7.23318632e+04, 0.00000000e+00, 0.00000000e+00,
            2.45104090e+09, 2.53956520e+05, 0.00000000e+00, 1.99462425e+06,
            1.99462425e+06, 1.99462425e+06, 3.26158758e+07, 5.51322865e+04,
            5.18293747e+08, 1.52310134e-02, 3.39176067e+02, 6.81489156e+03,
            2.66962590e+02, 6.77778710e+03, 3.74944460e+02, 6.92880700e+03,
            5.28852071e+02, 8.93030853e+02, 2.49956593e+00, 5.45906269e+01,
            1.66439403e+02, 4.62125266e-01, 0.00000000e+00, 1.84694760e+08,
            4.01064942e+08, 5.08073920e+08, 2.49756762e+09, 2.46152297e+09,
            1.85959911e+09, 2.71241855e+04, 7.06899358e+01, 4.08180563e+02,
            1.37977617e+00, 3.49188164e+09]),
     array([2.99066700e+06, 7.23262441e+04, 0.00000000e+00, 0.00000000e+00,
            2.61382196e+09, 2.65084568e+05, 0.00000000e+00, 1.99462425e+06,
            1.99462425e+06, 1.99462425e+06, 3.34249703e+07, 5.64178526e+04,
            5.30965262e+08, 1.81228244e-02, 4.09763475e+02, 7.04301256e+03,
            2.74098853e+02, 7.07116127e+03, 3.85430977e+02, 7.19330845e+03,
            5.40419542e+02, 1.08371235e+03, 2.95931019e+00, 6.45379099e+01,
            2.03904667e+02, 5.11556206e-01, 0.00000000e+00, 1.99404699e+08,
            4.14198717e+08, 4.89508882e+08, 2.60675743e+09, 2.42516652e+09,
            2.03201446e+09, 2.77923977e+04, 7.53970809e+01, 4.39973036e+02,
            1.44009614e+00, 3.47276714e+09]),
     array([2.99066700e+06, 7.23262441e+04, 0.00000000e+00, 0.00000000e+00,
            2.55846131e+09, 2.61390686e+05, 0.00000000e+00, 1.99462425e+06,
            1.99462425e+06, 1.99462425e+06, 3.31168578e+07, 5.61496845e+04,
            5.28448721e+08, 1.55653909e-02, 4.54157780e+02, 7.04486461e+03,
            2.77771540e+02, 7.00787799e+03, 3.80998288e+02, 7.16263392e+03,
            5.37869143e+02, 8.21733248e+02, 3.07988112e+00, 4.18266526e+01,
            1.69619926e+02, 5.11603552e-01, 0.00000000e+00, 1.91745474e+08,
            4.09224175e+08, 4.97844436e+08, 2.57841571e+09, 2.37670182e+09,
            1.90309777e+09, 2.76736531e+04, 7.40735980e+01, 4.22012404e+02,
            1.42446832e+00, 3.36627034e+09])]




```python
# Sum of channel values over time to the third power
mycruncher.sums_cubed()
```




    [array([ 1.92288060e+08,  6.94298326e+05,  0.00000000e+00,  0.00000000e+00,
             6.09512315e+12,  6.40731260e+06,  0.00000000e+00,  1.65847398e+08,
             1.65847398e+08,  1.65847398e+08,  8.23177046e+09,  5.53178222e+05,
             5.04221025e+11, -1.55543032e-04,  3.65344809e+02,  2.58163535e+04,
             1.20947939e+01,  2.46568356e+04,  2.05625057e+02,  2.45975817e+04,
             7.10796755e+01,  9.96182890e+01, -1.48520740e-02, -4.35589968e+00,
             1.53278176e+01, -1.34159296e-02,  0.00000000e+00,  1.01753538e+11,
            -3.22542353e+10,  1.49847668e+11,  5.84157948e+12,  1.43604884e+12,
             5.65690138e+11,  1.93195077e+05,  2.86981674e+01,  3.38936129e+02,
             7.67175294e-02,  3.45922125e+12]),
     array([ 1.92288060e+08,  6.94219888e+05,  0.00000000e+00,  0.00000000e+00,
             6.71230136e+12,  6.83293668e+06,  0.00000000e+00,  1.65847398e+08,
             1.65847398e+08,  1.65847398e+08,  8.47749243e+09,  5.72455267e+05,
             5.22686792e+11, -2.15949030e-04,  4.91595938e+02,  2.71626676e+04,
             1.59472299e+01,  2.60249519e+04,  2.30437505e+02,  2.57402611e+04,
             1.09502185e+02,  1.77166444e+02, -1.23906496e-02, -1.00046839e+01,
             3.95596489e+00, -1.53204480e-02,  0.00000000e+00,  1.07198521e+11,
            -8.07906243e+09,  1.33418617e+11,  6.22978579e+12,  1.37991105e+12,
             2.11274187e+11,  2.00370633e+05,  3.16235461e+01,  3.58246493e+02,
             8.18518941e-02,  3.47182961e+12]),
     array([ 1.92288060e+08,  6.94219888e+05,  0.00000000e+00,  0.00000000e+00,
             6.48782731e+12,  6.68524869e+06,  0.00000000e+00,  1.65847398e+08,
             1.65847398e+08,  1.65847398e+08,  8.38161564e+09,  5.68591824e+05,
             5.19172832e+11, -2.03925755e-04,  5.81324273e+02,  2.68583738e+04,
             2.43883094e+01,  2.56430480e+04,  2.16015743e+02,  2.55127871e+04,
             8.86497887e+01,  8.67920936e+01, -2.82224593e-02, -3.91787274e+00,
             4.61963408e+00, -1.47915829e-02,  0.00000000e+00,  1.03774269e+11,
            -1.63234993e+10,  1.42749896e+11,  6.11488921e+12,  1.49365291e+12,
             3.20648189e+11,  1.99215329e+05,  3.06624163e+01,  3.45734171e+02,
             8.03371253e-02,  3.37793176e+12])]




```python
# Sum of channel values over time to the fourth power
mycruncher.sums_fourth()
```




    [array([1.27193675e+10, 6.72058716e+06, 0.00000000e+00, 0.00000000e+00,
            1.52191817e+16, 1.61942556e+08, 0.00000000e+00, 1.41881502e+10,
            1.41881502e+10, 1.41881502e+10, 2.23468362e+12, 5.63005878e+06,
            4.97577410e+14, 2.75297219e-06, 4.38130767e+02, 1.06767943e+05,
            1.74151600e+02, 1.01872083e+05, 4.17986394e+02, 1.04873960e+05,
            9.00199729e+02, 2.22800798e+03, 2.07158246e-02, 7.26447975e+00,
            6.04853833e+01, 5.39994955e-04, 0.00000000e+00, 6.62801332e+13,
            3.18222414e+14, 4.30309506e+14, 1.45026648e+16, 1.32816255e+16,
            1.00332283e+16, 1.40368241e+06, 1.20486157e+01, 3.26802571e+02,
            4.46744190e-03, 3.85369085e+16]),
     array([1.27193675e+10, 6.71962967e+06, 0.00000000e+00, 0.00000000e+00,
            1.73077680e+16, 1.76438826e+08, 0.00000000e+00, 1.41881502e+10,
            1.41881502e+10, 1.41881502e+10, 2.30428942e+12, 5.89126197e+06,
            5.21876656e+14, 4.09519763e-06, 6.40737567e+02, 1.13515183e+05,
            1.71822081e+02, 1.10169337e+05, 4.85685686e+02, 1.11323508e+05,
            9.79741027e+02, 3.99063936e+03, 2.96466358e-02, 1.03558992e+01,
            9.78748918e+01, 7.05342834e-04, 0.00000000e+00, 7.18301735e+13,
            3.29207326e+14, 4.06090981e+14, 1.57982336e+16, 1.34969948e+16,
            1.21547740e+16, 1.47386976e+06, 1.37061298e+01, 3.54137028e+02,
            4.86815847e-03, 3.96254617e+16]),
     array([1.27193675e+10, 6.71962967e+06, 0.00000000e+00, 0.00000000e+00,
            1.64965054e+16, 1.71182977e+08, 0.00000000e+00, 1.41881502e+10,
            1.41881502e+10, 1.41881502e+10, 2.27616268e+12, 5.84004675e+06,
            5.17362253e+14, 3.78177162e-06, 8.00439475e+02, 1.11706745e+05,
            1.86324902e+02, 1.07836380e+05, 4.50472627e+02, 1.10040069e+05,
            9.54738092e+02, 2.50292316e+03, 3.37720417e-02, 4.30915266e+00,
            6.32818249e+01, 6.91903169e-04, 0.00000000e+00, 6.69559817e+13,
            3.24116649e+14, 4.16535980e+14, 1.54023817e+16, 1.27874548e+16,
            1.05403788e+16, 1.46350347e+06, 1.31312447e+01, 3.28398422e+02,
            4.74606571e-03, 3.72049225e+16])]




```python
# Second moment of the timeseries for each channel
mycruncher.second_moments()
```




    [array([1.33666667e+02, 5.37563724e-01, 0.00000000e+00, 0.00000000e+00,
            1.53384134e+06, 1.58591217e+02, 0.00000000e+00, 1.47370425e+03,
            1.47370425e+03, 1.47370425e+03, 8.34218479e+03, 1.73206485e+01,
            1.62826473e+05, 1.87476156e-05, 3.60739343e-01, 4.76022226e+00,
            3.31329783e-01, 4.83865350e+00, 4.53046962e-01, 5.22143747e+00,
            5.71470377e-01, 1.11430033e+00, 3.11790852e-03, 6.51974319e-02,
            1.86127885e-01, 3.86423328e-04, 0.00000000e+00, 1.23167724e+05,
            5.00106768e+05, 5.82195026e+05, 1.93484970e+06, 2.70869585e+06,
            2.31749656e+06, 8.73772149e+00, 4.89708356e-02, 2.62091685e-01,
            1.04777949e-03, 4.06824110e+06]),
     array([1.33666667e+02, 5.37621765e-01, 0.00000000e+00, 0.00000000e+00,
            1.63571375e+06, 1.65538992e+02, 0.00000000e+00, 1.47370425e+03,
            1.47370425e+03, 1.47370425e+03, 8.35886306e+03, 1.77365476e+01,
            1.66963034e+05, 2.23382251e-05, 4.28063595e-01, 4.87968906e+00,
            3.41619997e-01, 5.07193978e+00, 4.70837494e-01, 5.42677721e+00,
            5.84309253e-01, 1.35274627e+00, 3.69397228e-03, 7.49878085e-02,
            2.32707258e-01, 4.26987461e-04, 0.00000000e+00, 1.38871124e+05,
            5.14009511e+05, 5.63236811e+05, 2.01634071e+06, 2.69869134e+06,
            2.52777804e+06, 8.96204384e+00, 5.21462422e-02, 2.94395155e-01,
            1.09131321e-03, 4.10673170e+06]),
     array([1.33666667e+02, 5.37621765e-01, 0.00000000e+00, 0.00000000e+00,
            1.59914839e+06, 1.63148160e+02, 0.00000000e+00, 1.47370425e+03,
            1.47370425e+03, 1.47370425e+03, 8.35105509e+03, 1.76826861e+01,
            1.66459663e+05, 1.90874804e-05, 4.69710266e-01, 4.92601194e+00,
            3.45998417e-01, 5.03650401e+00, 4.63178539e-01, 5.40273005e+00,
            5.79468973e-01, 1.02539958e+00, 3.84297147e-03, 4.87909052e-02,
            1.89596489e-01, 4.30600735e-04, 0.00000000e+00, 1.29593504e+05,
            5.08798804e+05, 5.71431464e+05, 1.99843580e+06, 2.63168328e+06,
            2.36849084e+06, 8.94413264e+00, 5.13740049e-02, 2.73449573e-01,
            1.08198954e-03, 3.95941669e+06])]




```python
# Third moment of the timeseries for each channel
mycruncher.third_moments()
```




    [array([ 0.00000000e+00,  1.04578434e+00,  0.00000000e+00,  0.00000000e+00,
             3.94767725e+07,  1.54227646e+01,  0.00000000e+00,  3.36895113e+04,
             3.36895113e+04,  3.36895113e+04, -5.20236650e+04, -5.19943844e+01,
            -4.73889823e+07, -1.64965164e-07,  1.69420113e-01, -2.67129717e+00,
             5.91565473e-02, -3.74347926e+00,  8.81354619e-02, -4.64587100e+00,
             6.25981817e-01,  2.05897979e-01, -3.31601828e-06,  5.35617821e-03,
            -6.62345831e-02,  1.88154176e-06,  0.00000000e+00, -2.92704786e+07,
            -7.69871917e+07, -2.23500517e+08, -3.08118295e+08, -3.33225423e+09,
             1.15168683e+09, -1.61412350e+01, -1.07475623e-03, -9.11508028e-02,
            -3.40501759e-06, -2.42407639e+09]),
     array([ 0.00000000e+00,  1.04733005e+00,  0.00000000e+00,  0.00000000e+00,
             4.34894071e+07,  1.63269772e+01,  0.00000000e+00,  3.36895113e+04,
             3.36895113e+04,  3.36895113e+04, -9.31343873e+04, -5.41397262e+01,
            -4.94188476e+07, -2.33542671e-07,  2.18511388e-01, -2.78803397e+00,
             4.45158510e-02, -4.27741500e+00,  1.42934253e-01, -5.25401120e+00,
             6.90837849e-01,  2.79001996e-01, -2.36678155e-05,  4.73746389e-03,
            -1.01499492e-01,  2.58879026e-06,  0.00000000e+00, -4.09098387e+07,
            -9.60108982e+07, -2.13668341e+08, -3.30594643e+08, -3.10962171e+09,
             9.86796183e+08, -1.67955312e+01, -1.17565997e-03, -1.27318005e-01,
            -3.61927012e-06, -1.66830321e+09]),
     array([ 0.00000000e+00,  1.04733005e+00,  0.00000000e+00,  0.00000000e+00,
             2.66731465e+07,  9.28978962e+00,  0.00000000e+00,  3.36895113e+04,
             3.36895113e+04,  3.36895113e+04, -7.96664510e+04, -5.37072130e+01,
            -4.90223915e+07, -2.20753852e-07,  2.55907360e-01, -3.14774696e+00,
             5.95058583e-02, -4.25154440e+00,  1.13091416e-01, -5.30044132e+00,
             6.65957055e-01,  1.76085656e-01, -5.18387234e-05,  3.87832060e-03,
            -8.22107277e-02,  3.17105444e-06,  0.00000000e+00, -3.56423594e+07,
            -9.03028987e+07, -2.16698524e+08, -3.37946545e+08, -2.90247866e+09,
             1.01265819e+09, -1.66296266e+01, -1.29909239e-03, -1.08896152e-01,
            -3.73793431e-06, -1.76017867e+09])]




```python
# Fourth moment of the timeseries for each channel
mycruncher.fourth_moments()
```




    [array([3.21601332e+04, 3.44360404e+00, 0.00000000e+00, 0.00000000e+00,
            2.43097390e+12, 2.55095341e+04, 0.00000000e+00, 3.39569151e+06,
            3.39569151e+06, 3.39569151e+06, 1.58498572e+08, 5.15307667e+02,
            4.55459785e+10, 3.06556334e-09, 2.37642092e-01, 3.28922739e+01,
            2.23991186e-01, 3.73738647e+01, 4.37451084e-01, 4.61649792e+01,
            1.55761531e+00, 2.79764035e+00, 2.57913196e-05, 9.06909402e-03,
            8.98454452e-02, 3.00027025e-07, 0.00000000e+00, 3.02030150e+10,
            4.03019005e+11, 5.56559180e+11, 4.31031003e+12, 1.85725589e+13,
            1.27638339e+13, 1.27548618e+02, 2.80917505e-03, 1.38923674e-01,
            1.23362554e-06, 4.61511943e+13]),
     array([3.21601332e+04, 3.45256143e+00, 0.00000000e+00, 0.00000000e+00,
            2.76447638e+12, 2.77911904e+04, 0.00000000e+00, 3.39569151e+06,
            3.39569151e+06, 3.39569151e+06, 1.57628652e+08, 5.41849680e+02,
            4.80084271e+10, 4.57357607e-09, 3.25917050e-01, 3.38973659e+01,
            2.17601724e-01, 4.22909594e+01, 5.18838858e-01, 5.02612284e+01,
            1.72886585e+00, 4.99632383e+00, 3.70699371e-05, 1.18012360e-02,
            1.51218760e-01, 4.44174684e-07, 0.00000000e+00, 4.01343707e+10,
            4.22805166e+11, 5.29887978e+11, 4.68393307e+12, 1.85493620e+13,
            1.54127659e+13, 1.34722411e+02, 3.17693880e-03, 1.84043401e-01,
            1.33672552e-06, 4.69717784e+13]),
     array([3.21601332e+04, 3.45256143e+00, 0.00000000e+00, 0.00000000e+00,
            2.61309287e+12, 2.68711060e+04, 0.00000000e+00, 3.39569151e+06,
            3.39569151e+06, 3.39569151e+06, 1.57798668e+08, 5.37554869e+02,
            4.76306262e+10, 4.16305670e-09, 3.96417218e-01, 3.49011375e+01,
            2.37648630e-01, 4.14267338e+01, 4.77039025e-01, 5.00044956e+01,
            1.67160143e+00, 3.13727175e+00, 4.24131240e-05, 5.27286681e-03,
            1.02255753e-01, 4.65810260e-07, 0.00000000e+00, 3.34086922e+10,
            4.14771141e+11, 5.39756925e+11, 4.59734736e+12, 1.72790574e+13,
            1.34023646e+13, 1.34010005e+02, 3.08808609e-03, 1.49276924e-01,
            1.31399086e-06, 4.40840636e+13])]




```python
# Mean of channel values over time
mycruncher.means()
```




    [[60.0,
      9.474406991260869,
      0.0,
      0.0,
      0.0076519825218477056,
      0.0033865418227215756,
      0.0,
      0.002866609238451929,
      0.01073330711610486,
      0.0,
      -0.0011934993757802812,
      1235.3682896379526,
      12.588014981273426,
      0.0,
      31.882022471910112,
      31.882022471910112,
      31.882022471910112,
      179.93543645443197,
      7.176954299625444,
      695.867722596754,
      -0.0005170903333333338,
      0.25040255430711617,
      1.9359125018726577,
      -0.044236254681648,
      0.0,
      1.9034188451935086,
      0.1226722996254682,
      0.0,
      1.8516907228464454,
      -0.2979419600499377,
      0.0,
      47.88283395755306,
      55.30990012484407,
      56.599937578027685,
      -0.0243848314606742,
      -0.001627328838951311,
      -0.05436598501872664,
      0.14717891385767798,
      -0.01380261154806491,
      0.0,
      32.518601747815296,
      -0.049951031423220875,
      -0.13459880898876392,
      0.04102214631710354,
      0.4493361922596746,
      -0.24992296504369546,
      -0.06272093196004998,
      -4.7685468164793874e-05,
      0.003685510611735314,
      106.16842821473152,
      -9.165780274656669,
      321.03495630461924,
      287.6853932584268,
      3542.421972534333,
      -6.291830586766546,
      100.94622222222222,
      17.413308988764058,
      262.9173046192261,
      -267.7313233458178,
      3431.79088639201,
      -4.1369833957553,
      102.34585767790254,
      -45.691667915106116,
      268.0115960474407,
      1009.1048689138577,
      3440.648314606741,
      -1.692430711610488,
      46.18862921348314,
      879.3237952559303,
      -8.096102621722837,
      35.18141448189762,
      809.1064669163546,
      -5.59885120099875,
      -6.771250936329537,
      992.3348189762796,
      -6.774709113607989,
      327.7384644194755,
      24.464835205992472,
      228.26409987515603,
      1087.7556029962536,
      603.6278152309613,
      -64.03380774032458,
      267.14305867665445,
      -5.220037203495644,
      -3426.267166042447,
      1096.4189662921344,
      -373.2260549313359,
      -81.4319600499376,
      301.16474406991256,
      -6.267034706616722,
      -6830.574282147316,
      1582.7207240948812,
      24384.92334581773,
      -82.6205405742822,
      1341.1046192259678,
      12259.833957553059,
      -81.01440199750307,
      206.8501872659176,
      82.26067415730314,
      0.0,
      0.0,
      239.9812734082397,
      75.26054931335811,
      0.0,
      0.0,
      344.18476903870163,
      61.4156054931336,
      0.0,
      0.0,
      534.8826466916355,
      48.70661672908852,
      176.12229463171042,
      0.0,
      533.1922596754057,
      48.81697877652923,
      174.3412871410737,
      0.0,
      343.9250936329588,
      61.428714107365856,
      0.0,
      0.0,
      239.37702871410735,
      75.35805243445704,
      0.0,
      0.0,
      206.93882646691637,
      82.24594257178501,
      0.0,
      0.0,
      5.012502546816474,
      0.19819502758052412,
      0.49749073657927567,
      0.02597667248439455,
      5142.646327988027,
      5333.483915784192,
      5572.51262075349],
     [60.0,
      9.474033707865114,
      0.0,
      0.0,
      0.03283327340823969,
      0.013122361660424454,
      0.0,
      -0.01284832734082398,
      -0.00287691510611736,
      0.0,
      -0.004702302197253443,
      1275.729088639201,
      12.860911360798994,
      0.0,
      31.882022471910112,
      31.882022471910112,
      31.882022471910112,
      182.67509001248496,
      7.259319892759066,
      704.2122874244691,
      -0.0005357460174781524,
      0.2889659013732835,
      1.9781520831460653,
      -0.023996333707865237,
      0.0,
      1.9380342896379557,
      0.10173368672908836,
      0.0,
      1.8851081756554322,
      -0.30061906741573047,
      0.0,
      48.651730337078654,
      55.503077403245854,
      56.17011985018718,
      -0.014246950099875253,
      0.0007398029463171062,
      -0.07472525905118595,
      0.14783562883895113,
      -0.014548521847690368,
      0.0,
      32.2991760299625,
      -0.046484241985018694,
      -0.1091923223470663,
      0.045484814731585485,
      0.4295111036204734,
      -0.28842948189762785,
      -0.06103158052434459,
      -8.631790012484369e-05,
      0.0032408079900124844,
      107.52211273408261,
      -4.5598420724094835,
      326.6009987515607,
      200.88243196004956,
      3604.7809941323358,
      -6.943665630461931,
      104.16716101123606,
      13.065028714107347,
      266.2573255930088,
      -180.79640074906365,
      3520.8263295880147,
      -2.5269280024968785,
      103.48089563046182,
      -46.96173807740324,
      279.98964918851436,
      1032.3938451935082,
      3490.4889076154823,
      -1.1822389263420725,
      44.306890636704104,
      894.4999525592999,
      -7.4003844519350785,
      37.584518976279654,
      828.5431883270904,
      -7.0282006117353255,
      -4.671125468164779,
      1013.236978776528,
      -5.503523558052432,
      331.7733695380775,
      55.610321598002464,
      218.82730212234702,
      1112.6716052808983,
      573.5697153558052,
      -95.2311335830213,
      268.9428252184772,
      -6.095474968789013,
      -3416.710362047441,
      1122.8166782147314,
      -350.3167642946318,
      -84.42491498127339,
      299.610921847691,
      -7.5480345817727965,
      -6818.077403245942,
      1705.6204207240949,
      24422.837328339574,
      -85.26811260923832,
      1413.8332584269667,
      12327.005118601748,
      -84.04151485642937,
      207.51960049937594,
      82.1396129837704,
      0.0,
      0.0,
      240.35493133583023,
      75.19784019975032,
      0.0,
      0.0,
      344.11810237203514,
      61.44808988764045,
      0.0,
      0.0,
      537.2176029962544,
      48.78640449438206,
      183.4595096129838,
      0.0,
      535.7464419475655,
      48.90007490636703,
      182.53712958801518,
      0.0,
      343.965792759051,
      61.44862671660423,
      0.0,
      0.0,
      239.6947565543071,
      75.30254681647932,
      0.0,
      0.0,
      207.5868913857677,
      82.11802746566785,
      0.0,
      0.0,
      5.072975640823974,
      0.20489618869696627,
      0.5048609113607996,
      0.02658118928963794,
      5244.4318210422,
      5454.1970474040645,
      5658.68218172676],
     [60.0,
      9.474033707865114,
      0.0,
      0.0,
      0.00866469313358312,
      0.006198385018726631,
      0.0,
      -0.010118690012484402,
      -0.005855488514357055,
      0.0,
      -0.003956569912609255,
      1262.9076154806492,
      12.774282147315851,
      0.0,
      31.882022471910112,
      31.882022471910112,
      31.882022471910112,
      181.64068002496833,
      7.23994442046193,
      702.3365064544334,
      -0.0005873394681647938,
      0.31189457128589265,
      1.9669964334581787,
      -0.027973770037453532,
      0.0,
      1.926760839201001,
      0.11169044282147292,
      0.0,
      1.8813252521847679,
      -0.30336131061173544,
      0.0,
      48.34194382022473,
      55.44149937578016,
      56.28647690387004,
      -0.022014284519350856,
      0.0014400059925093638,
      -0.058541762734082355,
      0.14887484019975034,
      -0.014425856004993782,
      0.0,
      32.44052434456931,
      -0.04170461036204748,
      -0.1335427977528088,
      0.04495726179775274,
      0.44528449438202145,
      -0.31130436454431964,
      -0.061294516729088665,
      -7.97915505617981e-05,
      0.002564930137328339,
      107.0317183770287,
      -5.9075274656679095,
      325.8791510611737,
      225.5651660424467,
      3586.1991310861436,
      -6.020777448189766,
      103.56011985018746,
      14.704978776529334,
      265.8601298377028,
      -213.71460799001244,
      3501.6539325842728,
      -3.3031430461922606,
      103.31495518102376,
      -46.89713520599254,
      276.7788125218477,
      1033.3007365792757,
      3485.2263245942568,
      -0.4796607240948807,
      45.90145011235957,
      892.593811485643,
      -7.011572072409482,
      37.38971672908864,
      818.9630746566805,
      -7.0211141073658005,
      -6.003073033707897,
      1012.9851186017486,
      -5.273469475655436,
      331.34439750312134,
      45.74713233458185,
      223.8239712858927,
      1104.7896159800246,
      579.2107016229713,
      -86.09002122347052,
      270.2588002496876,
      -6.0232438701623,
      -3423.9225967540574,
      1113.9927478152322,
      -351.0959385767792,
      -92.59016342072411,
      302.54099363295865,
      -7.479307965043686,
      -6827.042446941324,
      1690.835315855181,
      24614.2802372035,
      -93.5909787765293,
      1401.759269975031,
      12408.918352059927,
      -92.17196891385774,
      207.2119850187265,
      82.19675405742828,
      0.0,
      0.0,
      240.12347066167277,
      75.22797752808991,
      0.0,
      0.0,
      344.33308364544325,
      61.42038701622972,
      0.0,
      0.0,
      538.7612983770289,
      48.69012484394514,
      185.57489695380772,
      0.0,
      537.1769038701627,
      48.80917602996255,
      184.52663333333345,
      0.0,
      344.27640449438246,
      61.41054931335828,
      0.0,
      0.0,
      239.46741573033728,
      75.33284644194752,
      0.0,
      0.0,
      207.29750312109869,
      82.17106117353298,
      0.0,
      0.0,
      5.0601133992384515,
      0.20273726108163578,
      0.503395828339576,
      0.026388878302122346,
      5231.190499175547,
      5422.4812892524,
      5648.792241896808]]




```python
# Median of channel values over time
mycruncher.medians()
```




    [[60.0,
      9.4,
      0.0,
      0.0,
      -0.0152,
      -0.0276,
      0.0,
      -0.0129,
      -0.00186,
      0.0,
      0.0605,
      2300.0,
      24.0,
      0.0,
      0.0,
      0.0,
      0.0,
      192.0,
      10.0,
      974.0,
      9.87e-06,
      -0.0882,
      3.34,
      -0.0895,
      0.0,
      3.4,
      0.0527,
      0.0,
      3.41,
      -0.479,
      0.0,
      53.8,
      63.8,
      63.4,
      -0.076,
      0.00191,
      -0.0533,
      0.278,
      -0.0154,
      0.0,
      34.1,
      -0.0382,
      -0.0987,
      0.0517,
      1.07,
      0.0909,
      -0.049,
      -2.78e-05,
      0.0108,
      157.0,
      11.2,
      281.0,
      -162.0,
      5850.0,
      -2.82,
      167.0,
      60.5,
      343.0,
      -1170.0,
      6220.0,
      -0.873,
      170.0,
      -85.7,
      300.0,
      1780.0,
      6190.0,
      3.76,
      -6.56,
      1620.0,
      -2.4,
      38.2,
      1650.0,
      -2.35,
      -12.2,
      1650.0,
      -2.56,
      463.0,
      145.0,
      564.0,
      2320.0,
      1160.0,
      -312.0,
      472.0,
      -7.19,
      -3410.0,
      2340.0,
      121.0,
      0.842,
      572.0,
      -5.34,
      -6820.0,
      1720.0,
      45900.0,
      -0.389,
      1460.0,
      23300.0,
      1.31,
      205.0,
      82.7,
      0.0,
      0.0,
      238.0,
      75.5,
      0.0,
      0.0,
      344.0,
      61.1,
      0.0,
      0.0,
      547.0,
      47.6,
      192.0,
      0.0,
      545.0,
      47.7,
      189.0,
      0.0,
      344.0,
      61.2,
      0.0,
      0.0,
      238.0,
      75.5,
      0.0,
      0.0,
      205.0,
      82.7,
      0.0,
      0.0,
      7.04,
      0.383,
      0.693,
      0.0547,
      6147.974788497429,
      6443.5992279160255,
      6352.441055216491],
     [60.0,
      9.4,
      0.0,
      0.0,
      0.117,
      0.02552,
      0.0,
      -0.03231,
      0.01418,
      0.0,
      0.004886,
      2398.0,
      24.65,
      0.0,
      0.0,
      0.0,
      0.0,
      196.9,
      10.17,
      986.4,
      0.000154,
      -0.1408,
      3.457,
      -0.08416,
      0.0,
      3.519,
      0.03997,
      0.0,
      3.513,
      -0.5135,
      0.0,
      53.72,
      63.82,
      63.2,
      -0.0305,
      -0.0003637,
      -0.07139,
      0.292,
      -0.01592,
      0.0,
      35.66,
      -0.03485,
      -0.06208,
      0.04558,
      0.9622,
      0.1401,
      -0.06272,
      4.552e-05,
      0.002535,
      166.0,
      19.35,
      291.0,
      -359.3,
      6095.0,
      -2.791,
      174.1,
      45.68,
      351.7,
      -855.8,
      6495.0,
      -0.4677,
      184.2,
      -88.95,
      311.2,
      1898.0,
      6467.0,
      3.637,
      -9.077,
      1676.0,
      -1.928,
      56.65,
      1698.0,
      -3.582,
      -8.165,
      1705.0,
      -1.605,
      498.0,
      185.5,
      581.2,
      2391.0,
      1091.0,
      -331.5,
      513.9,
      -6.541,
      -3413.0,
      2370.0,
      247.6,
      24.38,
      563.1,
      -7.612,
      -6800.0,
      1968.0,
      47190.0,
      22.74,
      1529.0,
      24460.0,
      24.06,
      203.4,
      83.11,
      0.0,
      0.0,
      237.4,
      75.71,
      0.0,
      0.0,
      352.7,
      60.5,
      0.0,
      0.0,
      575.9,
      46.4,
      233.4,
      0.0,
      576.9,
      46.37,
      234.8,
      0.0,
      351.9,
      60.55,
      0.0,
      0.0,
      237.5,
      75.68,
      0.0,
      0.0,
      203.2,
      83.13,
      0.0,
      0.0,
      7.13,
      0.4012,
      0.7469,
      0.05634,
      6430.205455864467,
      6652.936625328998,
      6545.911792286847],
     [60.0,
      9.4,
      0.0,
      0.0,
      0.06227,
      0.05863,
      0.0,
      -0.03541,
      -0.002971,
      0.0,
      -0.01397,
      2371.0,
      24.48,
      0.0,
      0.0,
      0.0,
      0.0,
      195.4,
      10.16,
      985.9,
      0.0001695,
      -0.1488,
      3.561,
      -0.0827,
      0.0,
      3.523,
      0.0494,
      0.0,
      3.506,
      -0.492,
      0.0,
      54.02,
      63.79,
      63.26,
      -0.01255,
      0.003159,
      -0.0528,
      0.3088,
      -0.0157,
      0.0,
      35.88,
      -0.03186,
      -0.05205,
      0.04774,
      1.063,
      0.1502,
      -0.0637,
      1.755e-05,
      -0.006538,
      178.7,
      15.65,
      287.2,
      -296.6,
      6470.0,
      -2.425,
      181.9,
      51.77,
      348.6,
      -986.7,
      6594.0,
      -0.8536,
      186.6,
      -86.68,
      309.6,
      1826.0,
      6491.0,
      3.89,
      -11.05,
      1729.0,
      -1.842,
      47.17,
      1700.0,
      -3.31,
      -14.86,
      1703.0,
      -1.428,
      541.0,
      183.8,
      568.3,
      2379.0,
      1056.0,
      -311.2,
      540.6,
      -6.471,
      -3433.0,
      2355.0,
      355.7,
      26.79,
      595.8,
      -6.576,
      -6829.0,
      1972.0,
      49940.0,
      24.26,
      1511.0,
      25860.0,
      27.34,
      204.1,
      83.11,
      0.0,
      0.0,
      238.6,
      75.63,
      0.0,
      0.0,
      353.3,
      60.45,
      0.0,
      0.0,
      579.9,
      46.21,
      239.1,
      0.0,
      580.3,
      46.2,
      239.7,
      0.0,
      352.5,
      60.49,
      0.0,
      0.0,
      237.9,
      75.61,
      0.0,
      0.0,
      204.0,
      83.13,
      0.0,
      0.0,
      7.127,
      0.3976,
      0.8075,
      0.05607,
      6669.062226850188,
      6753.521723878587,
      6566.748865522116]]




```python
# Standard deviation of channel values over time
mycruncher.stddevs()
```




    [[11.561430130683084,
      0.7331873733878951,
      0.0,
      0.0,
      0.4221677852002905,
      0.2988545890176859,
      0.0,
      0.29680050948971887,
      0.2804489345312724,
      0.0,
      0.28325223310950853,
      1238.4834854625656,
      12.593300487097336,
      0.0,
      38.38885575394094,
      38.38885575394094,
      38.38885575394094,
      91.3355614638876,
      4.161808318042927,
      403.51762400937525,
      0.00432985168909129,
      0.6006158033858908,
      2.181793358239024,
      0.5756125282671347,
      0.0,
      2.1996939568646208,
      0.6730876332012791,
      0.0,
      2.285046491763002,
      0.7559565975672955,
      0.0,
      17.66320923227619,
      15.782304428494266,
      14.566749967488587,
      1.0556042475104248,
      0.05583823531576016,
      0.2553378779299062,
      0.4314254102373269,
      0.01965765316500327,
      0.0,
      5.005030282853597,
      0.05034601556996795,
      0.3756090844194291,
      0.07129516531590052,
      2.709583970822169,
      0.6013534715670306,
      0.15137751063909416,
      0.004025696466459002,
      0.1558485086064321,
      104.68076582462815,
      117.13564128733154,
      170.98398133797983,
      2388.3637772507086,
      3929.2791821500177,
      28.374588525336335,
      110.35523498386983,
      121.39353288010066,
      266.20546159099746,
      2545.7712872014504,
      4065.9159106790994,
      27.703435754461644,
      109.07160925562953,
      121.4659179299338,
      240.99424478608455,
      2505.9684559788084,
      4070.3852649158703,
      28.781307260069106,
      369.1045153617038,
      1128.1051547516993,
      14.156331301568944,
      410.41441841686344,
      1195.0740911631435,
      12.116972867394031,
      338.34982874288994,
      1083.471107304341,
      13.206763512190289,
      350.9525949048192,
      707.1822733554046,
      763.0170550199543,
      1390.9887496023634,
      1645.8116083114671,
      1522.3326046923735,
      656.9720746786908,
      21.33428132281557,
      79.27684095115445,
      1424.9737891385917,
      1914.4149637134717,
      682.1491503373102,
      895.8908523103685,
      25.664838684573144,
      129.74447099230906,
      2738.344179442863,
      71207.06477914946,
      682.0287835883779,
      1928.3427894371289,
      35544.42968948133,
      682.2403300129915,
      7.1250663068873195,
      1.5240032193791366,
      0.0,
      0.0,
      5.66531368526615,
      0.8629373510214501,
      0.0,
      0.0,
      15.900150918428556,
      1.543766383188357,
      0.0,
      0.0,
      70.52580272442043,
      3.7263871528135644,
      96.96459761863547,
      0.0,
      72.99729264907616,
      3.892797069527047,
      99.43777004188988,
      0.0,
      15.143570330985696,
      1.4691754061192963,
      0.0,
      0.0,
      5.003386784097885,
      0.7400868473160634,
      0.0,
      0.0,
      7.370691572253048,
      1.5810477544007806,
      0.0,
      0.0,
      2.955963716621018,
      0.22129355075252238,
      0.5119489089002696,
      0.032369422078765905,
      2707.224813418054,
      2533.0271028814677,
      2156.8963554119446],
     [11.561430130683084,
      0.7332269535454926,
      0.0,
      0.0,
      0.5557081647522517,
      0.2750502366587759,
      0.0,
      0.2782350398857892,
      0.16575337228547643,
      0.0,
      0.16449329953901878,
      1278.9502523867716,
      12.866195690799154,
      0.0,
      38.38885575394094,
      38.38885575394094,
      38.38885575394094,
      91.42681807774244,
      4.211478083157693,
      408.6111036426619,
      0.004726333155580127,
      0.6542656916910718,
      2.2090018254733588,
      0.5844826750005133,
      0.0,
      2.2520967516055768,
      0.6861759933264066,
      0.0,
      2.3295444215175354,
      0.7644012380329501,
      0.0,
      17.21825084770148,
      15.698434678012527,
      14.787749157298895,
      1.1630762103914263,
      0.060778057589886726,
      0.27383901928229043,
      0.4823974065213812,
      0.020663674919107602,
      0.0,
      6.376284914320325,
      0.05733790855195877,
      0.5402200588390991,
      0.08947456367420155,
      3.2281039736688175,
      0.6550201185769958,
      0.18964630370028732,
      0.005259815318994018,
      0.1361427711027469,
      107.05487029937424,
      118.91121675346052,
      175.37300522115956,
      2421.617244363199,
      3993.1777369600077,
      30.426512448119396,
      116.7443554665705,
      120.48110208383315,
      272.7134073761195,
      2533.4652810193147,
      4223.196014850622,
      28.084347220880872,
      111.66405368479433,
      120.26571395908634,
      241.51843734535362,
      2484.9508761836482,
      4153.396141601142,
      28.934403133102457,
      379.90627970863954,
      1154.3223056678323,
      14.862399402193326,
      417.60052903402385,
      1207.4251376580796,
      12.399571516391175,
      343.61615942881394,
      1110.3788053357905,
      13.529524676198456,
      372.6541613265185,
      716.9445658600292,
      750.4910466651899,
      1419.9791220731713,
      1642.7694108862381,
      1589.8987518915833,
      731.7096998044608,
      25.497735626685316,
      83.05252165528827,
      1453.8559212795142,
      2113.6987202111695,
      744.764208988474,
      1005.4251332983663,
      32.96854283904926,
      127.99821121272115,
      3082.359826125036,
      79712.61956877731,
      744.7107478981657,
      2025.5516010288468,
      39715.263032599956,
      744.8134572570991,
      9.45192066909056,
      2.001751286939204,
      0.0,
      0.0,
      7.250866151818676,
      1.1129767777439081,
      0.0,
      0.0,
      19.539771136694867,
      1.9219452073836238,
      0.0,
      0.0,
      84.23690421784822,
      4.659155223528575,
      109.78323839729504,
      0.0,
      87.1225290703267,
      4.862283310918592,
      112.08430708179866,
      0.0,
      18.778033981270386,
      1.8466428997606135,
      0.0,
      0.0,
      6.455289228464607,
      0.9604150193308523,
      0.0,
      0.0,
      9.764873246107578,
      2.069080348218857,
      0.0,
      0.0,
      2.99366728934785,
      0.2283555170258708,
      0.5425819340067154,
      0.033035029984337626,
      2709.5031178927015,
      2633.4893156489547,
      2157.5877260206257],
     [11.561430130683084,
      0.7332269535454926,
      0.0,
      0.0,
      0.6465259604703142,
      0.2645434303048697,
      0.0,
      0.2560997420193159,
      0.1286256316061192,
      0.0,
      0.12494472579220692,
      1264.5743912343526,
      12.772946393289061,
      0.0,
      38.38885575394094,
      38.38885575394094,
      38.38885575394094,
      91.38410743040492,
      4.205078609375974,
      407.9946851424283,
      0.004368922113818549,
      0.6853541169675764,
      2.2194620827693443,
      0.5882163011423742,
      0.0,
      2.2442156785738376,
      0.6805722143462501,
      0.0,
      2.324377346295555,
      0.7612285941499476,
      0.0,
      17.38011127167545,
      15.725666594472607,
      14.798080624343562,
      1.0126201538506312,
      0.06199170488723658,
      0.22088663425311134,
      0.43542678995777606,
      0.020750921315114013,
      0.0,
      6.254685911770509,
      0.055564676116708456,
      0.6371856558233399,
      0.09353792222876663,
      2.8946203602092373,
      0.686051308717175,
      0.17153662218197654,
      0.004981437631780133,
      0.11329077764571914,
      106.18812566248229,
      118.16146057072811,
      174.55826002970358,
      2407.7010021216424,
      3998.1794013396247,
      29.48605245008144,
      114.64833020852673,
      120.89643539843621,
      271.18201107182966,
      2539.6422384183797,
      4188.862918181703,
      27.8487072509521,
      110.86137601616353,
      120.80227414668005,
      242.4984082967727,
      2493.5310354879093,
      4142.492973651154,
      29.125360263818262,
      374.81594892509355,
      1157.6084026322044,
      14.535280450869354,
      416.15116831560806,
      1206.4820143537884,
      12.4195290408057,
      342.6430812500062,
      1105.9580875537217,
      13.455340409653093,
      359.99097769688285,
      713.3013413779325,
      755.9308592108007,
      1413.6604249692273,
      1622.2463677255937,
      1538.9902022831586,
      664.6725421270559,
      25.913244482801655,
      69.60940787112197,
      1446.2796706407137,
      1929.4499773350512,
      688.419076086655,
      901.2018812806789,
      33.36957214113764,
      104.79290889088216,
      3111.807979236976,
      71781.46958180134,
      688.2176359417301,
      2035.4340154422196,
      35891.60515811243,
      688.500506470746,
      9.075365191719364,
      1.938781346439049,
      0.0,
      0.0,
      6.92953284195198,
      1.0693369686650185,
      0.0,
      0.0,
      19.337554533473856,
      1.902123191131009,
      0.0,
      0.0,
      82.94757741828663,
      4.586511160850778,
      107.96324604915868,
      0.0,
      85.88142613050412,
      4.79338444131304,
      110.21377197860907,
      0.0,
      18.461233264306344,
      1.8148093650816797,
      0.0,
      0.0,
      6.186750325390249,
      0.921361359184171,
      0.0,
      0.0,
      9.401461822626302,
      2.0092497087254007,
      0.0,
      0.0,
      2.9906742784629037,
      0.22665834400825344,
      0.5229240607343845,
      0.03289360947611398,
      2707.3452183337304,
      2626.9930667370977,
      2164.3596586801423]]




```python
# Skew of channel values over time
mycruncher.skews()
```

    /Users/gbarter/devel/pCrunch/pCrunch/aeroelastic_output.py:363: RuntimeWarning: invalid value encountered in divide
      return self.third_moments / np.sqrt(self.second_moments) ** 3





    [array([ 0.        ,  2.65336626,         nan,         nan,  0.02078122,
             0.00772224,         nan,  0.59549649,  0.59549649,  0.59549649,
            -0.06827807, -0.72129051, -0.7212567 , -2.03223151,  0.78194229,
            -0.25720616,  0.3101788 , -0.35171318,  0.28902529, -0.38938752,
             1.44900917,  0.17504459, -0.01904679,  0.32174313, -0.82483607,
             0.24769592,         nan, -0.67714933, -0.21768293, -0.50312488,
            -0.11448448, -0.74747688,  0.32644179, -0.62494164, -0.09917534,
            -0.67932975, -0.10039556, -0.29541754]),
     array([ 0.        ,  2.65685773,         nan,         nan,  0.02078847,
             0.00766576,         nan,  0.59549649,  0.59549649,  0.59549649,
            -0.12186788, -0.72479038, -0.72437278, -2.21204139,  0.78021022,
            -0.25864842,  0.22294603, -0.37447278,  0.44241491, -0.41560236,
             1.54672098,  0.17733024, -0.10541872,  0.23070655, -0.90416767,
             0.29340951,         nan, -0.7905145 , -0.26053364, -0.5054796 ,
            -0.11546489, -0.70141934,  0.24553819, -0.62601271, -0.09872956,
            -0.79706518, -0.10039149, -0.20046128]),
     array([ 0.        ,  2.65685773,         nan,         nan,  0.01318989,
             0.00445792,         nan,  0.59549649,  0.59549649,  0.59549649,
            -0.10439109, -0.72228777, -0.72182343, -2.64719164,  0.79494613,
            -0.28790992,  0.2923806 , -0.37614296,  0.35876246, -0.42207743,
             1.50973577,  0.16958377, -0.21759733,  0.35986151, -0.99582507,
             0.35488819,         nan, -0.7639969 , -0.24881844, -0.5016603 ,
            -0.11962245, -0.67985861,  0.27781476, -0.62169182, -0.11156421,
            -0.76154691, -0.10502606, -0.22341377])]




```python
# Kurtosis of channel values over time
mycruncher.kurtosis()
```

    /Users/gbarter/devel/pCrunch/pCrunch/aeroelastic_output.py:367: RuntimeWarning: invalid value encountered in divide
      return self.fourth_moments / self.second_moments ** 2





    [array([ 1.79999626, 11.91662575,         nan,         nan,  1.03328332,
             1.01424824,         nan,  1.56353482,  1.56353482,  1.56353482,
             2.27753858,  1.71766437,  1.71790841,  8.72204274,  1.82614957,
             1.45157486,  2.04037492,  1.59631635,  2.13129248,  1.69329444,
             4.76949898,  2.25313579,  2.65305849,  2.13355049,  2.59342195,
             2.00924873,         nan,  1.99093218,  1.61138777,  1.6420046 ,
             1.15136769,  2.53134453,  2.37652806,  1.67062648,  1.1713961 ,
             2.02241265,  1.12368238,  2.78849312]),
     array([ 1.79999626, 11.94504332,         nan,         nan,  1.03323302,
             1.01416014,         nan,  1.56353482,  1.56353482,  1.56353482,
             2.25600854,  1.72242627,  1.72217318,  9.16555105,  1.77864998,
             1.42357926,  1.86455742,  1.64399072,  2.34040176,  1.70666827,
             5.06379094,  2.73034727,  2.71665559,  2.09867975,  2.79245263,
             2.43625957,         nan,  2.08109812,  1.60028736,  1.67032893,
             1.15208055,  2.54696248,  2.4121411 ,  1.67735782,  1.1683227 ,
             2.12353261,  1.12238904,  2.78512275]),
     array([ 1.79999626, 11.94504332,         nan,         nan,  1.02182686,
             1.00953455,         nan,  1.56353482,  1.56353482,  1.56353482,
             2.26266696,  1.71919969,  1.71896989, 11.42654978,  1.79676966,
             1.43829729,  1.98512149,  1.6331359 ,  2.22360198,  1.71309921,
             4.97820023,  2.98377363,  2.8718777 ,  2.21497613,  2.84463906,
             2.51223068,         nan,  1.98926669,  1.60219868,  1.65298885,
             1.15113675,  2.49489807,  2.38912024,  1.67517723,  1.17004497,
             1.99635686,  1.12239627,  2.81202515])]




```python
# Integration of channel values over time
mycruncher.integrated()
```




    [[2400.0,
      378.97999999999786,
      0.0,
      0.0,
      0.3185869000000005,
      0.14550599999999986,
      0.0,
      0.1238327000000003,
      0.4336164500000013,
      0.0,
      -0.05739965000000015,
      49418.5000000001,
      503.5474999999994,
      0.0,
      1274.6250000000027,
      1274.6250000000027,
      1274.6250000000027,
      7193.41423,
      287.1878172,
      27845.280289999842,
      -0.020711242850000006,
      10.003839800000002,
      77.43830802500003,
      -1.7891495000000004,
      0.0,
      76.13219975000014,
      4.92485059999999,
      0.0,
      74.07748844999993,
      -11.894325499999992,
      0.0,
      1915.0549999999994,
      2211.9290000000015,
      2265.0244999999986,
      -1.0179124999999978,
      -0.0663082699999994,
      -2.1841702000000014,
      5.90296549999999,
      -0.5518263425000008,
      0.0,
      1301.1575000000018,
      -1.998296308500001,
      -5.37945729999999,
      1.6409869600000027,
      18.096164499999983,
      -9.984624749999996,
      -2.503213324999998,
      -0.0017465529999999578,
      0.14546470000000017,
      4247.790550000006,
      -373.28950000000145,
      12845.775000000003,
      11638.800000000048,
      141710.67499999955,
      -252.9578150000002,
      4039.1762000000026,
      700.2380250000007,
      10525.91304999999,
      -10789.389499999954,
      137291.47499999971,
      -164.61218500000027,
      4095.034099999997,
      -1825.0463000000018,
      10718.451921700002,
      40307.95000000012,
      137653.18999999968,
      -67.60385,
      1860.062100000001,
      35178.7429999998,
      -324.39865999999967,
      1394.0406500000024,
      32357.76399999987,
      -223.76649059999963,
      -270.6436000000014,
      39687.209499999786,
      -271.0613499999999,
      13116.400499999996,
      938.341650000001,
      9134.002199999997,
      43506.04314999992,
      24142.068999999934,
      -2483.303999999993,
      10708.779500000006,
      -208.3732400000001,
      -137049.2499999993,
      43851.38709999997,
      -14869.82850000002,
      -3261.624999999996,
      12081.422999999995,
      -250.1499899999996,
      -273220.49999999854,
      63259.41500000002,
      978053.6799999999,
      -3309.1901500000045,
      53616.86500000014,
      491683.8499999996,
      -3244.919299999999,
      8273.37499999999,
      3290.5600000000027,
      0.0,
      0.0,
      9598.774999999981,
      3010.4925000000017,
      0.0,
      0.0,
      13768.749999999967,
      2456.4900000000034,
      0.0,
      0.0,
      21400.924999999963,
      1947.947500000003,
      7051.540400000006,
      0.0,
      21333.34999999996,
      1952.3600000000017,
      6980.23355,
      0.0,
      13758.324999999968,
      2457.0200000000013,
      0.0,
      0.0,
      9574.649999999994,
      3014.390000000004,
      0.0,
      0.0,
      8276.900000000001,
      3289.975000000002,
      0.0,
      0.0,
      200.57578450000014,
      7.928010890849989,
      19.910353999999995,
      1.0389841154999977,
      205721.34172072908,
      213361.9521229238,
      222926.62068151266],
     [2400.0,
      378.96504999999746,
      0.0,
      0.0,
      1.319167600000003,
      0.530118834500001,
      0.0,
      -0.5039730100000003,
      -0.1100756999999994,
      0.0,
      -0.1925219529999998,
      51033.00000000001,
      514.4632500000001,
      0.0,
      1274.6250000000027,
      1274.6250000000027,
      1274.6250000000027,
      7303.404855000001,
      290.48297545500014,
      28179.172211350007,
      -0.021429528000000007,
      11.550784350000002,
      79.12952742999997,
      -0.9820056650000071,
      0.0,
      77.51794579999994,
      4.087346653499998,
      0.0,
      75.41282243499998,
      -11.998013649999987,
      0.0,
      1945.7567999999983,
      2219.6544999999987,
      2247.5897999999997,
      -0.6289273514999976,
      0.029151708000000786,
      -3.0017662000000023,
      5.935106935000008,
      -0.581702432499999,
      0.0,
      1292.3457500000009,
      -1.8593578915000002,
      -4.3658050100000025,
      1.819851830000002,
      17.345049699999965,
      -11.52932075000001,
      -2.4336200500000005,
      -0.0032839318999999417,
      0.12746785999999993,
      4302.261365000001,
      -189.61167499999854,
      13067.56749999999,
      8178.941399999981,
      144211.8613150001,
      -279.2013834999997,
      4168.579298500003,
      525.7666500000001,
      10660.485890000009,
      -7301.538349999994,
      140863.24449999968,
      -100.33969149999992,
      4140.756369999999,
      -1874.75261,
      11197.6677,
      41216.64850000001,
      139651.31575000004,
      -46.960244000000024,
      1785.133469999999,
      35786.84810000001,
      -296.60269230000034,
      1490.817485,
      33133.96694249998,
      -281.04890950000004,
      -190.66432499999956,
      40521.90100000001,
      -220.17104349999983,
      13280.388449999993,
      2180.92588,
      8750.80595,
      44502.38946650002,
      22926.844600000004,
      -3718.756900000003,
      10789.382650000003,
      -243.47789750000075,
      -136666.75000000015,
      44907.57783749994,
      -13930.881409999998,
      -3376.9708450000026,
      12031.752419999986,
      -301.4360349999992,
      -272720.3000000004,
      68181.99985000012,
      980548.6350000048,
      -3410.685660000001,
      56530.16700000008,
      494850.5550000024,
      -3361.6366699999953,
      8300.110000000002,
      3285.715250000002,
      0.0,
      0.0,
      9613.6925,
      3007.984999999998,
      0.0,
      0.0,
      13765.970000000003,
      2457.80175,
      0.0,
      0.0,
      21493.60250000002,
      1951.1709999999975,
      7343.310860000004,
      0.0,
      21434.760000000013,
      1955.7157499999973,
      7306.357040000011,
      0.0,
      13759.865000000002,
      2457.8245000000015,
      0.0,
      0.0,
      9587.302500000005,
      3012.1710000000026,
      0.0,
      0.0,
      8302.792500000005,
      3284.8527499999955,
      0.0,
      0.0,
      202.99534991499982,
      8.196019896663495,
      20.20908700000004,
      1.0631599173000015,
      209783.67374170545,
      218196.94679014338,
      226368.3448174983],
     [2400.0,
      378.96504999999746,
      0.0,
      0.0,
      0.3346287100000009,
      0.24648306999999942,
      0.0,
      -0.3997262850000001,
      -0.22972106500000075,
      0.0,
      -0.16016337500000002,
      50520.17499999997,
      510.99799999999993,
      0.0,
      1274.6250000000027,
      1274.6250000000027,
      1274.6250000000027,
      7261.979235000009,
      289.7080012895002,
      28104.14880850003,
      -0.023498758199999966,
      12.465597579999997,
      78.68295716000006,
      -1.1400762400000017,
      0.0,
      77.06649161000017,
      4.485052234999996,
      0.0,
      75.26183135000012,
      -12.108480489999994,
      0.0,
      1933.3918500000027,
      2217.190300000002,
      2252.283649999998,
      -0.9320778449999938,
      0.05718031500000082,
      -2.3503220974999963,
      5.973379850000001,
      -0.5768068255000001,
      0.0,
      1298.033499999998,
      -1.6682321449999986,
      -5.356510799999996,
      1.7990458350000018,
      17.942769,
      -12.441977299999998,
      -2.4472698950000007,
      -0.003057701599999953,
      0.10272345200000044,
      4282.461820999998,
      -243.35397499999942,
      13038.794999999986,
      9162.584900000005,
      143465.19269999984,
      -242.1996118000003,
      4144.021549999998,
      591.3696499999993,
      10644.453200000022,
      -8618.145050000001,
      140090.9400000001,
      -131.39225399999998,
      4133.961705000002,
      -1872.335765000001,
      11069.343941500012,
      41256.59449999998,
      139438.5218,
      -18.880237000000086,
      1849.243327000003,
      35710.274650000036,
      -281.0148365000004,
      1483.5806550000011,
      32751.28114,
      -280.7572700000003,
      -243.1158249999987,
      40512.32650000002,
      -210.96875250000002,
      13261.828120000011,
      1787.325149999995,
      8951.212550000004,
      44187.21902000005,
      23154.126099999994,
      -3357.9053499999964,
      10837.189949999984,
      -240.59971700000008,
      -136956.22500000018,
      44554.669100000036,
      -13974.667339999989,
      -3704.4522950000014,
      12141.904294999997,
      -298.7404089999998,
      -273080.7000000005,
      67593.41864999999,
      987655.1735000011,
      -3744.4679500000047,
      56048.17151250004,
      497857.68000000226,
      -3687.725354999998,
      8287.747500000005,
      3288.0125000000035,
      0.0,
      0.0,
      9604.39000000001,
      3009.197000000002,
      0.0,
      0.0,
      13774.550000000003,
      2456.6904999999974,
      0.0,
      0.0,
      21555.697500000024,
      1947.2999999999988,
      7428.564622999993,
      0.0,
      21492.32999999999,
      1952.0595000000012,
      7386.576664999998,
      0.0,
      13772.264999999987,
      2456.298999999998,
      0.0,
      0.0,
      9578.164999999994,
      3013.3894999999975,
      0.0,
      0.0,
      8291.157499999996,
      3286.9862499999995,
      0.0,
      0.0,
      202.48095163950026,
      8.109684856894507,
      20.148357925000003,
      1.055469599249997,
      209258.71271657402,
      216929.12875779468,
      225974.72766396723]]




```python
# Special instance of the integration that specifically uses
# the Power channel string to integrate over time and calculate energy
mycruncher.compute_energy('GenPwr')
```




    [49418.5000000001, 51033.00000000001, 50520.17499999997]


