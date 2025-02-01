# pCrunch's Aeroelastic Output class

The `AeroelasticOutput` class is a general container for time-series based data for a single environmental condition (i.e., a single incoming wind spead and turbulence seed value).  This might be a single run of your aeroelastic multibody simulation tool (OpenFAST or HAWC2 or Bladed or QBlade or in-house equivalents) in a larger parametric variation for design load case (DLC) analysis.  The `AeroelasticOutput` class provides data containers and common or convenient manipulations of the data for engineering analysis.  

Analysis that involve multiple time-series simulations, such as a full run of multiple wind speeds and seeds, which yield multiple AeroelasticOutput instances, is done in the *Crunch class*.

This file lays out some workflows and showcases capabilities of the `AeroelasticOutput` class.

## Creating a new class instance

The `AeroelasticOutput` class can be initialized from an output file or from existing data structures.  pCrunch provides a reader for OpenFAST output files (both binary and ascii).  To expand pCrunch for use with other aeroelastic multibody codes, users could simply use the `openfast_readers.py` file as a template.  If you already have the data in Python, then data structures such as dictionaries, lists, NumPy arrays, and Pandas DataFrames can all be used as a constructor.  Here are some examples with each `myobj` representing a valid AeroelatsicOutput instance:


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pCrunch import AeroelasticOutput, read, FatigueParams

thisdir = os.path.realpath('')
datadir = os.path.join(thisdir, '..', 'pCrunch', 'test', 'data')

# OpenFAST output files
myobj_of_ascii = read( os.path.join(datadir, 'DLC2.3_1.out') )
myobj_of_bin   = read( os.path.join(datadir, 'Test2.outb') )

# Existing data structures
mydata = {
    "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "WindVxi": [7, 7, 7, 7, 7, 8, 8, 8, 8, 8],
    "WindVyi": [0] * 10,
    "WindVzi": [0] * 10,
}

# From a dictionary
myobj_from_dict  = AeroelasticOutput(mydata)

# From a Pandas DataFrame
myobj_from_df    = AeroelasticOutput( pd.DataFrame(mydata) )

# From Python lists
chan_labels      = list( mydata.keys() )
ts_data          = [m for m in mydata.values()]
myobj_from_list  = AeroelasticOutput(ts_data, chan_labels)

# From a Numpy array
myobj_from_numpy = AeroelasticOutput(np.array(ts_data), chan_labels)

# As a copy from an existing output (especially helpful when needing to filter the core data)
myobj_copy       = myobj_from_numpy.copy()
```

Additional, optional arguments can also be passed that specify a label, a description, and a vector of units for the data channels:


```python
myunits = ['s', 'm/s', 'm/s', 'm/s']
myobj_from_dict = AeroelasticOutput(mydata, name='pseudodata', description='pCrunch example', units=myunits)
```

## Data structures and access

pCrunch stores the time series data as a Numpy array and the channel names as a list.  More sophisticated data containers, such as netcdf or hdf5, could be adopted in future work, but the simplicity, familiarity, and accessibility of the data containers should help users adopt pCrunch into their workflows.  Easy converstions back to a Python dictionary or Pandas dataframe are available:


```python
myobj_from_dict.channels
```




    ['Time', 'WindVxi', 'WindVyi', 'WindVzi']




```python
myobj_from_dict.data
```




    array([[ 1,  7,  0,  0],
           [ 2,  7,  0,  0],
           [ 3,  7,  0,  0],
           [ 4,  7,  0,  0],
           [ 5,  7,  0,  0],
           [ 6,  8,  0,  0],
           [ 7,  8,  0,  0],
           [ 8,  8,  0,  0],
           [ 9,  8,  0,  0],
           [10,  8,  0,  0]])




```python
myobj_from_dict.to_dict()
```




    {'Time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     'WindVxi': [7, 7, 7, 7, 7, 8, 8, 8, 8, 8],
     'WindVyi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     'WindVzi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}




```python
myobj_from_dict.to_df()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>WindVxi</th>
      <th>WindVyi</th>
      <th>WindVzi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Accessing the data for a particular channel is done using familiar dictionary or DataFrame syntax.  A `.time` property is also available as that is assumed to be common to all datasets


```python
myobj_from_dict['Time']
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
myobj_from_dict.time
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
myobj_from_dict['WindVxi']
```




    array([7, 7, 7, 7, 7, 8, 8, 8, 8, 8])



If working with the data or summary statistics outside of the AeroelasticOutput object, it can be help to have an easy way to grab the index into the channel vector.  This is available via:


```python
myobj_from_dict.chan_idx('WindVxi')
```




    1



## Adding new channels, dropping channels, and math operations on channel data

Significant new capability has been added in pCrunch v2 to enable easy addition of new data channels, especially from mathematically manipulating existing channels.  There are also easy short cuts to add channels that are vector magnitudes and load rose sectors based on vector components, which is helpful for tower and blade loading analysis.

As with the constructor, new channel data can be added with a dictionary, list, Pandas Series, or Numpy array:


```python
# Inputting a dictionary new channel
myobj_from_dict.add_channel( {'New1': np.sin(myobj_from_dict['Time'])} )

# As a DataFrame
myobj_from_dict.add_channel( pd.DataFrame({'New2': np.cos(myobj_from_dict['Time'])}) )

# As a Numpy array or list and channel as a string
myobj_from_dict.add_channel( np.tan(myobj_from_dict['Time']), 'New3' )

myobj_from_dict.to_df()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>WindVxi</th>
      <th>WindVyi</th>
      <th>WindVzi</th>
      <th>New1</th>
      <th>New2</th>
      <th>New3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.841471</td>
      <td>0.540302</td>
      <td>1.557408</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.909297</td>
      <td>-0.416147</td>
      <td>-2.185040</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.141120</td>
      <td>-0.989992</td>
      <td>-0.142547</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.756802</td>
      <td>-0.653644</td>
      <td>1.157821</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.958924</td>
      <td>0.283662</td>
      <td>-3.380515</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.279415</td>
      <td>0.960170</td>
      <td>-0.291006</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.656987</td>
      <td>0.753902</td>
      <td>0.871448</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.989358</td>
      <td>-0.145500</td>
      <td>-6.799711</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.412118</td>
      <td>-0.911130</td>
      <td>-0.452316</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.544021</td>
      <td>-0.839072</td>
      <td>0.648361</td>
    </tr>
  </tbody>
</table>
</div>



A new feature in pCrunch that restores some of the old capability in mcrunch is the ability to write string expressions to add a new channel.  String names should match channel names and all standard python math expressions are allowed.  Users can also use `calculate_channel` in addition to `add_channel` for mcrunch consistency.


```python
myobj_from_dict.add_channel( 'WindVxi**2 + WindVyi + New1/New2', 'New4' )
myobj_from_dict.to_df()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>WindVxi</th>
      <th>WindVyi</th>
      <th>WindVzi</th>
      <th>New1</th>
      <th>New2</th>
      <th>New3</th>
      <th>New4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.841471</td>
      <td>0.540302</td>
      <td>1.557408</td>
      <td>50.557408</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.909297</td>
      <td>-0.416147</td>
      <td>-2.185040</td>
      <td>46.814960</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.141120</td>
      <td>-0.989992</td>
      <td>-0.142547</td>
      <td>48.857453</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.756802</td>
      <td>-0.653644</td>
      <td>1.157821</td>
      <td>50.157821</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.958924</td>
      <td>0.283662</td>
      <td>-3.380515</td>
      <td>45.619485</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.279415</td>
      <td>0.960170</td>
      <td>-0.291006</td>
      <td>63.708994</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.656987</td>
      <td>0.753902</td>
      <td>0.871448</td>
      <td>64.871448</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.989358</td>
      <td>-0.145500</td>
      <td>-6.799711</td>
      <td>57.200289</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.412118</td>
      <td>-0.911130</td>
      <td>-0.452316</td>
      <td>63.547684</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.544021</td>
      <td>-0.839072</td>
      <td>0.648361</td>
      <td>64.648361</td>
    </tr>
  </tbody>
</table>
</div>



Channels can also be dropped using string wildcards


```python
myobj_from_dict.drop_channel('New*')
myobj_from_dict.to_df()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>WindVxi</th>
      <th>WindVyi</th>
      <th>WindVzi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Vector magnitudes

Computing vector magnitudes is a common operation, which can be done by hand using one of the approaches above, or in the constructor by passing in a dictionary:


```python
mc = {"Wind": ["WindVxi", "WindVyi", "WindVzi"]}
myobj_with_mag = AeroelasticOutput(mydata, magnitude_channels=mc)
myobj_with_mag.to_df()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>WindVxi</th>
      <th>WindVyi</th>
      <th>WindVzi</th>
      <th>Wind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



The magnitude channels can also be added after the fact too:


```python
myobj_with_mag = AeroelasticOutput(mydata)
myobj_with_mag.add_magnitude_channels(mc)
myobj_with_mag.to_df()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>WindVxi</th>
      <th>WindVyi</th>
      <th>WindVzi</th>
      <th>Wind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



### Load Roses

Neither vector components nor magnitude correctly capture the load impacts on a tower base or blade root.  A more appropriate approach is a load rose, where the 360-degree annulus is divided into sectors and the vector components are combined with sin() and cos() to compute the load impacts on each sector.  pCrunch automates this process and it results in the creation of n_sector new channels of data.  An example:


```python
lr = {'TwrBs': ['TwrBsFxt', 'TwrBsFyt']}
myobj_of_bin.add_load_rose(lr, nsec=6)
```

    Added channel, TwrBs0
    Added channel, TwrBs60
    Added channel, TwrBs120
    Added channel, TwrBs180
    Added channel, TwrBs240
    Added channel, TwrBs300


### Binning, windowing, averaging

Another common operation is to downsample the time series signals in various ways.  Options include:

- Trim the data to remove transients or otherwise narrow the series
- Windowed smoothing via correlation
- Binned averages


```python
# Trimming data can be done to the full data set
print( myobj_of_bin.elapsed_time, myobj_of_bin.num_timesteps )
myobj_of_bin.trim_data(100, 600)
print( myobj_of_bin.elapsed_time, myobj_of_bin.num_timesteps )
```

    600.0000089406967 6001
    499.9000074490905 5000



```python
# Time windowing convolves an averaging window with the time signal and sets this as the new data array with the same timestep, 
# but a shorter signal that covers the valid windowing region.
myobj_of_bin.time_averaging(30.0)
print( myobj_of_bin.elapsed_time, myobj_of_bin.num_timesteps, myobj_of_bin.dt)
```

    470.1000070050359 4702 0.10000000149011612



```python
# Time binning results in a downsampled data set that represents the average for each bin
myobj_of_bin.time_binning(30.0)
print( myobj_of_bin.elapsed_time, myobj_of_bin.num_timesteps, myobj_of_bin.dt)
```

    445.10000663250685 16 30.000000447034836


## Frequency domain spectra

Power spectral density in the frequency domain is made readily available using the SciPy `welch` function.  A new AeroelasticOutput object is returned with frequency taking the place of time and PSD content in the frequency domain taking the place of the time-domain data.


```python
freq_obj = myobj_of_ascii.psd()
plt.loglog(freq_obj['Freq'], freq_obj['TwrBsFyt'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD')
plt.grid()
```


    
![png](output_33_0.png)
    


## Statistics, extremes, and many other quantities


```python
# Many other statistics of the data are readily available. A quick summary for each channel of data is available in a dictionary
myobj_from_df.summary_stats()
```




    {'Time': {'min': 1,
      'max': 10,
      'std': 2.8722813232690143,
      'mean': 5.5,
      'median': 5.5,
      'abs': 10,
      'integrated': 49.5},
     'WindVxi': {'min': 7,
      'max': 8,
      'std': 0.5,
      'mean': 7.5,
      'median': 7.5,
      'abs': 8,
      'integrated': 67.5},
     'WindVyi': {'min': 0,
      'max': 0,
      'std': 0.0,
      'mean': 0.0,
      'median': 0.0,
      'abs': 0,
      'integrated': 0.0},
     'WindVzi': {'min': 0,
      'max': 0,
      'std': 0.0,
      'mean': 0.0,
      'median': 0.0,
      'abs': 0,
      'integrated': 0.0}}




```python
myobj_from_df.summary_stats()['WindVxi']['mean']
```




    7.5




```python
# It is helpful to know the value of other channels when one of interest is at its extreme value
myobj_from_df.extremes()
```




    {'Time': {'Time': 10, 'WindVxi': 8, 'WindVyi': 0, 'WindVzi': 0},
     'WindVxi': {'Time': 6, 'WindVxi': 8, 'WindVyi': 0, 'WindVzi': 0},
     'WindVyi': {'Time': 1, 'WindVxi': 7, 'WindVyi': 0, 'WindVzi': 0},
     'WindVzi': {'Time': 1, 'WindVxi': 7, 'WindVyi': 0, 'WindVzi': 0}}




```python
# This can be done for the whole dataset (which can be a large NxN output), or specific channels
myobj_of_ascii.extremes(['RotTorq','TwrBsFyt'])
```




    {'RotTorq': {'Time': 54.45, 'RotTorq': 2650.0, 'TwrBsFyt': -22.0},
     'TwrBsFyt': {'Time': 70.85, 'RotTorq': -212.0, 'TwrBsFyt': 53.6}}



A larger laundry list of statistics are available as data properties (meaning they don't have to be called as a function):


```python
# Indices to the minimum value for each channel
myobj_from_df.idxmins
```




    array([0, 0, 0, 0])




```python
# Indices to the maximum value for each channel
myobj_from_df.idxmaxs
```




    array([9, 5, 0, 0])




```python
# Minimum value of each channel
myobj_from_df.minima
```




    array([1, 7, 0, 0])




```python
# Maximum value of each channel
myobj_from_df.maxima
```




    array([10,  8,  0,  0])




```python
# Maximum value of absolute values of each channel
myobj_from_df.absmaxima
```




    array([10,  8,  0,  0])




```python
# The range of data values (max - min)
myobj_from_df.ranges
```




    array([9, 1, 0, 0])




```python
# Channel indices which vary in time
myobj_from_df.variable
```




    array([0, 1])




```python
# Channel indices which are constant in time
myobj_from_df.constant
```




    array([2, 3])




```python
# Sum of channel values over time
myobj_from_df.sums
```




    array([55, 75,  0,  0])




```python
# Sum of channel values over time to the second power
myobj_from_df.sums_squared
```




    array([385, 565,   0,   0])




```python
# Sum of channel values over time to the third power
myobj_from_df.sums_cubed
```




    array([3025, 4275,    0,    0])




```python
# Sum of channel values over time to the fourth power
myobj_from_df.sums_fourth
```




    array([25333, 32485,     0,     0])




```python
# Second moment of the timeseries for each channel
myobj_from_df.second_moments
```




    array([8.25, 0.25, 0.  , 0.  ])




```python
# Third moment of the timeseries for each channel
myobj_from_df.third_moments
```




    array([0., 0., 0., 0.])




```python
# Fourth moment of the timeseries for each channel
myobj_from_df.fourth_moments
```




    array([1.208625e+02, 6.250000e-02, 0.000000e+00, 0.000000e+00])




```python
# Mean of channel values over time
myobj_from_df.means
```




    array([5.5, 7.5, 0. , 0. ])




```python
# Median of channel values over time
myobj_from_df.medians
```




    array([5.5, 7.5, 0. , 0. ])




```python
# Standard deviation of channel values over time
myobj_from_df.stddevs
```




    array([2.87228132, 0.5       , 0.        , 0.        ])




```python
# Skew of channel values over time
myobj_from_df.skews
```

    /Users/gbarter/devel/pCrunch/pCrunch/aeroelastic_output.py:363: RuntimeWarning: invalid value encountered in divide
      return self.third_moments / np.sqrt(self.second_moments) ** 3





    array([ 0.,  0., nan, nan])




```python
# Kurtosis of channel values over time
myobj_from_df.kurtosis
```

    /Users/gbarter/devel/pCrunch/pCrunch/aeroelastic_output.py:367: RuntimeWarning: invalid value encountered in divide
      return self.fourth_moments / self.second_moments ** 2





    array([1.77575758, 1.        ,        nan,        nan])




```python
# Integration of channel values over time
myobj_from_df.integrated
```




    array([49.5, 67.5,  0. ,  0. ])




```python
# Special instance of the integration that specifically uses
# the Power channel string to integrate over time and calculate energy
myobj_of_ascii.compute_energy('GenPwr') 
```




    72637.25



## Calculating fatigue

pCrunch can compute damage equivalent loads and, optionally, traditional Palmgren-Miner damage.  Computing these quantities requires additional inputs for material properties, S-N curve parameters, and some algorithm choices (although most of the work is handed off to the `fatpack` module.  These additional parameters would most likely vary from one channel to the next.  For instance, blade composites will use different inputs that the structural steel in the tower or the fancy steel in the low-speed shaft.  To facilitate these additional inputs, pCrunch provides a `FatigueParams` class that is simply a container.  Association between a load channel and a FatigueParams instance is done with a dictionary, similar to the magnitude channels.

Instead of using the same examples as above, here we'll build a couple of sinusoids to understand the numerics a bit better.  One sinusoid is centered at y=0 and the other at y=40kN.


```python
# Build a FatigueParams instance that we'll use for all channels
myparam = FatigueParams(lifetime = 30.0,             # Lifetime in years
                        load2stress = 25.0,          # Factor based on cross-section to convert channel force/moment to stress
                        slope = 3.0,                 # Slope of S-N curve
                        ultimate_stress = 6e8,       # Yield stress of the material
                        S_intercept = 5e9,           # S-intercept on S-N curve (catastrophic load amplitude for 1 cycle)
                        goodman_correction = False,  # Apply Goodman correction for mean load value?
                        return_damage = True,        # Compute Palmgren-Miner damage?
                        )

# Our time series
t   = np.linspace(0, 600, 10000)

# Sinusoids centered at 0 and 40kN, with amplitude of 40kN
y0  = 80e3 * np.sin(2*np.pi*t/60.0) # Will have +/- values
y80 = y0 + 80e3                     # All + values
zeros = np.zeros(y0.shape)

# Simple dictionary for AeroelasticOutput instance
mydata = {"Time":t,
          "Signal0":y0,
          "Signal80":y80,
          "Zeros":zeros}

# Magnitude channels (in this case, just an absolute value operation on the sinusoids)
mymagnitudes = {"Mag0":["Signal0", "Zeros"],
                "Mag80":["Signal80", "Zeros"]}

# Create the instance
myobj = AeroelasticOutput(mydata, magnitude_channels=mymagnitudes)

# The channels we will be computing fatigue on
myfatigues = {"Signal0":myparam,
              "Signal80":myparam,
              "Mag0":myparam,
              "Mag80":myparam}

# Loop over channels and pass in channel-specific fatigue parameters
dels = np.zeros(len(myfatigues))
dams = np.zeros(len(myfatigues))
for ik, k in enumerate(myfatigues.keys()):
    dels[ik], dams[ik] = myobj.compute_del(k, myfatigues[k])

# Organize the output into a table
pd.DataFrame(np.c_[dels, dams], index=myfatigues.keys(), columns=['DELs', 'Damage'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DELs</th>
      <th>Damage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Signal0</th>
      <td>40869.837167</td>
      <td>0.008073</td>
    </tr>
    <tr>
      <th>Signal80</th>
      <td>40869.837167</td>
      <td>0.008073</td>
    </tr>
    <tr>
      <th>Mag0</th>
      <td>25746.384881</td>
      <td>0.002018</td>
    </tr>
    <tr>
      <th>Mag80</th>
      <td>40869.837167</td>
      <td>0.008073</td>
    </tr>
  </tbody>
</table>
</div>



A couple of points to highlight in the results:

- The Signal0 and Signal80 have the same DEL and Damage values because the amplitude of the variations are equivalent
- The Mag80 signal is also equivalent because the signal is unchanged by the magnitude operation
- The Mag0 signal has noticeably less fatigue accumulation.  This is because by taking the absolute value of the signal centered at zero, we have doubled the frequency but halved the amplitude.  These effects combine in nonlinear ways, but the net result is a drop in fatigue accumulation.
  
Now let's add the Goodman Correction, which should calculate additional fatigue impacts based on the mean value of the signals, not just the amplitude of variation.  We can do this by either regenerating a new FatigueParams instance with the Goodman flag set to True, or pass in a keyword to the `compute_del` function that overrides the inputs.


```python
dels2 = np.zeros(len(myfatigues))
dams2 = np.zeros(len(myfatigues))
for ik, k in enumerate(myfatigues.keys()):
    dels2[ik], dams2[ik] = myobj.compute_del(k, myfatigues[k], goodman_correction=True)
    
pd.DataFrame(np.c_[dels, dels2, dams, dams2], index=myfatigues.keys(), columns=['DELs', 'DELs-Goodman', 'Damage', 'Damage-Goodman'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DELs</th>
      <th>DELs-Goodman</th>
      <th>Damage</th>
      <th>Damage-Goodman</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Signal0</th>
      <td>40869.837167</td>
      <td>40869.837167</td>
      <td>0.008073</td>
      <td>0.008073</td>
    </tr>
    <tr>
      <th>Signal80</th>
      <td>40869.837167</td>
      <td>41006.525582</td>
      <td>0.008073</td>
      <td>0.008154</td>
    </tr>
    <tr>
      <th>Mag0</th>
      <td>25746.384881</td>
      <td>25789.367156</td>
      <td>0.002018</td>
      <td>0.002028</td>
    </tr>
    <tr>
      <th>Mag80</th>
      <td>40869.837167</td>
      <td>41006.525582</td>
      <td>0.008073</td>
      <td>0.008154</td>
    </tr>
  </tbody>
</table>
</div>



As expected, there is now a difference between Signal0 and Signal80 in the Goodman columns because of the higher mean loads in the Signal80 case.
