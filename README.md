# pCrunch

IO and Post Processing for generic time series data of multibody aeroelastic wind turbine simulations.  Readers are provided for OpenFAST outputs, but the analysis tools are equally applicable to HAWC2, Bladed, QBlade, ADAMS, or other tools.  pCrunch attempts to capture the best of legacy tools MCrunch, MLife, and MExtremes, while also taking inspiration from other similar utilities available on Github.

## Installation as a Library

pCrunch is installable through pip via `pip install pCrunch` or conda, `conda install pCrunch`.

## Development Setup

To set up pCrunch for development, follow these steps:

1. Download the [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3) variant of Anaconda
2. Open a terminal (or the Miniforge terminal on Windows) and create a new environment for the project with the following.

        conda config --add channels conda-forge
        conda install git
        git clone https://github.com/NREL/pCrunch.git
        cd pCrunch
        conda env create --name pcrunch-dev -f environment.yml
        conda activate pcrunch-dev


3. To activate/deactivate the environment, use the following commands.

        conda activate pcrunch-dev
        conda deactivate pcrunch-dev
		

4. Install additional packages for testing

        conda install pytest treon


5. Install pCrunch as an editable package with following commands.

        pip install -e . -v


## Examples and Documentation

For an up to date example of the core functionalities, see the examples-directory for Jupyter notebook examples, or the docs-directory for the same material.

There are two primary analysis classes in pCrunch:

1. The `AeroelasticOutputs` class
2. The `Crunch` class.

### The AeroelasticOutputs class

The `AeroelasticOutput` class is a general container for time-series based data for a single environmental condition (i.e., a single incoming wind spead and turbulence seed value).  This might be a single run of your aeroelastic multibody simulation tool (OpenFAST or HAWC2 or Bladed or QBlade or in-house equivalents) in a larger parametric variation for design load case (DLC) analysis.  The `AeroelasticOutput` class provides data containers and common or convenient manipulations of the data for engineering analysis.

### The Crunch class

The `Crunch` class is a general analysis tool for batches of time-series based data across multiple environmental conditions (i.e., a full wind speed and turbulence seed sweep). The methods are agnostic to the aeroelastic multibody simulation tool (OpenFAST or HAWC2 or Bladed or QBlade or in-house equivalents). The `AeroelasticOutput` class provides the data containers for each individual simulation.  The `AeroelasticOutput` class provides many analysis capabilities and the `Crunch` class extends them into their batch versions.

The `Crunch` class supports keeping all time series data in memory and a lean "streaming" version where outputs are processed and then deleted, retaining only the critical statistics and analysis outputs.


