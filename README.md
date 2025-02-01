# pCrunch

IO and Post Processing for generic time series data of multibody aeroelastic wind turbine simulations.  Readers are provided for OpenFAST outputs, but the analysis tools are equally applicable to HAWC2, Bladed, QBlade, ADAMS, or other tools.

## Installation as a Library

pCrunch is installable through pip via `pip install pCrunch` or conda, `conda install pCrunch`.

## Development Setup

To set up pCrunch for development, follow these steps:

1. Download the [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3) variant of Anaconda
2. Open a terminal (or the Miniforge terminal on Windows) and create a new environment for the project with the following.

        conda create -n pcrunch-dev

3. To activate/deactivate the environment, use the following commands.

        conda activate pcrunch-dev
        conda deactivate pcrunch-dev

4. Install dependencies needed for running and development

        conda install fatpack numpy pandas pyyaml ruamel.yaml scipy numexpr pytest treon

4. Clone the repository:

        git clone https://github.com/NREL/pCrunch.git

5. Navigate to the top level of the repository (`<path-to-pCrunch>/pCrunch/`) and install pCrunch as an editable package
   with following commands.

        pip install -ev .

## Examples and Documentation

For an up to date example of the core functionalities, see the examples directory

