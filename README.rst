pCrunch
=======

IO and Post Processing for generic time series data of multibody aeroelastic wind turbine simulations.  Readers are provided for OpenFAST outputs, but the analysis tools are equally applicable to HAWC2, Bladed, QBlade, ADAMS, or other tools.

Installation as a Library
------------------------------------

pCrunch is installable through pip via ``pip install pCrunch`` or conda, ``conda install pCrunch``.

Development Setup
-------------------

To set up pCrunch for development, follow these steps:

1. Download the [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3) variant of Anaconda
2. From the terminal, install pip by running: ``conda install -c anaconda pip``
3. Next, create a new environment for the project with the following.

    .. code-block:: console

        conda create -n pcrunch-dev

   To activate/deactivate the environment, use the following commands.

    .. code-block:: console

        conda activate pcrunch-dev
        conda deactivate pcrunch-dev

4. Clone the repository:
   ``git clone https://github.com/NREL/pCrunch.git``

5. Navigate to the top level of the repository
   (``<path-to-pCrunch>/pCrunch/``) and install pCrunch as an editable package
   with following commands.

    .. code-block:: console

       pip install -e '.[dev]'

Examples
--------

For an up to date example of the core functionalities, see `example.ipynb`. More
examples coming soon.
