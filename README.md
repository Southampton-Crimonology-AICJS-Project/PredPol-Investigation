# AI-CJS [![DOI](https://zenodo.org/badge/319669776.svg)](https://zenodo.org/badge/latestdoi/319669776)

### Artificial Intelligence in the criminal justice system

Sharepoint: https://sotonac.sharepoint.com/teams/AI-CJS

GitHub: https://github.com/Southampton-RSG/AI_CJS

This project aims to:

Understand the predictive policing algorithms, specifically PredPol, used in the criminal justice system.

### Install

Requires:

_A ssh key must be used to be able to download from the github: https://github.com/settings/keys, https://help.github.com/articles/generating-an-ssh-key/_

_Anaconda must be installed prior to these steps_

What is installed:

This install process will make and setup a Conda env called ai-cjs-env

During install 'postcodes_io' and 'osgridconverter' from pypi will be built into Conda packages locally using conda
 skeleton.

The package 'ai_cjs' will be installed using the github release given in './conda_setup/ai_cjs_build/meta.yaml
' Source files are in './ai_cjs'.

Run:

To install run './install.sh' (This in turn runs './conda_setup/install.sh' where full install details can be found.)

#### Modify and reinstall ai_cjs

To make changes to ai_cjs: 

- Edit the files in './ai_cjs/' 
- Push to GitHub 
- Create a new release
- Point './conda_setup/ai_cjs_build/meta.yaml' to the release
- run './conda_setup/ai_cjs_build/reinstall_ai_cjs.sh' to remove ai_cjs from ai-cjs-env and replace with the new
 version.
 
 ### Run the model
 
 The implementation in this release is a HPC release intended to be run on systems with multiple processors. The
 number of processor cores for a given model **MUST** be a square number and should not exceed the physical number of
 cores available (e.g. no hyperthreads).
 
 Scripts to launch the models using the PBS Script Format are given in './bash_pbs/'
 
 To test the installation and ensure correct running use './bash_pbs/test'
 
 For the analysis used in this releases companion paper see: './bash_pbs/paper_jobs/run_all'
 
 
#### Additional data analysis

Requires:

jupyter notebook

or

jupyter lab

There are several jupyter notebooks used to test routines and provide additional data analysis. To use these start a
 notebook server in this directory 'jupyter notebook' or 'jupyter lab' and navigate to the desired .ipynb file
  ensuring ai-cjs-env is the selected python kernal.
