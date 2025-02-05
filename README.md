README file:
Project Documentation:
  Project description: main objectives, assumptions, hypothesis, etc.
  Folder/module sutructure, including sub-modules.
  Key stages and what was done in each (e.g., data import --> data processing --> modeling --> analysis --> visualisation/graphs).
  Important definitions and explanations of key parameters/configurations.
Data Description and Link to the Dataset
References to Papers or Articles used
Instructions for running the project (commands).


# Predicting fMRI Response Dissimilarity to Natural Scenes
COURSE: Data Science and Advanced Python Concepts Workshop for Neuroscience

BAR-ILAN UNIVERSITY

## Project Documentation
This project did not aim to replicate or directly compare results from existing papers. Instead, we adopted an exploratory approach to a novel task in an existing problem space.

The goal of this project is to predict how different our brain response will be to viewing a pair of images representing natural scenes, i.e., predict dissimilarity between fMRI responses to pairs of images.

Based on the literature we assume that it is possible to predict a stimulus given an fMRI response (decoding), or vice versa, predicting the fMRI response given a stimulus (encoding).
Moreover, the similarity between brain responses and machine learning models has also been explored, and moderately high correlations have been reported.

Based on these assumptions we hypothesise that our novel approach could result successful: to build a machine learning model capable of predicting how dissimilarly a given brain region in the visual cortex will process two different images.

Two different machine learning model architectures were explored, the first consists of a simple 2-Layer Siamese CNN, which did not succeed to get encouraging results. The second one consists of a Multi-Layer Perceptron model that takes two concatenated embeddings as input, corresponding to the each pair of images, which are obtained using a pretrained model.

**PDF report file:** This repository contains a PDF file with the project report, where the project backgrpung, design and methodology are carefully detailed. It follows the structure of a scientific report with the following parts: introduction, methodology, results, and conclusion and discussion. As specified in the guidelines, no citations or references are included.

## Project structure
Full dataset results are run via various Google Colab workbooks (impossible to run with the full dataset on a local machine), but this repository has all corresponding code split out into organized files (per `Directory structure` detailed below).

We have deprecated code here from the first model simply for storage/tracking purposes, not for running. We also have certain functions which we used for testing, but are not called in main.py as they are not part of the core analysis.

### Directory structure
<details>
<summary>Directory Structure</summary>

```bash
predicting_fmri_response_dissimilarity_to_natural_scenes/
  ├── src/
  │   ├── project/
  │   │   ├── config/
  │   │   │   ├── __init__.py
  │   │   │   ├── clip_config.py
  │   │   │   └── dep_config.py
  │   │   ├── models/
  │   │   │   ├── __init__.py
  │   │   │   ├── dep_models.py
  │   │   │   └── pytorch_models.py
  │   │   ├── utils/
  │   │   │   ├── __init__.py
  │   │   │   ├── clip_utils.py
  │   │   │   ├── data_utils.py
  │   │   │   ├── pytorch_data.py
  │   │   │   ├── pytorch_training.py
  │   │   │   └── visualizations.py
  │   │   ├── __init__.py
  │   │   ├── dep_main.py
  │   │   ├── logger.py
  │   │   └── main.py
```
</details>

As shown in the directory structure, our project contains three package-like folders for configuration, machine learning models, and different utilities to use the pretrained model, to preprocess our fMRI data, to prepare our data for the ML model using pytorch, to train our model, and to visualize the results. ...logger... main file
Three specific files contain the first/deprecated model inside the configuration and model folders, and its main file in source. 

### On first run:
```bash 
# install Virtualenv - a tool to set up your Python environments
pip install virtualenv
# create virtual environment (serve only this project):
python3.10 -m venv venv
# activate virtual environment
source venv/bin/activate
# if on Windows
venv/Scripts/activate
+ (venv) should appear as prefix to all command (run next command just after activating venv)
# update venv's python package-installer (pip) to its latest version
pip install --upgrade pip
# install projects packages
pip install -e .[dev]
``` 

#### Setup your data folder
1. Download Google Drive for Desktop [here](https://support.google.com/a/users/answer/13022292?hl=en)
2. Ensure you have added a [Shortcut](https://support.google.com/drive/answer/9700156?hl=en&co=GENIE.Platform%3DDesktop) from the desired data folder to any folder in your Drive. For the data folder, either setup "algonauts_2023_tutorial_data" per instructions [here](https://docs.google.com/forms/d/e/1FAIpQLSehZkqZOUNk18uTjRTuLj7UYmRGz-OkdsU25AyO3Wm6iAb0VA/viewform?usp=sf_link), or request access to the mini testing data "mini_data_for_python" folder [here](https://drive.google.com/drive/folders/19mXhFsOlFWu2vyPkj5In2VQS-Buu4K48?usp=sharing)
3. Copy the full Shorcut path for use later, e.g. `~/Library/CloudStorage/GoogleDrive-<your_email>/.shortcut-targets-by-id/<shortcut_id>`
4. Ensure your data folder is in "offline mode" locally (Right click the folder, e.g. "mini_data_for_python" > Make available offline).

### Running the Models
Details on how to run this code locally with a small subset of data are noted here as well.

#### Configuration files:
1. [DEPRECATED] dep_config.py --> user configurations for the first model
2. clip_config.py --> user configurations for the second model; NOTE: "SWEEP" to True means that you will loop over the layers in "LAYERS_LIST" after it finishes running with the parameterized layer number, HIDDEN_LAYERS

    a. [**RECOMMENDED**] Update `LOAD_EMBEDDINGS_FILE` in `clip_config.py` with the proper Shortcut path such that you can use the pre-run embeddings, rather than loading from scratch. If using Windows ensure that the path uses backslashes instead of forwardslashes. If using THINGSvision features, replace `LOAD_EMBEDDINGS_FILE` with the path to the THINGSvision features instead. (NOT RECOMMENDED - Otherwise, update `LOAD_EMBEDDINGS_FILE` TO `""`.)

#### [DEPRECATED] First model (Siamese CNN)
```bash 
python3.10 -m src.project.dep_main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```

#### Second model (simple MLP)
```bash 
python3.10 -m src.project.main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```
##### Note - to use THINGSvision features, update the above command for the second model with the optional flag `--thingsvision`

## Modify package dependencies (add/remove/update external modules/packages):
#### Add new module:
1. Add package to pyproject.toml
2. Run:
```bash 
pip install -e .[dev]
```

#### Remove new module:
1. Remove the package from pyproject.toml
2. Run:
```bash 
pip uninstall <package-name>
```
note: if you're don't remember the exact package name copy it from: 
```bash
pip list
```

#### Steps in case of a package failure:
Cases like package installation interuppted in the middle or something like that
1. Try to remove package and install it again.
2. If it doesn't help delete venv folder 
3. repeat 'On first run' steps


## Health check (Lint):
#### Lint Project:
Check formatting, type hinting, lint code & docstrings
```bash
tox run
```
