# Predicting fMRI Response Dissimilarity to Natural Scenes
COURSE: Data Science and Advanced Python Concepts Workshop for Neuroscience

BAR-ILAN UNIVERSITY

## Project Documentation
### Project Description
This project did not aim to replicate or directly compare results from existing papers. Instead, we adopted an **exploratory approach** to a novel task in an existing problem space.

The goal of this project is to predict how different our brain response will be to viewing a pair of images representing natural scenes, i.e., **predict dissimilarity between fMRI responses to pairs of images.**

Based on the literature we assume that it is possible to predict a stimulus given an fMRI response (decoding), or vice versa, predicting the fMRI response given a stimulus (encoding).
Moreover, the similarity between brain responses and Machine Learning (ML) models has also been explored, and moderately high correlations have been reported. Specifically, the paper *THINGSvision: A Python Toolbox for Streamlining the Extraction of Activations From Deep Neural Networks* explores the correlation between various pretrained model embeddings and fMRI responses. The authors noted high correlation between a few models, namely CLIP-ViT, and actual fMRI responses, giving us the basis to begin our prediction work.

Based on these assumptions we hypothesize that our novel approach could result successful: **to build an ML model capable of predicting how dissimilarly a given brain region in the visual cortex will process two different images.**

Two different ML model architectures were explored, the first consists of a **simple 2-Layer Siamese CNN,** which did not succeed to get encouraging results. The second one consists of a **Multi-Layer Perceptron model** that takes two concatenated embeddings as input, corresponding to each pair of images, which are obtained using a pretrained model.

**PDF report file:** This repository contains a PDF file with the project report, where the project background, design and methodology are carefully detailed. It follows the structure of a scientific report with the following parts: introduction, methodology, results, and conclusion and discussion. As specified in the guidelines, no citations or references are included.

### Project structure
Full dataset results are run via various Google Colab workbooks (impossible to run with the full dataset on a local machine), but this repository has all corresponding code split out into organized files (per `Directory structure` detailed below).

#### Directory structure
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

As shown in the directory structure, our project contains three package-like folders for configuration, ML models, and different utilities to use the pretrained model, to preprocess our fMRI data, to prepare our data for the ML model using pytorch, to train our model, and to visualize the results. Additionally, the logger.py file is used for logging, while main.py serves as the main entry point of the project. There are also certain functions which we used for testing, but are not called in main.py as they are not part of the core analysis.

**"dep_main.py", "dep_models.py", and "dep_config.py" are deprecated code and left simply for storage/tracking purposes (NOTE: they are not up-to-date with coding standards)**.

Outside of the source/project/ we have the TOML file which defines the project metadata, dependencies, and tool configurations; the Tox file which automates running linting; and the test files which contain unit tests to validate the project's functionality.

**About the test files:** tests were excluded for any deprecated code, visualizations.py, and main.py (unit tests already cover functions called in main.py). Additionally, magic number checking was excluded in test files.

### Configuration files: important explanations
1. [DEPRECATED] dep_config.py --> user configurations for the first/deprecated model
2. clip_config.py --> user configurations for the second model. In this file, several configurations can be updated: for user/experimental parameters (e.g., subject, data directory, ROIs, ROI classes, desired image number), for the CLIP model (pretrained model to use, path to file), and for the RDM/training parameters (e.g., metric to calculate dissimilarity, accuracy metric, activation function, number of epochs, learning rate, batch size, test size, number of hidden layers, use K fold cross validation or not).
   NOTE: "SWEEP" to True means that you will loop over the layers in "LAYERS_LIST" after it finishes running with the parameterized layer number, HIDDEN_LAYERS

    a. [**RECOMMENDED**] Update `LOAD_EMBEDDINGS_FILE` in `clip_config.py` with the proper Shortcut path such that you can use the pre-run embeddings, rather than loading from scratch. If using Windows ensure that the path uses backslashes instead of forwardslashes. If using THINGSvision features, replace `LOAD_EMBEDDINGS_FILE` with the path to the THINGSvision features instead. (NOT RECOMMENDED - Otherwise, update `LOAD_EMBEDDINGS_FILE` TO `""`.)

### Key stages
In our main file we have three main key stages:

STAGE <1> --> **Loading the fMRI data (labels of the model).** FMRI data is loaded for the ROIs selected in the configuration file, and left and right hemispheres are concatenated.
The Representational Dissimilarity Matrix (RDM) is built according to the dissimilarity metric chosen in the configuration files (1 - Pearson correlation or Euclidean distance).

As a final part of this stage, the first results are calculated and plotted: the pairs of images with the smallest and largest RDM values. These results aim only to understand our dataset and our labels better, and to visualize what do we demand from our ML model.

STAGE <2> --> **Loading the CLIP-ViT embeddings.** Embeddings of the pretrained model are either loaded (we have them saved) or calculated from scratch (in case more/different images are wanted). These saved embeddings can be either the normal CLIP-ViT embeddings (corresponding to the last layer of the model), or the embeddings obtained using the THINGSvision python toolbox to extract specifically the visual layers of the CLIP-ViT model. This is specified in the configuration file as well.

STAGE <3> --> **Training and evaluation of the model.** Data is split into train and test sets, the model is trained, evaluated with the test set, and plotted. Depending on parameters specified in the configuration file, the model will run only for a specified number of hidden layers or for a list of hidden layers, as explained in the above section.

## Data description
In this study, we utilized the Natural Scenes Dataset, which contains high-resolution fMRI data collected from eight participants as they viewed 10,000 natural scene images. Each image was presented  aproximately three times, and the fMRI responses were averaged across repetitions. (Link provided in References.)

## References
Algonauts 2023 Challenge --> Algonauts Project. (n.d.). *Algonauts Project: The Challenge.* Retrieved February 4, 2025, from http://algonauts.csail.mit.edu/challenge.html

Natural Scenes Dataset --> Natural Scenes Dataset. (n.d.). *Natural Scenes Dataset.* Retrieved February 4, 2025, from https://naturalscenesdataset.org/

THINGSvision python toolbox and paper --> Hebart, M. N., & Baker, C. I. (2021). THINGSvision: A Python Toolbox for Streamlining the Extraction of Activations From Deep Neural Networks. *Frontiers in Neuroinformatics,* 15, 679838. https://doi.org/10.3389/fninf.2021.679838

## Instructions for running the project
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

#### [DEPRECATED] First model (Siamese CNN)
```bash 
python3.10 -m src.project.dep_main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```

#### Second model (simple MLP)
```bash 
python3.10 -m src.project.main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```
##### Note - to use THINGSvision features, update the above command for the second model with the optional flag `--thingsvision`

### Modify package dependencies (add/remove/update external modules/packages):
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

### Steps in case of a package failure:
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
