# Predicting fMRI Response Dissimilarity to Natural Scenes
**COURSES:**
1. Machine Learning and Neural Networks for Neuroscience
2. Data Science and Advanced Python Concepts Workshop for Neuroscience

BAR-ILAN UNIVERSITY

## Project Documentation
### Project Description
This project did not aim to replicate or directly compare results from existing papers. Instead, we adopted an **exploratory approach** to a novel task in an existing problem space.

The goal of this project is to predict how different our brain response will be to viewing a pair of images representing natural scenes, i.e., **predict dissimilarity between fMRI responses to pairs of images.**

Based on the literature we assume that it is possible to predict a stimulus given an fMRI response (decoding), or vice versa, predicting the fMRI response given a stimulus (encoding).
Moreover, the similarity between brain responses and Machine Learning (ML) models has also been explored, and moderately high correlations have been reported. Specifically, the paper *THINGSvision: A Python Toolbox for Streamlining the Extraction of Activations From Deep Neural Networks* explores the correlation between various pretrained model embeddings and fMRI responses. The authors noted high correlation between a few models, namely CLIP-ViT, and actual fMRI responses, giving us the basis to begin our prediction work.

Based on these assumptions we hypothesized that our novel approach could yield successful results: **to build an ML model capable of predicting how dissimilarly a given brain region in the visual cortex will process two different images.**

A few different ML model architectures were explored, the first consists of a **simple 2-Layer Siamese CNN,** which did not succeed to get encouraging results. The second one consists of a **Multi-Layer Perceptron model** that takes two concatenated embeddings as input, corresponding to each pair of images, which are obtained using a pretrained model. Our third model is a simple SVM for binary classificaiton of our data (similar vs. dissimilar), and our fourth model is a Contrastive Siamese Network (both third and fourth models are represented in our most up-to-date `main.py` file).

**PDF report files:**
1. Machine learning ("A machine learning approach to predict
brain response dissimilarity to pairs of
visual stimuli"): This report builds on top of our first results as detailed in the Python report, focusing on results form the third and fourth models.
2. Python ("[Python Report] Predicting fMRI Response Dissimilarity to Natural Scenes.pdf"): The project background, design and methodology are carefully detailed, up until the approximately mid February. It follows the structure of a scientific report with the following parts: introduction, methodology, results, and conclusion and discussion. As specified in the guidelines, no citations or references are included.

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
  │   │   ├── dep2_main.py
  │   │   ├── logger.py
  │   │   └── main.py
```
</details>

As shown in the directory structure, our project contains three package-like folders for configuration, ML models, and different utilities to use the pretrained model, to preprocess our fMRI data, to prepare our data for the ML model using pytorch, to train our model, and to visualize the results. Additionally, the logger.py file is used for logging, while main.py serves as the main entry point of the project. There are also certain functions which we used for testing, but are not called in main.py as they are not part of the core analysis.

**"dep2_main.py", "dep_main.py", "dep_models.py", and "dep_config.py" are deprecated code and left simply for storage/tracking purposes (NOTE: they are not up-to-date with coding standards)**.

Outside of the source/project/ we have the TOML file which defines the project metadata, dependencies, and tool configurations; the Tox file which automates running linting and tests; and the test files which contain unit tests to validate the project's functionality.

**About the test files:** tests were excluded for any deprecated code, visualizations.py, and main.py (unit tests already cover functions called in main.py). Additionally, magic number checking was excluded in test files.

### Configuration files: important explanations
1. [DEPRECATED] **dep_config.py** --> user configurations for the first/deprecated model
2. **clip_config.py** --> user configurations for the second and third models. In this file, several configurations can be updated: for user/experimental parameters (e.g., subject, data directory, ROIs, ROI classes, desired image number), for the CLIP model (pretrained model to use, path to file), and for the RDM/training parameters, some of which are shared across the second and third models (dissimilarity metric, number of epochs, learning rate, batch size, test size, and number of folds in k-fold cross-validation). Others which are unique to either model:
    1. **Second Model (Simple MLP)**: RDM/training parameters which are unique to this model are the accuracy metric, activation function, number of hidden layers, and whether to use K fold cross validation. Additionally, "SWEEP" to True means the code will loop over the layers in "LAYERS_LIST" after running with the parameterized layer number, HIDDEN_LAYERS.
    2. **Third / Fourth Models** (SVM, Constrastive Siamese Network - most updated `main.py` file):
         1. Distribution type is introduced as a data preprocessing parameter (all = no filtering of images, balanced = balanced distribution of extreme and mid-range values according to the dissimilarity metric, extremes = distribution of only extreme values according to the dissimilarity metric, colors = filtering images based on dominant color either Red, Blue, or Green). With the color option, we also introduced some more parameters: target height and width for resizing images, the color pair comparison of interest (two of Red, Blue, and Green), and an optional string containing a list of color mask files (if color masks are preloaded and saved, making it much more efficient than continually re-running image processing each trial).
         2. We also added one more hyperparameter in the Contrastive Model to tune the percentage of dropout layers, as well as a few parameters for SVM training (kernel, degree, and optional save and load paths for the model).
    4. [**RECOMMENDED**] Update `LOAD_EMBEDDINGS_FILE` in `clip_config.py` with the proper Shortcut path such that you can use the pre-run embeddings, rather than loading from scratch. If using Windows ensure that the path uses backslashes instead of forwardslashes. If using THINGSvision features, replace `LOAD_EMBEDDINGS_FILE` with the path to the THINGSvision features instead. (NOT RECOMMENDED - Otherwise, update `LOAD_EMBEDDINGS_FILE` TO `""`.)

### Key stages
In our main file we have three main key stages:

1. STAGE <1> **Loading the fMRI data (labels of the model).** FMRI data is loaded for the ROIs selected in the configuration file, and left and right hemispheres are concatenated.
    1. The Representational Dissimilarity Matrix (RDM) is built according to the dissimilarity metric chosen in the configuration files (1 - Pearson correlation or Euclidean distance).
    2. As a final part of this stage, the first results are calculated and plotted: the pairs of images with the smallest and largest RDM values. These results aim only to understand our dataset and our labels better, and to visualize what do we demand from our ML model.
2. STAGE <2> **Loading the CLIP-ViT embeddings.** Embeddings of the pretrained model are either loaded (we have them saved) or calculated from scratch (in case more/different images are wanted). These saved embeddings can be either the normal CLIP-ViT embeddings (corresponding to the last layer of the model), or the embeddings obtained using the THINGSvision python toolbox to extract specifically the visual layers of the CLIP-ViT model. This is specified in the configuration file as well.
3. STAGE <3> **Training and evaluation of the model.** Data is split into train and test sets, the model is trained, evaluated with the test set, and plotted. In the most updated main.py file, we train/evaluate both the SVM model for binary classification as well as the Contrastive Siamese Network to predict the actual RDM values.
    1. [Only relevant for Second Model, the MLP] Depending on parameters specified in the configuration file, the model will run only for a specified number of hidden layers or for a list of hidden layers, as explained in the above section.

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
2. Ensure you have added a [Shortcut](https://support.google.com/drive/answer/9700156?hl=en&co=GENIE.Platform%3DDesktop) from the desired data folder to any folder in your Drive. For the data folder, either setup "algonauts_2023_tutorial_data" per instructions [here](https://docs.google.com/forms/d/e/1FAIpQLSehZkqZOUNk18uTjRTuLj7UYmRGz-OkdsU25AyO3Wm6iAb0VA/viewform?usp=sf_link), or request access to the mini testing data "mini_data_for_python" folder [here](https://drive.google.com/drive/folders/19mXhFsOlFWu2vyPkj5In2VQS-Buu4K48?usp=sharing), which contains pre-loaded embeddings and data for Subject 1.
3. Copy the full Shorcut path for use later, e.g. `~/Library/CloudStorage/GoogleDrive-<your_email>/.shortcut-targets-by-id/<shortcut_id>`
4. Ensure your data folder is in "offline mode" locally (Right click the folder, e.g. "mini_data_for_python" > Make available offline).

### Running the Models

#### [DEPRECATED] First model (Siamese CNN)
```bash 
python3.10 -m src.project.dep_main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```

#### [NEWLY DEPRECATED] Second model (simple MLP)
```bash 
python3.10 -m src.project.dep2_main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```
###### Note - to use THINGSvision features, update the above command for the second model with the optional flag `--thingsvision`

#### Third/Fourth models (SVM, Contrastive Siamese Network)

##### Running locally
```bash 
python3.10 -m src.project.main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```
###### Note - to use THINGSvision features, update the above command for the second model with the optional flag `--thingsvision`

##### Running in Google Colab
Additionally, rather than running locally with `main.py`, it can be run with this [Google Colab notebook](https://colab.research.google.com/drive/1ZXQ6ZcRh0BVFiXxHQ_-fqcGrTaB2mAlc) (which imports relevant functions from GitHub and allows the user to update configurations live, while using Colab resources rather than local). You may request access as needed (it is under the same testing data folder [mini_data_for_python](https://drive.google.com/drive/folders/19mXhFsOlFWu2vyPkj5In2VQS-Buu4K48)).

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


### Running Tox - Tests and Linting:
Check formatting, type hinting, lint code & docstrings. Run tests.
```bash
tox
```
