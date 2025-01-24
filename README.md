# biu_python_final_project
BIU python class final project. Code is primarily run in Google Colab, link here: <INSERT_LINK>

## Directory structure
<details>
<summary>Directory Structure</summary>

```bash
biu_python_final_project/
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
  │   │   ├── dep_main.py
  │   │   └── main.py
```
</details>

## On first run:
```bash 
# install Virtualenv - a tool to set up your Python environments
pip install virtualenv
# create virtual environment (serve only this project):
python3.10 -m venv venv
# activate virtual environment
source venv/bin/activate
+ (venv) should appear as prefix to all command (run next command just after activating venv)
# update venv's python package-installer (pip) to its latest version
pip install --upgrade pip
# install projects packages
pip install -e .[dev]
``` 

### Setup your data folder
1. Download Google Drive for Desktop [here](https://support.google.com/a/users/answer/13022292?hl=en)
2. Ensure you have added a Shortcut from the desired data folder to any folder in your Drive. For the data folder, either setup "algonauts_2023_tutorial_data" per instructions [here](https://docs.google.com/forms/d/e/1FAIpQLSehZkqZOUNk18uTjRTuLj7UYmRGz-OkdsU25AyO3Wm6iAb0VA/viewform?usp=sf_link), or request access to the mini testing data "mini_data_for_python" folder [here](https://drive.google.com/drive/folders/19mXhFsOlFWu2vyPkj5In2VQS-Buu4K48?usp=sharing)
3. Copy the full Shorcut path for use later, e.g. `~/Library/CloudStorage/GoogleDrive-<your_email>/.shortcut-targets-by-id/<shortcut_id>`

## Running the Models

### Configuration files:
1. [DEPRECATED] dep_config.py --> user configurations for the first model
2. clip_config.py --> user configurations for the second model; NOTE: "SWEEP" to True means that you will loop over the layers in "LAYERS_LIST" after it finishes running with the parameterized layer number, HIDDEN_LAYERS

### [DEPRECATED] First model (Siamese CNN)
```bash 
python3.10 -m src.project.dep_main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```

### Second model (simple MLP)
```bash 
python3.10 -m src.project.main --root_dir "/absolute/path/to/Google/Drive/Shortcut"
```

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