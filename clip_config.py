# clip_config.py

#############################################
# USER / EXPERIMENTAL PARAMETERS (Pipeline 2)
#############################################
SUBJECT = 1
ROOT_DATA_DIR = 'algonauts_2023_tutorial_data'
ROI = "V1"
REGION_CLASS = "None"
DESIRED_IMAGE_NUMBER = 1000

#############################################
# CLIP MODEL
#############################################
PRETRAINED_MODEL = "openai/clip-vit-base-patch32"

#############################################
# RDM / TRAIN PARAMS
#############################################
METRIC = "euclidean"       # or "correlation"
ACCURACY = "r2"            # "spearman", "pearson", "r2"
ACTIVATION_FUNC = "linear" # or "sigmoid"
EPOCHS = 20
LEARNING_RATE = 0.05
BATCH_SIZE = 32
TEST_SIZE = 0.2
HIDDEN_LAYERS = 1

#############################################
# OTHER
#############################################
SWEEP_LAYERS = True  # Whether to sweep over hidden layer sizes
