#############################################
# USER / EXPERIMENTAL PARAMETERS (Pipeline 2)
#############################################
SUBJECT = 1  # options: [ "1", "2", "3", "4", "5", "6", "7", "8"]
ROOT_DATA_DIR = "mini_data_for_python"  # "algonauts_2023_tutorial_data"
ROI = "ALL"  # options: [ "ALL", "V1", "V2","V3","hV4","EBA","FBA-1","FBA-2","mTL-bodies","OFA","FFA-1","FFA-2","mTL-faces","aTL-faces","OPA","PPA","RSC","OWFA","VWFA-1","VWFA-2","mfs-words","mTL-words","early","midventral","midlateral","midparietal","ventral","lateral","parietal"]
REGION_CLASS = "None"  # options: [ "Visual", "Bodies", "Faces", "Places", "Words", "Streams" ]
DESIRED_IMAGE_NUMBER = 20  # if using mini_data_for_python, max number is 50
DISTRIBUTION_TYPE = "all"  # options: ["all","balanced","extremes","colors"]

#############################################
# COLOR IMAGE PROCESSING/LOADING
#############################################
NEW_WIDTH = 128
NEW_HEIGHT = 128
COLOR_ARRAY_MAP_FILES = "map_colors_subj1_first1000Images_3Colors.npy, map_colors_subj1_1000_to_2000Images_3Colors.npy"  # or "" to calculate color classification from scratch
# Color pair selection - can be 2 of 'R', 'G', 'B'
COLOR_PAIR = ("R", "B")

#############################################
# CLIP MODEL
#############################################
PRETRAINED_MODEL = "openai/clip-vit-base-patch32"
LOAD_EMBEDDINGS_FILE = "/absolute/path/to/Google/Drive/Shortcut/mini_data_for_python/embeddings_2500_1.npy"

#############################################
# RDM / TRAIN PARAMS
#############################################
METRIC = "correlation"  # "euclidean" or "correlation"
ACCURACY = "r2"  # "spearman", "pearson", or "r2"
ACTIVATION_FUNC = "linear"  # or "sigmoid"
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32
TEST_SIZE = 0.35
DROPOUT_PERCENTAGE = 0.5
HIDDEN_LAYERS = 1
K_FOLD = True  # default to using K-fold cross-validation
K_FOLD_SPLITS = 5
# SVM
KERNEL = "poly"  # options: ["linear","poly","rbf","sigmoid"]
DEGREE = 2  # polynomial degree for SVM
LOAD_SVM_MODEL_PATH = ""  # e.g. "svm_model__euclidean_red_blue_2000Images.pkl"
SAVE_SVM_MODEL_PATH = ""  # e.g. "svm_model__euclidean_red_blue_2000Images.pkl"

#############################################
# OTHER
#############################################
SWEEP_LAYERS = False  # Whether to sweep over hidden layer sizes
LAYERS_LIST = [0, 1, 2, 3]  # Sweep over these numbers of hidden layers
