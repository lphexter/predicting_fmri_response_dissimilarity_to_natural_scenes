#############################################
# USER / EXPERIMENTAL PARAMETERS (Pipeline 2)
#############################################
SUBJECT = 1
ROOT_DATA_DIR = "mini_data_for_python"  # "algonauts_2023_tutorial_data"
ROI = "ALL"  # options: [ "ALL", "V1", "V2","V3","hV4","EBA","FBA-1","FBA-2","mTL-bodies","OFA","FFA-1","FFA-2","mTL-faces","aTL-faces","OPA","PPA","RSC","OWFA","VWFA-1","VWFA-2","mfs-words","mTL-words","early","midventral","midlateral","midparietal","ventral","lateral","parietal"]
REGION_CLASS = "None"  # options: [ "Visual", "Bodies", "Faces", "Places", "Words", "Streams" ]
DESIRED_IMAGE_NUMBER = 50

#############################################
# CLIP MODEL
#############################################
PRETRAINED_MODEL = "openai/clip-vit-base-patch32"
LOAD_EMBEDDINGS_FILE = ""

#############################################
# RDM / TRAIN PARAMS
#############################################
METRIC = "correlation"  # "euclidean" or "correlation"
ACCURACY = "r2"  # "spearman", "pearson", or "r2"
ACTIVATION_FUNC = "linear"  # or "sigmoid"
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32
TEST_SIZE = 0.2
HIDDEN_LAYERS = 1
K_FOLD = True  # default to using K-fold cross-validation
K_FOLD_SPLITS = 5

#############################################
# OTHER
#############################################
SWEEP_LAYERS = False  # Whether to sweep over hidden layer sizes
LAYERS_LIST = [0, 1, 2, 3]  # Sweep over these numbers of hidden layers
