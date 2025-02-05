#############################################
# USER / EXPERIMENTAL PARAMETERS (Pipeline 1)
#############################################
SUBJECT = 1  # e.g., 1, 2, ... up to 8
DESIRED_IMAGE_NUMBER = 5  # how many images to use
ROI = "V1"  # options: [ "V1", "V2","V3","hV4","EBA","FBA-1","FBA-2","mTL-bodies","OFA","FFA-1","FFA-2","mTL-faces","aTL-faces","OPA","PPA","RSC","OWFA","VWFA-1","VWFA-2","mfs-words","mTL-words","early","midventral","midlateral","midparietal","ventral","lateral","parietal"]
REGION_CLASS = "None"  # e.g., "Visual", "Bodies", "Faces", "Places", "Words", "Streams"
ROOT_DATA_DIR = "mini_data_for_python"  # "algonauts_2023_tutorial_data"

# RDM, model, and training parameters
METRIC = "correlation"  # e.g. 'correlation' or 'euclidean'
ACCURACY = "spearman"  # e.g. 'spearman', 'pearson', 'r2'
ACTIVATION_FUNC = "sigmoid"  # e.g. 'linear' or 'sigmoid'
BATCH_SIZE = 32
NEW_WIDTH = 128
NEW_HEIGHT = 128
NUM_CHANNELS = 1  # 1 for grayscale
TEST_SIZE = 0.2  # Train/test split ratio
EPOCHS = 1
ALPHA = 0.5  # Weight for correlation vs MSE in custom loss
