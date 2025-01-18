#############################################
# USER / EXPERIMENTAL PARAMETERS (Pipeline 1)
#############################################
SUBJECT = 1  # e.g., 1, 2, ... up to 8
DESIRED_IMAGE_NUMBER = 500  # how many images to use
ROI = "V1v"  # e.g., "V1v" or "None"
REGION_CLASS = "None"  # e.g., "Visual", "Faces", etc.
ROOT_DATA_DIR = "algonauts_2023_tutorial_data"

# RDM, model, and training parameters
METRIC = "correlation"  # e.g. 'correlation' or 'euclidean'
ACCURACY = "spearman"  # e.g. 'spearman', 'pearson', 'r2'
ACTIVATION_FUNC = "sigmoid"  # e.g. 'linear' or 'sigmoid'
BATCH_SIZE = 32
NEW_WIDTH = 128
NEW_HEIGHT = 128
NUM_CHANNELS = 1  # 1 for grayscale
TEST_SIZE = 0.2  # Train/test split ratio
EPOCHS = 10
ALPHA = 0.5  # Weight for correlation vs MSE in custom loss
