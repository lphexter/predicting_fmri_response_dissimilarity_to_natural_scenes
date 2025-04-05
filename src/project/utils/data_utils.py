import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..config import clip_config
from ..logger import logger
from ..utils.visualizations import show_image_pair

#    CONSTANTS FOR COLOR CLASSIFICATION
COLOR_MAP = {"Blue": np.array([0, 0, 255]), "Red": np.array([255, 0, 0]), "Green": np.array([0, 255, 0])}


#########################################
#    fMRI DATA LOADING & ROI HANDLING
#########################################


def _get_fmri_voxels(root_data_dir, subj, roi, hemisphere, roi_class):
    """Gets indices for fMRI voxels for a specified brain ROI and hemisphere.

    Args:
        root_data_dir: Root directory of the data.
        subj: Subject number.
        roi: Region of interest (e.g., 'V1v').
        hemisphere: Hemisphere ('lh' or 'rh').
        roi_class: ROI class (e.g., 'prf-visualrois').

    Returns:
        Mask array for fMRI data of the specified ROI and hemisphere.
    """
    challenge_roi_class_dir = os.path.join(
        root_data_dir, f"subj0{subj}", "roi_masks", hemisphere[0] + "h." + roi_class + "_challenge_space.npy"
    )
    roi_map_dir = os.path.join(root_data_dir, f"subj0{subj}", "roi_masks", "mapping_" + roi_class + ".npy")

    # load the challenge space data- an array representing the brain with each element corresponding to a voxel and labeled with its ROI class
    challenge_roi_class = np.load(challenge_roi_class_dir)
    # load the ROI mapping dictionary (so we can translate human-readable ROI names to numerical indices)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    #  use roi_map dictionary to find the numerical index associated with the input roi (e.g., 'V1v').
    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    # compare the challenge_roi_class data (the brain array) with the roi_mapping (the numerical index of our target ROI)
    # where they match, the mask is set to 1 (true), indicating the voxel belongs to the ROI, otherwise, it's set to 0 (false)
    # return the mask array
    return np.asarray(challenge_roi_class == roi_mapping, dtype=int)


def _get_concatenated_roi_data(root_data_dir, subj, roi, lh_fmri, rh_fmri, desired_image_number):  # noqa: PLR0913
    """Gets the concatenated fMRI data for a specified ROI from both hemispheres.

    Args:
        root_data_dir (str): Root data directory
        subj (str): Subject number (1-8)
        roi (str): Region of interest (e.g., 'V1v').
        lh_fmri (np.ndarray): Left hemisphere fMRI data.
        rh_fmri (np.ndarray): Right hemisphere fMRI data.
        desired_image_number (int): Data for how many images the user wants

    Returns:
        np.ndarray: Concatenated fMRI data for the specified ROI.
    """
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = "prf-visualrois"
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = "floc-bodies"
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = "floc-faces"
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = "floc-places"
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = "floc-words"
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = "streams"
    elif roi == "ALL":
        return np.concatenate((lh_fmri, rh_fmri), axis=1)[:desired_image_number]
    else:
        raise ValueError(f"ROI '{roi}' not recognized in known classes.")

    challenge_roi_rh = _get_fmri_voxels(root_data_dir, subj, roi, "rh", roi_class)
    roi_data_rh = rh_fmri[:, challenge_roi_rh == 1]

    challenge_roi_lh = _get_fmri_voxels(root_data_dir, subj, roi, "lh", roi_class)
    roi_data_lh = lh_fmri[:, challenge_roi_lh == 1]

    return np.concatenate((roi_data_lh, roi_data_rh), axis=1)[:desired_image_number]


def prepare_fmri_data(  # noqa: PLR0912, C901
    root_data_dir,
    subj=clip_config.SUBJECT,
    desired_image_number=clip_config.DESIRED_IMAGE_NUMBER,
    roi=clip_config.ROI,
    region_class=clip_config.REGION_CLASS,
):
    """Loads and processes fMRI data for a given subject and region of interest (ROI).

    Args:
        root_data_dir (str): Root directory containing the fMRI dataset.
        subj (int, optional): Subject number.
        desired_image_number (int, optional): Number of images to process.
        roi (str, optional): Specific region of interest.
        region_class (str, optional): Category of brain regions.

    Returns:
        np.ndarray: Scaled fMRI data corresponding to the selected ROIs.
    """
    fmri_dir = os.path.join(root_data_dir, f"subj0{subj}", "training_split", "training_fmri")
    lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
    rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))

    rois = []
    if region_class != "None":
        if region_class == "Visual":
            rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
        elif region_class == "Bodies":
            rois = ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]
        elif region_class == "Faces":
            rois = ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]
        elif region_class == "Places":
            rois = ["OPA", "PPA", "RSC"]
        elif region_class == "Words":
            rois = ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
        elif region_class == "Streams":
            rois = ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
        else:
            raise ValueError(f"Unrecognized region_class '{region_class}'.")
    elif roi != "None":
        rois = [roi + "v", roi + "d"] if roi in ["V1", "V2", "V3"] else [roi]
    else:
        rois = [
            "V1v",
            "V1d",
            "V2v",
            "V2d",
            "V3v",
            "V3d",
            "hV4",
            "EBA",
            "FBA-1",
            "FBA-2",
            "mTL-bodies",
            "OFA",
            "FFA-1",
            "FFA-2",
            "mTL-faces",
            "aTL-faces",
            "OPA",
            "PPA",
            "RSC",
            "OWFA",
            "VWFA-1",
            "VWFA-2",
            "mfs-words",
            "mTL-words",
            "early",
            "midventral",
            "midlateral",
            "midparietal",
            "ventral",
            "lateral",
            "parietal",
        ]

    roi_data_full = np.array([])
    if roi == "None" and region_class == "None":
        roi_data_full = np.concatenate((lh_fmri, rh_fmri), axis=1)[:desired_image_number]
    else:
        for r in rois:
            data_r = _get_concatenated_roi_data(root_data_dir, subj, r, lh_fmri, rh_fmri, desired_image_number)
            roi_data_full = np.concatenate((roi_data_full, data_r), axis=1) if roi_data_full.size else data_r

    scaler = StandardScaler()
    fmri_data = scaler.fit_transform(roi_data_full)

    try:
        if fmri_data is None or not isinstance(fmri_data, np.ndarray):
            msg = "Invalid fMRI data loaded."
            raise ValueError(msg)
    except (TypeError, ValueError) as e:
        logger.error("Error loading fMRI data: %s", e)
        sys.exit(1)

    logger.info("fMRI data shape: %s", fmri_data.shape)

    return fmri_data


#########################################
#     RDM (Representational Dissimilarity Matrix)
#########################################


# return high, low, and closest to 1 values of the RDM along with their pairs
def analyze_rdm(rdm, images, metric=clip_config.METRIC):
    """Analyzes the Representational Dissimilarity Matrix (RDM).

    Identifies the highest, lowest, and (if applicable) closest-to-1 dissimilarity values.

    Args:
        rdm (np.ndarray): The computed RDM matrix.
        images (list): List of images corresponding to the dataset.
        metric (str, optional): Similarity metric used.
    """
    try:
        all_metrics = {}

        triu_indices = np.triu_indices_from(rdm, k=1)
        upper_tri_values = rdm[triu_indices]

        # 1. Lowest value and corresponding indices
        lowest_value = np.min(upper_tri_values)
        lowest_idx = np.where(rdm == lowest_value)

        # 2. Highest value and corresponding indices
        highest_value = np.max(upper_tri_values)
        highest_idx = np.where(rdm == highest_value)

        # Convert from 2D matrix indices to image indices
        lowest_pair = lowest_idx[0]
        highest_pair = highest_idx[0]

        all_metrics["low"] = {"value": lowest_value, "pair": lowest_pair}
        all_metrics["high"] = {"value": highest_value, "pair": highest_pair}

        # 3. Only for correlation, calculate value closest to 1
        if metric == "correlation":
            closest_to_1_value = upper_tri_values[np.argmin(np.abs(upper_tri_values - 1))]
            closest_to_1_idx = np.where(rdm == closest_to_1_value)
            closest_to_1_pair = closest_to_1_idx[0]
            all_metrics["closest_to_1"] = {"value": closest_to_1_value, "pair": closest_to_1_pair}

        if not isinstance(all_metrics, dict) or len(all_metrics) == 0:
            raise ValueError("Invalid RDM analysis results.")

        logger.info("RDM Value Analysis Results: %s", all_metrics)

        for key, value in all_metrics.items():
            if "pair" not in value or "value" not in value:
                raise KeyError(f"Missing expected keys in all_metrics for {key}")
            title = f"Pair images of {key} value with score {value['value']}"
            image_pair = value["pair"]
            show_image_pair(image_pair[0], image_pair[1], images, title)

    except (ValueError, KeyError) as e:
        logger.error("Error analyzing RDM: %s", e)
        sys.exit(1)


# not used in main.py - for initial testing, kept for tracking
def compare_rdms(raw_rdm, features_rdm):
    """Computes Pearson and Spearman correlation coefficients between two RDMs.

    Args:
        raw_rdm (np.ndarray): Ground truth RDM.
        features_rdm (np.ndarray): RDM computed from extracted features.
    """
    upper_tri_indices = np.triu_indices(features_rdm.shape[0], k=1)

    features_upper_tri = features_rdm[upper_tri_indices]
    rdm_upper_tri = raw_rdm[upper_tri_indices]

    corr, _ = pearsonr(features_upper_tri, rdm_upper_tri)
    logger.info("Pearson Correlation between features RDM (upper tri) and true RDM (upper tri): %s", corr)

    corr, _ = spearmanr(features_upper_tri, rdm_upper_tri)
    logger.info("Spearman Correlation between features RDM (upper tri) and true RDM (upper tri): %s", corr)


def create_rdm(roi_data, metric="correlation"):
    """Creates an RDM from ROI data using a specified metric.

    Args:
        roi_data (np.ndarray): Data extracted from the selected brain regions.
        metric (str, optional): Distance metric for RDM computation.

    Returns:
        np.ndarray: The computed RDM.
    """
    distances = pdist(roi_data, metric=metric)
    rdm = squareform(distances)
    try:
        if rdm is None or not isinstance(rdm, np.ndarray):
            msg = "Invalid RDM created."
            raise ValueError(msg)
    except ValueError as e:
        logger.error("Error creating RDM: %s", e)
        sys.exit(1)
    return rdm


def create_binary_rdm(rdm, metric="correlation"):
    """Creates a binary RDM from a continuous RDM based on the specified metric.

    For 'correlation', values < 1 are considered similar and values >= 1 are dissimilar.
    For 'euclidean', a threshold is calculated from the data (here, the median of the
    non-zero values is used to roughly split the data into two equal groups).

    Args:
        rdm: A NumPy array representing the continuous RDM.
        metric: A string indicating which metric to use ('correlation' or 'euclidean').

    Returns:
        A binary RDM (NumPy array) with entries labeled as 'similar' or 'dissimilar'.
    """
    if metric == "correlation":
        threshold = 1
    elif metric == "euclidean":
        # Exclude zero values (which often correspond to self-similarity)
        valid_values = rdm[rdm > 0]
        if valid_values.size == 0:
            raise ValueError("No non-zero values found in the RDM to compute a threshold.")
        threshold = np.median(valid_values)
    else:
        raise ValueError("Unsupported metric. Please use 'correlation' or 'euclidean'.")

    return np.where(rdm > threshold, "dissimilar", "similar")


#############################################
#     IMAGE PROCESSING AND CLASSIFICATION
#############################################


def preprocess_images(image_dir, num_images, new_width, new_height, grayscale=True):  # noqa: FBT002
    """Loads, resizes, and normalizes images from a directory.

    Args:
        image_dir (str): Path to the image directory.
        num_images (int): Number of images to process.
        new_width (int): Width to resize images to.
        new_height (int): Height to resize images to.
        grayscale (boolean): Convert images to grayscale, default to True.

    Returns:
        np.ndarray: Normalized image data.
    """

    def min_max_norm(X):  # noqa: N803
        X = np.array(X, dtype=np.float32)
        return (X - X.min()) / (X.max() - X.min() + 1e-8)

    image_files = sorted(f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)))  # noqa: PTH113
    image_data = []

    for image_file in tqdm(image_files[:num_images], desc="Resizing images"):
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path)
        if grayscale:
            img = img.convert("L")
        img = img.resize((new_width, new_height))
        image_data.append(np.array(img, dtype=np.float32))

    return min_max_norm(image_data)


def closest_color(pixel, color_map):
    """Return color classification

    Given a pixel (R, G, B) and a dictionary of named colors (RGB),
    returns the name of the color that is closest in Euclidean distance.
    """
    distances = {}
    for color_name, color_rgb in color_map.items():
        dist = np.linalg.norm(pixel - color_rgb)
        distances[color_name] = dist

    # Return the color with the minimum distance
    return min(distances, key=distances.get)


def classify_images_rgb(images, threshold=0.7):
    """Classifies a list of images (NumPy arrays) to either Blue(0), Red(1), or Green(2), or Unclassified (-1)

    Based on the majority of pixels that are closest to each color definition.
    If the fraction of the dominant color is below 'threshold', the image is assigned -1 (unclassified).

    :param images: list of (H, W, 3) NumPy arrays (BGR or RGB).
    :param threshold: float, the minimum fraction of pixels required to classify
                      the image as a specific color.
    :return: A NumPy array (or list) of length len(images), where each index i
             contains 0,1,2 according to the dominant color (Blue/Red/Green),
             or -1 if the image is unclassified.
    """
    # Prepare an array for labels, initialized to -1 (meaning "unclassified")
    labels = np.full(len(images), -1, dtype=int)

    for idx, img in enumerate(images):
        # Convert to float for distance calculations, reshape to (N, 3) because (R, G, B)
        pixels = img.reshape(-1, 3).astype(np.float32)
        total_pixels = len(pixels)

        # Count how many pixels are closest to each color
        color_counts = {"Blue": 0, "Red": 0, "Green": 0}

        for pixel in pixels:
            color_name = closest_color(pixel, COLOR_MAP)
            color_counts[color_name] += 1

        # Find which color is the majority for this image
        dominant_color = max(color_counts, key=color_counts.get)
        dominant_count = color_counts[dominant_color]

        # Calculate the fraction of pixels for the dominant color
        fraction_dominant = dominant_count / total_pixels

        if fraction_dominant >= threshold:
            labels[idx] = clip_config.COLOR_TO_LABEL[dominant_color]
        else:
            labels[idx] = -1

    return labels


def load_color_map_files(color_map_files, root_data_dir):
    """Loads and concatenates color map files with pre-validation to minimize per-iteration try/except overhead.

    Args:
        color_map_files (str): Comma-separated file paths for np.load.
        root_data_dir (str): Directory path of all color mask files.

    Returns:
        np.ndarray: Concatenated color maps.

    Raises:
        FileNotFoundError: If any file does not exist.
        ValueError: If the loaded color maps cannot be concatenated.
        Exception: For any other unexpected errors during file loading or concatenation.
    """
    # Build full file paths and validate existence upfront.
    file_list = [os.path.join(root_data_dir, f.strip()) for f in color_map_files.split(",")]
    missing_files = [file for file in file_list if not os.path.isfile(file)]  # noqa: PTH113
    if missing_files:
        raise FileNotFoundError(f"The following files were not found: {missing_files}")

    try:
        # Load all color maps.
        color_maps = [np.load(file) for file in file_list]
        # Concatenate the loaded color maps.
        return np.concatenate(color_maps, axis=0)
    except ValueError as e:
        raise ValueError(f"Error concatenating color maps: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error loading files: {e}") from e  # noqa: TRY002


def get_equal_color_data(embeddings, roi_data, color_mask_list, desired_colors):
    """Given two color classes, return embeddings and raw data cut to an equal amount per class.

    Given input embeddings, ROI data, a comma-separated string of color map file paths,
    and a tuple of two desired colors (e.g. ('R', 'G') or ('B', 'R')),
    this function:
      - Loads and concatenates the color maps.
      - Finds indices corresponding to each desired color.
      - Trims each set of indices to the same (minimum) length.
      - Selects the corresponding embeddings and ROI data.
      - Returns the concatenated embeddings and ROI data.

    Args:
        embeddings (np.ndarray): Array of shape (n_samples, feature_dim)
        roi_data (np.ndarray): Array of shape (n_samples, roi_feature_dim)
        color_mask_list(str): Comma-separated file paths for np.load.
        desired_colors (tuple): Tuple of two color labels (e.g. ('R', 'G')).

    Returns:
        combined_embeddings (np.ndarray): Array of shape (2*N, feature_dim)
        combined_roi_data (np.ndarray): ROI data concatenated vertically (shape (2*N, roi_feature_dim))
    """
    # R = 1, B = 0, G = 2; convert tuple to integers based on map
    desired_colors = [1 if color == "R" else 0 if color == "B" else 2 for color in desired_colors]

    # Extract indices for each desired color
    color1, color2 = desired_colors
    indices1 = np.where((color_mask_list == color1) & (np.arange(len(color_mask_list)) < embeddings.shape[0]))[0]
    indices2 = np.where((color_mask_list == color2) & (np.arange(len(color_mask_list)) < embeddings.shape[0]))[0]

    # Determine the minimum count to equalize the groups
    min_count = min(len(indices1), len(indices2))
    indices1 = indices1[:min_count]
    indices2 = indices2[:min_count]
    combined_indices = np.concatenate((indices1, indices2), axis=0)

    # Select embeddings corresponding to each color
    combined_embeddings = embeddings[combined_indices, :]

    # Select ROI data corresponding to each color
    combined_roi_data = roi_data[combined_indices, :]

    return combined_roi_data, combined_embeddings


#############################################
#     DATA PREP FUNCTIONS FOR DEPRECATED MODEL
#############################################


def prepare_data_for_cnn(rdm, test_size=0.2):
    """Prepares data pairs and labels for training a CNN using the RDM.

    Args:
        rdm (np.ndarray): The computed RDM matrix.
        test_size (float, optional): Fraction of data used for testing.

    Returns:
        tuple: Training and testing indices for the CNN model.
    """
    from sklearn.model_selection import train_test_split

    num_images = rdm.shape[0]
    row_indices, col_indices = np.triu_indices(num_images, k=1)
    rdm_values = rdm[row_indices, col_indices]

    train_indices, test_indices, y_train, y_test = train_test_split(
        np.arange(len(rdm_values)), rdm_values, test_size=test_size, random_state=42
    )

    X_train_indices = (row_indices[train_indices], col_indices[train_indices])
    X_test_indices = (row_indices[test_indices], col_indices[test_indices])
    return X_train_indices, X_test_indices, y_train, y_test


def data_generator(image_data, pair_indices, y_data, batch_size=clip_config.BATCH_SIZE):
    """Generator function that yields batches of paired image data and labels for training.

    Args:
        image_data (np.ndarray): The dataset of image data.
        pair_indices (tuple): Tuple containing row and column indices for image pairs.
        y_data (np.ndarray): Labels corresponding to image pairs.
        batch_size (int, optional): Number of samples per batch.

    Yields:
        tuple: Batch of paired images and labels.
    """
    num_samples = len(y_data)
    row_indices, col_indices = pair_indices

    while True:
        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size

            batch_rows = row_indices[offset:end]
            batch_cols = col_indices[offset:end]
            batch_x1 = image_data[batch_rows]
            batch_x2 = image_data[batch_cols]
            batch_y = y_data[offset:end]

            # Add channel dimension
            batch_x1 = batch_x1[..., np.newaxis]
            batch_x2 = batch_x2[..., np.newaxis]

            yield (batch_x1, batch_x2), batch_y


# Create an RDM from a row vector of predictions for visualization/analysis
def create_rdm_from_vectors(vectors):
    """Converts a 1D vector of pairwise dissimilarity values into an RDM matrix.

    Args:
        vectors (np.ndarray): Pairwise dissimilarity values.

    Returns:
        np.ndarray: Square RDM matrix.
    """
    num_images = int(np.sqrt(len(vectors) * 2)) + 1
    rdm_out = np.zeros((num_images, num_images))
    idx = 0
    for i in range(num_images):
        for j in range(i + 1, num_images):
            if idx < len(vectors):
                rdm_out[i, j] = rdm_out[j, i] = vectors[idx]
            idx += 1
    return rdm_out
