import os

import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..config.clip_config import METRIC

#########################################
#    fMRI DATA LOADING & ROI HANDLING
#########################################


def _get_fmri_voxels(root_data_dir, subj, roi, hemisphere, roi_class):
    challenge_roi_class_dir = os.path.join(
        root_data_dir, f"subj0{subj}", "roi_masks", hemisphere[0] + "h." + roi_class + "_challenge_space.npy"
    )
    roi_map_dir = os.path.join(root_data_dir, f"subj0{subj}", "roi_masks", "mapping_" + roi_class + ".npy")

    challenge_roi_class = np.load(challenge_roi_class_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    return np.asarray(challenge_roi_class == roi_mapping, dtype=int)  # challenge_roi


def _get_concatenated_roi_data(root_data_dir, subj, roi, lh_fmri, rh_fmri, desired_image_number):  # noqa: PLR0913
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
        raise ValueError(f"ROI '{roi}' not recognized in known classes.")  # noqa: TRY003, EM102

    challenge_roi_rh = _get_fmri_voxels(root_data_dir, subj, roi, "rh", roi_class)
    roi_data_rh = rh_fmri[:, challenge_roi_rh == 1]

    challenge_roi_lh = _get_fmri_voxels(root_data_dir, subj, roi, "lh", roi_class)
    roi_data_lh = lh_fmri[:, challenge_roi_lh == 1]

    return np.concatenate((roi_data_lh, roi_data_rh), axis=1)[:desired_image_number]


def prepare_fmri_data(subj, desired_image_number, roi, region_class, root_data_dir):  # noqa: C901, PLR0912
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
            raise ValueError(f"Unrecognized region_class '{region_class}'.")  # noqa: TRY003, EM102
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
    return scaler.fit_transform(roi_data_full)


#########################################
#     IMAGE PREPROCESSING
#########################################


def preprocess_images(image_dir, num_images, new_width, new_height):
    def min_max_norm(X):  # noqa: N803
        X = np.array(X, dtype=np.float32)
        return (X - X.min()) / (X.max() - X.min() + 1e-8)

    image_files = sorted(f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)))  # noqa: PTH113
    image_data = []

    for image_file in tqdm(image_files[:num_images], desc="Resizing images"):
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path).convert("L")
        img = img.resize((new_width, new_height))
        image_data.append(np.array(img, dtype=np.float32))

    return min_max_norm(image_data)


#########################################
#     RDM (Representational Dissimilarity Matrix)
#########################################


def create_rdm(roi_data, metric="correlation"):
    distances = pdist(roi_data, metric=metric)
    return squareform(distances)


def create_rdm_from_vectors(vectors):
    num_images = int(np.sqrt(len(vectors) * 2)) + 1
    rdm_out = np.zeros((num_images, num_images))
    idx = 0
    for i in range(num_images):
        for j in range(i + 1, num_images):
            if idx < len(vectors):
                rdm_out[i, j] = rdm_out[j, i] = vectors[idx]
            idx += 1
    return rdm_out


def compare_rdms(raw_rdm, features_rdm):
    # calculate correlation between features rdm and true rdm
    # Get the upper triangular part (excluding the diagonal)
    upper_tri_indices = np.triu_indices(features_rdm.shape[0], k=1)

    # Extract the upper diagonal elements from both RDMs
    features_upper_tri = features_rdm[upper_tri_indices]
    rdm_upper_tri = raw_rdm[upper_tri_indices]

    # Now calculate the Pearson correlation
    corr, _ = pearsonr(features_upper_tri, rdm_upper_tri)
    print(f"Pearson Correlation between features RDM (upper tri) and true RDM (upper tri): {corr}")

    # Now calculate the Spearman correlation
    corr, _ = spearmanr(features_upper_tri, rdm_upper_tri)
    print(f"Spearman Correlation between features RDM (upper tri) and true RDM (upper tri): {corr}")


def analyze_rdm(rdm, metric=METRIC):
    all_metrics = {}

    # Get indices of the upper triangular part (excluding diagonal)
    triu_indices = np.triu_indices_from(rdm, k=1)

    # Extract the upper triangular values
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

    if metric == "correlation":
        closest_to_1_value = upper_tri_values[np.argmin(np.abs(upper_tri_values - 1))]
        closest_to_1_idx = np.where(rdm == closest_to_1_value)
        closest_to_1_pair = closest_to_1_idx[0]
        all_metrics["closest_to_1"] = {"value": closest_to_1_value, "pair": closest_to_1_pair}

    return all_metrics


#############################################
#     DATA PREP FUNCTIONS FOR DEPRECATED MODEL
#############################################


def prepare_data_for_cnn(rdm, test_size=0.2):
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


def data_generator(image_data, pair_indices, y_data, batch_size=32):
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
