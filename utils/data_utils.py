# utils/data_utils.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

#########################################
#    fMRI DATA LOADING & ROI HANDLING
#########################################

def _get_fmri_voxels(root_data_dir, subj, roi, hemisphere, roi_class):
    challenge_roi_class_dir = os.path.join(
        root_data_dir,
        f'subj0{subj}',
        'roi_masks',
        hemisphere[0] + 'h.' + roi_class + '_challenge_space.npy'
    )
    roi_map_dir = os.path.join(
        root_data_dir,
        f'subj0{subj}',
        'roi_masks',
        'mapping_' + roi_class + '.npy'
    )

    challenge_roi_class = np.load(challenge_roi_class_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
    return challenge_roi

def _get_concatenated_roi_data(root_data_dir, subj, roi, lh_fmri, rh_fmri, desired_image_number):
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = 'prf-visualrois'
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = 'floc-bodies'
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = 'floc-faces'
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = 'floc-places'
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = 'floc-words'
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = 'streams'
    else:
        raise ValueError(f"ROI '{roi}' not recognized in known classes.")

    challenge_roi_rh = _get_fmri_voxels(root_data_dir, subj, roi, 'rh', roi_class)
    roi_data_rh = rh_fmri[:, challenge_roi_rh == 1]

    challenge_roi_lh = _get_fmri_voxels(root_data_dir, subj, roi, 'lh', roi_class)
    roi_data_lh = lh_fmri[:, challenge_roi_lh == 1]

    roi_data_full = np.concatenate((roi_data_lh, roi_data_rh), axis=1)[:desired_image_number]
    return roi_data_full

def prepare_fmri_data(subj, desired_image_number, roi, region_class, root_data_dir):
    fmri_dir = os.path.join(root_data_dir, f'subj0{subj}', 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    rois = []
    if region_class != "None":
        if region_class == "Visual":
            rois = ["V1v","V1d","V2v","V2d","V3v","V3d","hV4"]
        elif region_class == "Bodies":
            rois = ["EBA","FBA-1","FBA-2","mTL-bodies"]
        elif region_class == "Faces":
            rois = ["OFA","FFA-1","FFA-2","mTL-faces","aTL-faces"]
        elif region_class == "Places":
            rois = ["OPA","PPA","RSC"]
        elif region_class == "Words":
            rois = ["OWFA","VWFA-1","VWFA-2","mfs-words","mTL-words"]
        elif region_class == "Streams":
            rois = ["early","midventral","midlateral","midparietal","ventral","lateral","parietal"]
        else:
            raise ValueError(f"Unrecognized region_class '{region_class}'.")
    elif roi != "None":
        if roi in ['V1','V2','V3']:
            rois = [roi+'v', roi+'d']
        else:
            rois = [roi]
    else:
        rois = [
            "V1v","V1d","V2v","V2d","V3v","V3d","hV4","EBA","FBA-1","FBA-2","mTL-bodies",
            "OFA","FFA-1","FFA-2","mTL-faces","aTL-faces","OPA","PPA","RSC","OWFA","VWFA-1",
            "VWFA-2","mfs-words","mTL-words","early","midventral","midlateral","midparietal",
            "ventral","lateral","parietal"
        ]

    roi_data_full = np.array([])
    if (roi == "None" and region_class == "None"):
        roi_data_full = np.concatenate((lh_fmri, rh_fmri), axis=1)[:desired_image_number]
    else:
        for r in rois:
            data_r = _get_concatenated_roi_data(root_data_dir, subj, r, lh_fmri, rh_fmri, desired_image_number)
            roi_data_full = np.concatenate((roi_data_full, data_r), axis=1) if roi_data_full.size else data_r

    scaler = StandardScaler()
    roi_data_full = scaler.fit_transform(roi_data_full)
    return roi_data_full


#########################################
#     IMAGE PREPROCESSING
#########################################

def preprocess_images(image_dir, num_images, new_width, new_height):
    def min_max_norm(X):
        X = np.array(X, dtype=np.float32)
        return (X - X.min()) / (X.max() - X.min() + 1e-8)

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    )
    image_data = []

    for image_file in tqdm(image_files[:num_images], desc="Resizing images"):
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path).convert('L')
        img = img.resize((new_width, new_height))
        image_data.append(np.array(img, dtype=np.float32))

    X = min_max_norm(image_data)
    return X


#########################################
#     RDM (Representational Dissimilarity)
#########################################

def create_rdm(roi_data, metric='correlation'):
    distances = pdist(roi_data, metric=metric)
    rdm = squareform(distances)
    return rdm

def create_rdm_from_vectors(vectors):
    num_images = int(np.sqrt(len(vectors) * 2)) + 1
    rdm_out = np.zeros((num_images, num_images))
    idx = 0
    for i in range(num_images):
        for j in range(i+1, num_images):
            if idx < len(vectors):
                rdm_out[i, j] = rdm_out[j, i] = vectors[idx]
            idx += 1
    return rdm_out
