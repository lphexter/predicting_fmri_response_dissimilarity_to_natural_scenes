import numpy as np
import pytest
from PIL import Image

from project.config import clip_config
from project.utils import data_utils
from project.utils.data_utils import (
    COLOR_MAP,
    _get_concatenated_roi_data,
    _get_fmri_voxels,
    analyze_rdm,
    classify_images_rgb,
    closest_color,
    create_binary_rdm,
    create_rdm,
    create_rdm_from_vectors,
    data_generator,
    get_equal_color_data,
    load_color_map_files,
    prepare_data_for_cnn,
    prepare_fmri_data,
    preprocess_images,
)

#########################################
# NEW TESTS FOR UPDATED CODE FUNCTIONS
#########################################


def test_create_binary_rdm_correlation():
    """Test create_binary_rdm for 'correlation' metric.

    For correlation, values < 1 should be 'similar' and values >= 1 'dissimilar'.
    """
    rdm = np.array([[0, 0.5, 1.2], [0.5, 0, 0.8], [1.2, 0.8, 0]])
    binary_rdm = create_binary_rdm(rdm, metric="correlation")
    expected = np.array(
        [["similar", "similar", "dissimilar"], ["similar", "similar", "similar"], ["dissimilar", "similar", "similar"]]
    )
    np.testing.assert_array_equal(binary_rdm, expected)


def test_create_binary_rdm_euclidean():
    """Test create_binary_rdm for 'euclidean' metric.

    The threshold is set as the median of non-zero values.
    """
    rdm = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]])
    # Non-zero values: [2, 4, 2, 6, 4, 6] → median = 4.
    binary_rdm = create_binary_rdm(rdm, metric="euclidean")
    # Values > 4 -> "dissimilar", values <= 4 -> "similar"
    expected = np.array(
        [["similar", "similar", "similar"], ["similar", "similar", "dissimilar"], ["similar", "dissimilar", "similar"]]
    )
    np.testing.assert_array_equal(binary_rdm, expected)


def test_closest_color():
    """Test closest_color for both a clearly dominant color and an ambiguous case.

    For a clearly dominant pixel:
      - The pixel [250, 10, 10] should be closest to "Red" in COLOR_MAP.

    For an ambiguous pixel:
      - The pixel [100, 100, 100] is equally distant from all colors, so the function should return None.
    """
    # Test a clearly dominant color
    pixel_red = np.array([250, 10, 10])
    result_red = closest_color(pixel_red, COLOR_MAP)
    assert result_red == "Red", f"Expected 'Red', but got '{result_red}'."

    # Test an ambiguous pixel
    ambiguous_pixel = np.array([100, 100, 100])
    result_ambiguous = closest_color(ambiguous_pixel, COLOR_MAP)
    assert result_ambiguous is None, f"Expected None for ambiguous pixel, but got '{result_ambiguous}'."


def test_classify_images_rgb():
    """Test classify_images_rgb with synthetic images.

    Create three images: one solid blue, one solid red, and one ambiguous.
    The ambiguous image should be labeled as Unclassified (-1).
    """
    # Create solid color images.
    blue_img = np.full((10, 10, 3), [0, 0, 255], dtype=np.uint8)
    red_img = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
    ambiguous_img = np.full((10, 10, 3), [100, 100, 100], dtype=np.uint8)
    images = [blue_img, red_img, ambiguous_img]
    labels = classify_images_rgb(images, threshold=0.7)
    expected = np.array(
        [
            clip_config.COLOR_TO_LABEL["Blue"],
            clip_config.COLOR_TO_LABEL["Red"],
            clip_config.COLOR_TO_LABEL["Unclassified"],  # Ambiguous
        ]
    )
    np.testing.assert_array_equal(labels, expected)


def test_load_color_map_files(tmp_path):
    """Test load_color_map_files by creating dummy color map files.

    Two files containing small numpy arrays are created and then concatenated.
    """
    root_data_dir = str(tmp_path)
    file1 = tmp_path / "map1.npy"
    file2 = tmp_path / "map2.npy"
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6]])
    np.save(file1, arr1)
    np.save(file2, arr2)
    color_map_files = f"{file1.name}, {file2.name}"

    # Change directory to tmp_path so that os.path.join works as expected.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    combined = load_color_map_files(color_map_files, root_data_dir)
    expected = np.concatenate([arr1, arr2], axis=0)
    assert combined.shape == expected.shape
    np.testing.assert_array_equal(combined, expected)
    monkeypatch.undo()


def test_get_equal_color_data():
    """Test get_equal_color_data to ensure balanced selection of two color classes.

    Creates dummy embeddings, ROI data, and a color mask list, and verifies that the returned
    arrays have equal numbers of samples from the specified classes.
    """
    embeddings = np.arange(20).reshape(10, 2)  # 10 samples, 2 features
    roi_data = np.arange(30).reshape(10, 3)  # 10 samples, 3 features
    # Assign first 6 samples to Blue (0) and next 4 samples to Red (1)
    color_mask_list = np.array([0] * 6 + [1] * 4)
    # Desired colors: Blue ('B') and Red ('R')
    combined_roi_data, combined_embeddings = get_equal_color_data(embeddings, roi_data, color_mask_list, ("B", "R"))
    # The minimum count between the two classes is 4, so expect 8 samples in total.
    assert combined_embeddings.shape[0] == 8
    assert combined_roi_data.shape[0] == 8

    # Verify that the selected indices correspond to the first 4 samples of each class.
    expected_indices = np.concatenate((np.arange(4), np.arange(6, 10)))
    np.testing.assert_array_equal(combined_embeddings, embeddings[expected_indices])
    np.testing.assert_array_equal(combined_roi_data, roi_data[expected_indices])


#########################################
# fMRI DATA LOADING & ROI HANDLING TESTS
#########################################


def test_get_fmri_voxels(tmp_path):
    """Test the private function _get_fmri_voxels.

    This function loads a "challenge" ROI array and a mapping dictionary from files,
    then returns a mask indicating which voxels match the ROI.
    """
    subj = "1"
    roi = "V1v"
    hemisphere = "rh"
    roi_class = "prf-visualrois"
    # Create a temp dir structure for the ROI files.
    root_data_dir = str(tmp_path)
    roi_dir = tmp_path / f"subj0{subj}" / "roi_masks"
    roi_dir.mkdir(parents=True)
    # Create the challenge file: <root_data_dir>/subj0{subj}/roi_masks/{hemisphere[0]}h.{roi_class}_challenge_space.npy
    challenge_file = roi_dir / f"{hemisphere[0]}h.{roi_class}_challenge_space.npy"
    # For testing, simulate a brain array where the value "A" marks our ROI.
    challenge_array = np.array(["A", "B", "A", "C"])
    np.save(challenge_file, challenge_array)
    # Create the ROI mapping file: mapping_{roi_class}.npy in the same directory.
    roi_map_file = roi_dir / f"mapping_{roi_class}.npy"
    # Create a mapping dictionary where key "A" corresponds to the human readable roi.
    mapping = {"A": roi}
    np.save(roi_map_file, mapping)

    mask = _get_fmri_voxels(root_data_dir, subj, roi, hemisphere, roi_class)
    # The expected mask is True where challenge_array equals "A".
    expected_mask = (challenge_array == "A").astype(int)
    np.testing.assert_array_equal(mask, expected_mask)


def test_get_concatenated_roi_data(tmp_path):
    """Test _get_concatenated_roi_data by creating dummy ROI mask files and dummy fMRI arrays.

    For a given ROI, the function should extract voxel columns from the left and right fMRI data
    (using the ROI mask) and then concatenate them.
    """
    subj = "1"
    roi = "V1v"
    root_data_dir = str(tmp_path)
    roi_class = "prf-visualrois"  # because "V1v" is in the visual ROI list

    # Create dummy ROI files for both hemispheres.
    roi_dir = tmp_path / f"subj0{subj}" / "roi_masks"
    roi_dir.mkdir(parents=True, exist_ok=True)
    # For both hemispheres, save the same challenge array.
    for hemi in ["rh", "lh"]:
        challenge_file = roi_dir / f"{hemi[0]}h.{roi_class}_challenge_space.npy"
        # Simulate an array where "A" indicates voxels in our ROI.
        challenge_array = np.array(["A", "B", "A", "C"])
        np.save(challenge_file, challenge_array)
    # Save the mapping file.
    roi_map_file = roi_dir / f"mapping_{roi_class}.npy"
    mapping = {"A": roi}  # So that "A" marks our ROI.
    np.save(roi_map_file, mapping)

    # Create dummy fMRI data for left and right hemispheres.
    # Let each fMRI array have 5 rows (images) and 4 voxels.
    lh_fmri = np.arange(5 * 4).reshape(5, 4)
    rh_fmri = np.arange(5 * 4, 5 * 8).reshape(5, 4)
    desired_image_number = 3

    # The ROI mask will select columns 0 and 2 (where the value equals "A").
    # So for each hemisphere, the extracted data will have 2 columns.
    # The function then concatenates the left and right data (resulting in 4 columns)
    # and slices the first 3 rows.
    result = _get_concatenated_roi_data(root_data_dir, subj, roi, lh_fmri, rh_fmri, desired_image_number)
    assert result.shape == (desired_image_number, 4)


def test_prepare_fmri_data(tmp_path, monkeypatch):
    """Test prepare_fmri_data for the branch where no ROI selection is applied.

    When both region_class and roi are "None", the function should simply load
    the left and right fMRI files, concatenate them, and return a scaled array.
    """
    subj = "1"
    desired_image_number = 4
    # Create dummy fMRI data (e.g., 5 rows and 10 columns each).
    lh_fmri = np.random.rand(5, 10)
    rh_fmri = np.random.rand(5, 10)
    # Build the expected directory structure.
    fmri_dir = tmp_path / f"subj0{subj}" / "training_split" / "training_fmri"
    fmri_dir.mkdir(parents=True, exist_ok=True)
    np.save(fmri_dir / "lh_training_fmri.npy", lh_fmri)
    np.save(fmri_dir / "rh_training_fmri.npy", rh_fmri)

    # Monkeypatch clip_config so that both ROI and region_class are "None".
    monkeypatch.setattr(clip_config, "ROI", "None")
    monkeypatch.setattr(clip_config, "REGION_CLASS", "None")
    monkeypatch.setattr(clip_config, "SUBJECT", subj)

    fmri_data = prepare_fmri_data(str(tmp_path), desired_image_number=4)
    # The expected data is the concatenation of lh and rh data (10 + 10 = 20 columns).
    expected_shape = (desired_image_number, 20)
    assert fmri_data.shape == expected_shape


#########################################
# REPRESENTATIONAL DISSIMILARITY MATRIX (RDM) TESTS
#########################################


def test_analyze_rdm(monkeypatch):
    """Test analyze_rdm by passing a simple RDM and dummy images.

    Monkeypatch the show_image_pair function to record its calls, and verify that
    it is called the expected number of times.
    """
    # Create a dummy 3x3 RDM.
    rdm = np.array([[0, 0.2, 0.5], [0.2, 0, 0.8], [0.5, 0.8, 0]])
    # Create three dummy images.
    dummy_img = Image.new("RGB", (10, 10))
    images = [dummy_img, dummy_img, dummy_img]

    # Collect calls to show_image_pair.
    calls = []

    def fake_show_image_pair(i, j, _imgs, title):
        calls.append((i, j, title))

    monkeypatch.setattr(data_utils, "show_image_pair", fake_show_image_pair)
    # Set the metric in the config to "correlation" so that an extra analysis branch is taken.
    monkeypatch.setattr(clip_config, "METRIC", "correlation")

    analyze_rdm(rdm, images, metric="correlation")
    # Expect three calls (for "low", "high", and "closest_to_1").
    assert len(calls) == 3


def test_create_rdm():
    """Test create_rdm by providing a dummy ROI data array.

    The function should return a square RDM (distance matrix) of the correct shape.
    """
    # Create dummy roi_data: for example, 4 samples with 3 features each.
    roi_data = np.array([[0, 1, 2], [2, 3, 4], [1, 0, 1], [5, 6, 7]])
    rdm = create_rdm(roi_data)
    # The RDM should be a 4x4 symmetric matrix.
    assert isinstance(rdm, np.ndarray)
    assert rdm.shape == (4, 4)


#########################################
# IMAGE PREPROCESSING & DATA PREP FOR CNN TESTS
#########################################


def test_preprocess_images(tmp_path):
    """Test preprocess_images by creating dummy image files.

    The function should load, resize, and normalize the images.
    """
    # Create a temporary directory and save some dummy images.
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    num_images = 3
    new_width, new_height = 8, 8
    # Create 5 dummy grayscale images.
    for i in range(5):
        img = Image.new("L", (16, 16), color=128)  # A mid-gray image.
        img.save(image_dir / f"img_{i}.jpg")

    processed = preprocess_images(str(image_dir), num_images, new_width, new_height)
    # The output should be a numpy array with shape (num_images, new_height, new_width)
    assert processed.shape == (num_images, new_height, new_width)
    # Check that the pixel values are normalized between 0 and 1.
    assert processed.min() >= 0
    assert processed.max() <= 1


def test_prepare_data_for_cnn():
    """Test prepare_data_for_cnn by providing a dummy RDM.

    The function should extract the upper triangular values and return training and testing splits.
    """
    # Create a dummy 4x4 RDM.
    rdm = np.array([[0, 0.1, 0.2, 0.3], [0.1, 0, 0.4, 0.5], [0.2, 0.4, 0, 0.6], [0.3, 0.5, 0.6, 0]])
    X_train_indices, X_test_indices, y_train, y_test = prepare_data_for_cnn(rdm, test_size=0.5)
    # For 4 images, the upper triangle (excluding the diagonal) has 6 values.
    # With test_size=0.5, we expect 3 training pairs and 3 test pairs.
    assert len(y_train) == 3
    assert len(y_test) == 3
    # The returned indices are tuples of arrays.
    assert isinstance(X_train_indices, tuple)
    assert isinstance(X_test_indices, tuple)
    assert len(X_train_indices[0]) == 3
    assert len(X_test_indices[0]) == 3


def test_data_generator():
    """Test data_generator by creating dummy image data and pair indices.

    The generator should yield batches of the correct shape, with an added channel dimension.
    """
    # Create dummy image data: 10 images of size 8x8.
    num_images = 10
    image_data = np.random.rand(num_images, 8, 8)
    # Create dummy pair indices for 5 pairs.
    row_indices = np.array([0, 1, 2, 3, 4])
    col_indices = np.array([5, 6, 7, 8, 9])
    y_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    batch_size = 2
    gen = data_generator(image_data, (row_indices, col_indices), y_data, batch_size=batch_size)
    (batch_x1, batch_x2), batch_y = next(gen)
    # Check that batch_x1 and batch_x2 have the shape (batch_size, 8, 8, 1)
    assert batch_x1.shape == (batch_size, 8, 8, 1)
    assert batch_x2.shape == (batch_size, 8, 8, 1)
    # Check that batch_y has the correct shape.
    assert batch_y.shape == (batch_size,)


def test_create_rdm_from_vectors():
    """Test create_rdm_from_vectors by providing a known vector.

    For 3 images, the vector length should be 3 and the expected RDM should be symmetric.
    """
    # For 3 images, the expected upper triangular values (excluding the diagonal) are provided in order.
    vectors = np.array([1, 2, 3])
    rdm_out = create_rdm_from_vectors(vectors)
    expected = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    np.testing.assert_array_equal(rdm_out, expected)
