import numpy as np
import pytest
import torch
from PIL import Image
from src.project.config import clip_config
from src.project.utils import clip_utils
from src.project.utils.clip_utils import get_image_embeddings, load_clip_model, load_images
from transformers import CLIPModel, CLIPProcessor


def dummy_clip_model():
    """Return dummy CLIP model and processor objects for testing."""

    class DummyModel:
        def to(self, _device):
            return self

        def get_image_features(self, **kwargs):  # noqa: ANN003
            # Assume the dummy inputs contain a key "dummy"
            dummy_tensor = kwargs["dummy"]
            batch_size = dummy_tensor.shape[0]
            return torch.rand(batch_size, 512)

    class DummyBatchEncoding(dict):
        def __init__(self, dummy_tensor):
            # Initialize the dict with key "dummy"
            super().__init__({"dummy": dummy_tensor})

        def to(self, _device):
            # Mock moving to device by returning self.
            return self

    class DummyProcessor:
        def __call__(self, images, return_tensors, padding):  # noqa: ARG002
            # Create a dummy tensor with shape (len(images), 10)
            dummy_tensor = torch.ones(len(images), 10)
            # Return an instance of DummyBatchEncoding (a dict subclass with .to())
            return DummyBatchEncoding(dummy_tensor)

    return DummyModel(), DummyProcessor()


def test_load_clip_model(monkeypatch):
    """Test that load_clip_model returns a model with get_image_features and a callable processor."""
    monkeypatch.setattr(
        CLIPModel,
        "from_pretrained",
        lambda _name: dummy_clip_model()[0],
    )
    monkeypatch.setattr(
        CLIPProcessor,
        "from_pretrained",
        lambda _name: dummy_clip_model()[1],
    )

    model, processor = load_clip_model("dummy-model", device="cpu")
    assert hasattr(model, "get_image_features")
    assert callable(processor)


def test_get_image_embeddings_from_file(tmp_path, monkeypatch):
    """Test get_image_embeddings when using an embeddings file (THINGSvision branch)."""
    # Create a dummy embeddings file with shape (5, 512)
    dummy_data = np.random.rand(5, 512).astype(np.float32)
    file_path = tmp_path / "embeddings.npy"
    np.save(file_path, dummy_data)

    monkeypatch.setattr(clip_config, "LOAD_EMBEDDINGS_FILE", str(file_path))

    # The images list is ignored in file-based mode.
    embeddings = get_image_embeddings([], desired_image_number=3, device="cpu", is_thingsvision=True)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 512


def test_get_image_embeddings_scratch(monkeypatch):
    """Test get_image_embeddings in scratch mode (no embeddings file)."""
    monkeypatch.setattr(clip_config, "LOAD_EMBEDDINGS_FILE", "")
    monkeypatch.setattr(clip_config, "PRETRAINED_MODEL", "dummy-model")

    # Create two dummy images.
    dummy_img = Image.new("RGB", (10, 10))
    images = [dummy_img, dummy_img]

    # Patch load_clip_model to return our dummy model/processor.
    monkeypatch.setattr(
        clip_utils,
        "load_clip_model",
        lambda *_args, **_kwargs: dummy_clip_model(),
    )

    embeddings = get_image_embeddings(images, desired_image_number=2, device="cpu", is_thingsvision=False)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 512


def test_load_images_success(tmp_path, monkeypatch):
    """Test load_images successfully loads images from the expected directory structure."""
    monkeypatch.setattr(clip_config, "ROOT_DATA_DIR", "data")
    monkeypatch.setattr(clip_config, "SUBJECT", "1")

    # Create the expected directory structure and add 3 dummy image files.
    image_dir = tmp_path / "data" / "subj01" / "training_split" / "training_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        img = Image.new("RGB", (10, 10))
        img.save(image_dir / f"img_{i}.jpg")

    images = load_images(tmp_path, desired_image_number=2)
    assert len(images) == 2
    for im in images:
        assert hasattr(im, "convert")


def test_load_images_missing_directory(tmp_path, monkeypatch):
    """Test load_images exits when the expected image directory is missing."""
    monkeypatch.setattr(clip_config, "ROOT_DATA_DIR", "nonexistent")
    monkeypatch.setattr(clip_config, "SUBJECT", "1")
    monkeypatch.setattr(clip_config, "DESIRED_IMAGE_NUMBER", 2)
    # Ensure we exit if missing directory
    with pytest.raises(SystemExit):
        load_images(tmp_path)
