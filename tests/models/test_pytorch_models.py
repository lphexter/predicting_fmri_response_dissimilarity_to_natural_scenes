import torch

from project.models import pytorch_models

###################
# NOTE: Only testing properties of DynamicLayerSizeNeuralNetwork as it's the final model Class we use in our experiments
###################


def test_forward_shape_linear():
    """Test that DynamicLayerSizeNeuralNetwork returns the expected output shape when using the "linear" activation (i.e. no additional bounding).

    For an input tensor of shape (batch_size, 1024), the output should be of shape (batch_size, 1).
    """
    hidden_layers = 2
    model = pytorch_models.DynamicLayerSizeNeuralNetwork(hidden_layers=hidden_layers, activation_func="linear")
    batch_size = 8
    input_tensor = torch.randn(batch_size, 1024)
    output = model(input_tensor)
    assert output.shape == (batch_size, 1)


def test_forward_shape_sigmoid():
    """Test that DynamicLayerSizeNeuralNetwork returns outputs in the expected range when using the "sigmoid" activation.

    Since the forward method applies torch.sigmoid(x) * 2, the outputs should be in [0, 2].
    """
    hidden_layers = 2
    model = pytorch_models.DynamicLayerSizeNeuralNetwork(hidden_layers=hidden_layers, activation_func="sigmoid")
    batch_size = 8
    input_tensor = torch.randn(batch_size, 1024)
    output = model(input_tensor)
    # Check that all outputs are between 0 and 2 (inclusive)
    assert output.min().item() >= 0
    assert output.max().item() <= 2


def test_layer_count():
    """Test that the model creates the expected number of layers.

    The architecture is built as follows:
      - Always an initial layer: 1024 -> 512.
      - Then, for each hidden layer, one linear layer (512 -> next_size) where next_size is half of current size.
      - Finally, a last layer from the last hidden layer's size to 1.

    Therefore, the expected number of layers is (hidden_layers + 2).
    """
    for hidden_layers in [0, 1, 2, 3]:
        model = pytorch_models.DynamicLayerSizeNeuralNetwork(hidden_layers=hidden_layers, activation_func="linear")
        expected_layers = hidden_layers + 2
        assert len(model.layers) == expected_layers, f"Expected {expected_layers} layers, got {len(model.layers)}"


def test_forward_values_linear():
    """Test that the network with linear activation can produce outputs outside of the [0, 2] range.

    Since the "linear" activation does not bound the output, we expect that if we force the final
    layer to produce a large negative or positive shift, the output will fall outside [0,2].
    """
    hidden_layers = 1
    model = pytorch_models.DynamicLayerSizeNeuralNetwork(hidden_layers=hidden_layers, activation_func="linear")

    # Manually adjust the final layer parameters to force the output outside [0, 2]
    final_layer = model.layers[-1]
    torch.nn.init.constant_(final_layer.weight, 1.0)
    torch.nn.init.constant_(final_layer.bias, -10.0)

    input_tensor = torch.randn(4, 1024)
    output = model(input_tensor)

    # With a bias of -10, we expect that many outputs will be less than 0.
    assert output.min().item() < 0 or output.max().item() > 2
