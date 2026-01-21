import os
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Adjust import based on your project structure
from src.mlopsproject.model import load_model, MODEL_PATH


@patch("os.path.exists")  # 1. Mock file existence check
@patch("torch.load")  # 2. Mock loading the .pth file
@patch("torch.nn.Module.load_state_dict")  # 3. Mock applying weights
def test_model_architecture(mock_load_state_dict, mock_torch_load, mock_path_exists):
    """
    Test architecture without real weights file.
    """
    # A. Setup Mocks
    mock_path_exists.return_value = True  # Pretend file exists
    mock_torch_load.return_value = {}  # Return empty dict for weights
    # mock_load_state_dict will just do nothing, preventing errors from the empty dict

    # B. Run Function
    # We pass None for cfg since your code doesn't strictly use it in the snippet
    model = load_model(cfg=None)

    # C. Verify Architecture
    # The mocks allowed us to reach this point with a randomly initialized model
    # Now we check if the head was correctly replaced
    assert isinstance(model, nn.Module)
    assert model.fc.out_features == 4, "Output features should be 4 (calories, fat, carb, protein)"

    # D. Verify Mocks were used
    # Ensure it actually tried to check for the file at the specific path
    mock_path_exists.assert_called_with(MODEL_PATH)


@patch("os.path.exists")
@patch("torch.load")
@patch("torch.nn.Module.load_state_dict")
def test_model_forward_pass(mock_load_state_dict, mock_torch_load, mock_path_exists):
    """
    Test forward pass with dummy inputs.
    """
    # Setup Mocks
    mock_path_exists.return_value = True
    mock_torch_load.return_value = {}

    # Run Function
    model = load_model(cfg=None)

    # 1. Create a dummy input (Batch=1, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 2. Move input to the same device as the model
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    # 3. Run the forward pass
    output = model(dummy_input)

    # 4. Check the output shape
    # Expected: [Batch_Size=1, Outputs=4]
    assert output.shape == torch.Size([1, 4])
