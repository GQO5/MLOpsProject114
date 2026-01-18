import os

import pytest
import torch

from src.mlopsproject.model import MODEL_PATH, load_model


# Check if the model file exists before running these tests
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_architecture():
    """Test if the model is constructed with the correct output head."""
    model = load_model()

    # Verify the last layer (fc) matches our 4 targets
    # (total_calories, total_fat, total_carb, total_protein)
    assert model.fc.out_features == 4


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_forward_pass():
    """Test if the model accepts a standard image tensor and returns the right shape."""
    model = load_model()

    # 1. Create a dummy input (Batch=1, Channels=3, Height=224, Width=224)
    # This matches the standard ResNet/ImageNet input size
    dummy_input = torch.randn(1, 3, 224, 224)

    # 2. Move input to the same device as the model (CPU or CUDA)
    # Since your load_model() automatically moves the model, we must match it
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    # 3. Run the forward pass
    output = model(dummy_input)

    # 4. Check the output shape
    # We expect [Batch_Size=1, Outputs=4]
    assert output.shape == torch.Size([1, 4])
