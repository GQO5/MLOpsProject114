import pytest
import torch
import torch.nn as nn
import os
from src.mlopsproject.model import load_model, MODEL_PATH

# We skip the test if the weights file is missing (e.g., in GitHub Actions)
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model weights not found")
def test_training_update():
    """
    Test related to Model Training (M16).
    
    Verifies that one step of the training loop actually modifies the model parameters.
    If this passes, your training logic (forward -> backward -> optimizer step) is valid.
    """
    # 1. Load the model and set to train mode
    model = load_model()
    model.train()
    device = next(model.parameters()).device

    # 2. Create Dummy Data (Batch Size 2 to satisfy BatchNorm requirements)
    # Your model expects [Batch, 3, 224, 224] and outputs [Batch, 4]
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    dummy_target = torch.randn(2, 4).to(device)

    # 3. Setup a simple Optimizer (just like in your train.py)
    # We test the 'fc' layer because that is what you train in Phase A
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 4. CRITICAL: Save the initial weights to compare later
    initial_weights = model.fc.weight.clone()

    # 5. Run the Training Step (Forward -> Loss -> Backward -> Step)
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()

    # 6. Assertions
    # Check A: Loss is not NaN (common bug check)
    assert not torch.isnan(loss), "Training step produced NaN loss"
    
    # Check B: Did the model actually learn?
    # The weights AFTER the step must be different from the weights BEFORE.
    assert not torch.equal(initial_weights, model.fc.weight), "Model weights did not update! Gradients are broken."