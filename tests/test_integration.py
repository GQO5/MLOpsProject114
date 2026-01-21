from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

# 1. IMPORTANT: We import the CLASS, not a 'train' function
from src.mlopsproject.train import Food101ResNet50


def test_import_train():
    # Verify that the class exists
    from src.mlopsproject.train import Food101ResNet50

    assert Food101ResNet50 is not None


@patch("src.mlopsproject.train.wandb")  # Mock wandb so it doesn't try to connect to the internet
@patch("src.mlopsproject.train.visualize")
@patch("src.mlopsproject.train.evaluate")
@patch("src.mlopsproject.train.torch.save")
@patch("src.mlopsproject.train.load_data")
def test_train_execution(mock_load_data, mock_save, mock_evaluate, mock_visualize, mock_wandb):
    """
    Integration Test: Instantiates the Food101ResNet50 class and runs train().
    """

    # ---------------------------------------------------------
    # 1. Setup Mock Data (Same as before)
    # ---------------------------------------------------------
    dummy_x = torch.randn(2, 3, 224, 224)
    dummy_y = torch.randn(2, 4)

    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = iter([(dummy_x, dummy_y)])
    mock_loader.__len__.return_value = 1

    mock_load_data.return_value = (
        mock_loader,  # train
        mock_loader,  # val
        mock_loader,  # test
        [],
        [],
        [],  # raw data
        torch.tensor(0.0),
        torch.tensor(1.0),
    )

    # ---------------------------------------------------------
    # 2. Setup Mock Model
    # ---------------------------------------------------------
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 4)  # Necessary for the freezing logic

        def forward(self, x):
            return torch.randn(x.shape[0], 4, requires_grad=True)

    real_model = TinyModel()

    # ---------------------------------------------------------
    # 3. Setup Mock Evaluate
    # ---------------------------------------------------------
    mock_evaluate.return_value = (
        0.5,  # val_mse
        [0.1] * 4,  # val_mae_per
        [0.1] * 4,  # val_r2_per
        torch.tensor([0.1] * 4),
        torch.tensor([0.1] * 4),
    )

    # ---------------------------------------------------------
    # 4. PREPARE CLASS DEPENDENCIES (New)
    # ---------------------------------------------------------

    # Mock Criterion (Loss function)
    mock_criterion = MagicMock()
    mock_criterion.return_value = torch.tensor(0.5, requires_grad=True)

    # Mock Optimizer CLASS
    # Your code does: self.optimizer = optimizer(params=...)
    # So we pass a Mock that acts as the optimizer CLASS
    mock_optimizer_cls = MagicMock()
    mock_optimizer_instance = MagicMock()
    # Necessary for the log: self.optimizer.param_groups[0]["lr"]
    mock_optimizer_instance.param_groups = [{"lr": 0.001}]
    mock_optimizer_cls.return_value = mock_optimizer_instance

    # Mock Scheduler CLASS
    mock_scheduler_cls = MagicMock()
    mock_scheduler_instance = MagicMock()
    mock_scheduler_cls.return_value = mock_scheduler_instance

    # ---------------------------------------------------------
    # 5. Execute the Test
    # ---------------------------------------------------------
    try:
        # Instantiate the class with all necessary mocks
        trainer = Food101ResNet50(
            criterion=mock_criterion,
            optimizer=mock_optimizer_cls,
            scheduler=mock_scheduler_cls,
            device="cpu",  # Use CPU for the test
            model=real_model,
            finetune=False,
        )

        # Execute the train method
        trainer.train(total_epochs=1)  # Only 1 epoch to keep it fast

    except Exception as e:
        pytest.fail(f"Train script crashed with error: {e}")

    # Verifications
    assert mock_load_data.called
    assert mock_save.called
    assert mock_wandb.log.called  # Verify that we tried to log to wandb
