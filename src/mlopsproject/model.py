import os

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models

# data paths and configuration
DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../data/food-nutrients")
MODEL_PATH = os.path.join(DATA_ROOT, "food101_model.pth")
TARGET_COLS = ["total_calories", "total_fat", "total_carb", "total_protein"]

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_model(cfg: DictConfig) -> nn.Module:
    # 1 load pretrained resnet50 with food101 weights
    # 2 replace classification head with regression head
    # 3 move to appropriate device

    # verify pretrained model exists
    assert os.path.exists(MODEL_PATH), f"model not found: {MODEL_PATH}"

    # load resnet50 with food101 weights, replace head for regression
    model = models.resnet50(weights=None, num_classes=101)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(TARGET_COLS))
    model = model.to(DEVICE)

    return model
