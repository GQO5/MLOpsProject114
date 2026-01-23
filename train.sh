#!/bin/bash

# Initialize git repo for DVC
git init

# Pull data from DVC
uv run dvc pull

# Set W&B API key
export WANDB_API_KEY=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

# Login to wandb
uv run wandb login $WANDB_API_KEY

# Preprocess data
uv run invoke preprocess-data

# Train the model
uv run invoke train