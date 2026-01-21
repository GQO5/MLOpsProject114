#!/bin/bash

# Initialize git repo for DVC
git init

# Pull data from DVC
uv run dvc pull

# Login to wandb if API key is provided
if [ -n "$WANDB_API_KEY" ]; then
    uv run wandb login $WANDB_API_KEY
fi

# Preprocess data
uv run invoke preprocess-data

# Train the model
uv run invoke train