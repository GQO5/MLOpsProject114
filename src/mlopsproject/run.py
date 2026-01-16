import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import wandb
from mlopsproject.data import load_data
from mlopsproject.model import load_model


def seed_everything(seed):
    """Set random seed for reproducibility."""
    print(f"Setting random seed to: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="../../configs", config_name="run.yaml", version_base=None)
def main(cfg):
    # Load configuration
    print("Configuration Loaded:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize Weights & Biases
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="MLOpsProject114",
        config=cfg_dict,
    )

    # Set random seed for reproducibility if specified
    if cfg.seed_run:
        seed_everything(cfg.seed)

    # Set device
    if cfg.device in ["unset", "auto"]:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(cfg.device)

    print(f"Using device: {device}")

    # For now, just load the data inside of the trainer
    # Instantiate Logger, Dataset, Model, and Trainer
    model = load_model(cfg)
    print("Model Loaded")

    trainer = hydra.utils.instantiate(cfg.trainer.init, model=model, device=device)
    model_trained, history, y_mean, y_std, test_loader, test_raw = trainer.train(
        **cfg.trainer.train
    )

    wandb.finish()


if __name__ == "__main__":
    main()
