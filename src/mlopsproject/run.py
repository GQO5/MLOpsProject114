import hydra
import torch
from omegaconf import OmegaConf
from mlopsproject.model import load_model
from mlopsproject.data import load_data


@hydra.main(config_path="../../configs", config_name="run.yaml", version_base=None)
def main(cfg):
    # Load configuration
    print("Configuration Loaded:")
    print(OmegaConf.to_yaml(cfg))

    # Set device
    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    print(f"Using device: {device}")

    # For now, just load the data inside of the trainer
    # Instantiate Logger, Dataset, Model, and Trainer
    model = load_model(cfg)
    print("Model Loaded:")
    print(model)
    trainer = hydra.utils.instantiate(cfg.trainer.init, model=model, device=device)
    model_trained, history, y_mean, y_std, test_loader, test_raw = trainer.train(**cfg.trainer.train)

if __name__ == "__main__":
    main()