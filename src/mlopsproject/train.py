import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


from mlopsproject.data import load_data
from mlopsproject.evaluate import evaluate
from mlopsproject.model import DEVICE, load_model
from mlopsproject.visualize import visualize


def set_finetune(model, finetune):
    """Configure model parameters for fine-tuning or head-only training."""
    if finetune:
        print("Fine-tuning all model parameters.")
        for param in model.parameters():
            param.requires_grad = finetune
    else:
        print("Freezing backbone parameters; training head only.")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")


class Food101ResNet50:
    def __init__(self, criterion, optimizer, scheduler, device, model, finetune):
        self.model = model
        self.criterion = criterion
        self.finetune = finetune
        set_finetune(self.model, self.finetune)
        self.optimizer = optimizer(params=self.model.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)
        self.device = device

    def train(self, total_epochs=250, validation_interval=10):
        # 1 load data and pretrained model
        # 2 two-phase training: head-only then full fine-tuning
        # 3 evaluate and save model
        # 4 generate visualizations

        # load data and model
        (
            train_loader,
            val_loader,
            test_loader,
            train_raw,
            val_raw,
            test_raw,
            y_mean,
            y_std,
        ) = load_data()

        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # training history for visualization
        history = {
            "epoch": [],
            "train_mse": [],
            "val_mse": [],
            "val_mae_per": [],
            "val_r2_per": [],
        }

        target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]

        for epoch in range(1, total_epochs + 1):
            self.model.train()
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}"):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            self.scheduler.step()
            train_mse, _, _, _, _ = evaluate(self.model, train_loader, y_mean, y_std)
            val_mse, val_mae_per, val_r2_per, _, _ = evaluate(
                self.model, val_loader, y_mean, y_std
            )

            # log metrics to wandb

            phase = "ft" if self.finetune else "head"

            metrics = {
                "epoch": epoch,
                "phase": phase,
                "train/mse": float(train_mse),
                "val/mse": float(val_mse),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }

            for name, mae in zip(target_cols, val_mae_per):
                metrics[f"val/mae_{name}"] = float(mae)

            for name, r2 in zip(target_cols, val_r2_per):
                metrics[f"val/r2_{name}"] = float(r2)

            wandb.log(metrics, step=epoch)

            history["epoch"].append(epoch)
            history["train_mse"].append(train_mse)
            history["val_mse"].append(val_mse)
            history["val_mae_per"].append(val_mae_per)
            history["val_r2_per"].append(val_r2_per)

            mae_str = " | ".join(
                [
                    f"{n.split('total_')[-1]}:{v:.1f}"
                    for n, v in zip(target_cols, val_mae_per)
                ]
            )
            r2_str = " | ".join(
                [
                    f"{n.split('total_')[-1]}:{v:.3f}"
                    for n, v in zip(target_cols, val_r2_per)
                ]
            )
            print(f"epoch={epoch} trainMSE={train_mse:.4f} valMSE={val_mse:.4f}")
            print("      val MAE:", mae_str)
            print("      val R2 :", r2_str)

        # save trained model with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(
            os.path.dirname(__file__),
            "../../models",
            f"model_{timestamp}_FT_{self.finetune}.pth",
        )
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # log model artifact to W&B
        model_artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description="Trained Food101 ResNet50 model",
        )

        model_artifact.add_file(model_save_path)
        wandb.log_artifact(model_artifact)

        print("Model artifact logged to W&B")

        # generate evaluation plots and sample predictions
        print("\nRunning visualization...")
        visualize(self.model, history, y_mean, y_std, test_loader, test_raw)
        print("Visualization complete!")

        # log figures artifact to W&B
        figures_dir = "reports/figures"

        figures_artifact = wandb.Artifact(
            name=f"figures-{wandb.run.id}",
            type="figures",
            description="Training curves, scatter plots, and sample predictions",
        )

        figures_artifact.add_dir(figures_dir)
        wandb.log_artifact(figures_artifact)
        print("Figures artifact logged to W&B")

        return self.model, history, y_mean, y_std, test_loader, test_raw


if __name__ == "__main__":
    trainer = Food101ResNet50()
    trainer.train()
