import datetime
import os
import typer

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


from mlopsproject.data import load_data
from mlopsproject.evaluate import evaluate
from mlopsproject.model import DEVICE, load_model
from mlopsproject.visualize import visualize

# Initialize the Typer application
app = typer.Typer()


@app.command()
def train(
    epochs_head: int = typer.Option(1, help="Number of epochs for head-only training"),
    epochs_ft: int = typer.Option(1, help="Number of epochs for full fine-tuning"),
    lr_head: float = typer.Option(
        1e-3, help="Learning rate for the head training phase"
    ),
    lr_ft: float = typer.Option(1e-4, help="Learning rate for the fine-tuning phase"),
):
    """
    Trains the model using a two-phase approach: Head-only and Fine-tuning.
    """
    print(f"🚀 Starting training with configuration:")
    print(f"   - Epochs (Head): {epochs_head}")
    print(f"   - Epochs (Fine-tune): {epochs_ft}")
    print(f"   - LR (Head): {lr_head}")
    print(f"   - LR (Fine-tune): {lr_ft}")

    # 1. Load data and pretrained model
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
    model = load_model()

    mse_loss = nn.MSELoss()

    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training history for visualization
    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "val_mae_per": [],
        "val_r2_per": [],
    }
    target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]

    # ----------------------------------------------------------------
    # PHASE A: Train regression head only (freeze backbone)
    # ----------------------------------------------------------------
    # This prevents catastrophic forgetting of pretrained features
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")

    # Use the lr_head argument
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head
    )

    # Use the epochs_head argument
    for epoch in range(1, epochs_head + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"[Head] Epoch {epoch}/{epochs_head}"):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = mse_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_mse, _, _, _, _ = evaluate(model, train_loader, y_mean, y_std)
        val_mse, val_mae_per, val_r2_per, _, _ = evaluate(
            model, val_loader, y_mean, y_std
        )

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
        print(f"[Head] epoch={epoch} trainMSE={train_mse:.4f} valMSE={val_mse:.4f}")
        print("      val MAE:", mae_str)
        print("      val R2 :", r2_str)

    # ----------------------------------------------------------------
    # PHASE B: Unfreeze all layers, fine-tune entire model
    # ----------------------------------------------------------------
    # Lower learning rate to avoid destroying pretrained features
    for p in model.parameters():
        p.requires_grad = True

    # Use the lr_ft argument
    optimizer = optim.AdamW(model.parameters(), lr=lr_ft)

    # Use the epochs_ft argument
    for e in range(1, epochs_ft + 1):
        epoch = epochs_head + e
        model.train()
        for x, y in tqdm(train_loader, desc=f"[FT] Epoch {e}/{epochs_ft}"):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = mse_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_mse, _, _, _, _ = evaluate(model, train_loader, y_mean, y_std)
        val_mse, val_mae_per, val_r2_per, _, _ = evaluate(
            model, val_loader, y_mean, y_std
        )

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
        print(f"[FT] epoch={epoch} trainMSE={train_mse:.4f} valMSE={val_mse:.4f}")
        print("     val MAE:", mae_str)
        print("     val R2 :", r2_str)

    # Save trained model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(
        os.path.dirname(__file__), "../../models", f"model_{timestamp}.pth"
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    # Generate evaluation plots and sample predictions
    print("\nRunning visualization...")
    visualize(model, history, y_mean, y_std, test_loader, test_raw)
    print("Visualization complete!")




if __name__ == "__main__":
    app()
