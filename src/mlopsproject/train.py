import datetime
import os

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


def train():
    # 1 load data and pretrained model
    # 2 two-phase training: head-only then full fine-tuning
    # 3 evaluate and save model
    # 4 generate visualizations

    # load data and model
    train_loader, val_loader, test_loader, train_raw, val_raw, test_raw, y_mean, y_std = load_data()
    model = load_model()

    mse_loss = nn.MSELoss()

    # training hyperparameters
    epochs_head = 1
    epochs_ft = 1
    lr_head = 1e-3
    lr_ft = 1e-4

    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # training history for visualization
    history = {"epoch": [], "train_mse": [], "val_mse": [], "val_mae_per": [], "val_r2_per": []}

    target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]

    # phase a: train regression head only (freeze backbone)
    # this prevents catastrophic forgetting of pretrained features
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)

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
        val_mse, val_mae_per, val_r2_per, _, _ = evaluate(model, val_loader, y_mean, y_std)

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_mae_per"].append(val_mae_per)
        history["val_r2_per"].append(val_r2_per)

        mae_str = " | ".join([f"{n.split('total_')[-1]}:{v:.1f}" for n, v in zip(target_cols, val_mae_per)])
        r2_str = " | ".join([f"{n.split('total_')[-1]}:{v:.3f}" for n, v in zip(target_cols, val_r2_per)])
        print(f"[Head] epoch={epoch} trainMSE={train_mse:.4f} valMSE={val_mse:.4f}")
        print("      val MAE:", mae_str)
        print("      val R2 :", r2_str)

    # phase b: unfreeze all layers, fine-tune entire model
    # lower learning rate to avoid destroying pretrained features
    for p in model.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=lr_ft)

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
        val_mse, val_mae_per, val_r2_per, _, _ = evaluate(model, val_loader, y_mean, y_std)

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_mae_per"].append(val_mae_per)
        history["val_r2_per"].append(val_r2_per)

        mae_str = " | ".join([f"{n.split('total_')[-1]}:{v:.1f}" for n, v in zip(target_cols, val_mae_per)])
        r2_str = " | ".join([f"{n.split('total_')[-1]}:{v:.3f}" for n, v in zip(target_cols, val_r2_per)])
        print(f"[FT] epoch={epoch} trainMSE={train_mse:.4f} valMSE={val_mse:.4f}")
        print("     val MAE:", mae_str)
        print("     val R2 :", r2_str)

    # save trained model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(os.path.dirname(__file__), "../../models", f"model_{timestamp}.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # generate evaluation plots and sample predictions
    print("\nRunning visualization...")
    visualize(model, history, y_mean, y_std, test_loader, test_raw)
    print("Visualization complete!")

    return model, history, y_mean, y_std, test_loader, test_raw


if __name__ == "__main__":
    train()
