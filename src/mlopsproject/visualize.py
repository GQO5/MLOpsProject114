"""
model visualization module for food nutrition estimation.
"""

import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from mlopsproject.evaluate import evaluate

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
TARGET_COLS = ["total_calories", "total_fat", "total_carb", "total_protein"]

# image preprocessing for resnet
img_tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def visualize(model, history, y_mean, y_std, test_loader, test_raw):
    # creates:
    # - training curves (mse, mae, r2 over epochs)
    # - scatter plots (predicted vs true for each nutrient)
    # - sample predictions (5 random test images with overlays)

    # create output directories
    figures_dir = os.path.join(os.path.dirname(__file__), "../../reports/figures")
    random_dir = os.path.join(figures_dir, "Random")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(random_dir, exist_ok=True)

    # generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # evaluate on test set
    test_mse, test_mae_per, test_r2_per, y_true_test, y_pred_test = evaluate(
        model, test_loader, y_mean, y_std
    )
    print("\nTEST metrics:")
    print(f"  MSE (standardized): {test_mse:.4f}")
    for name, mae, r2 in zip(TARGET_COLS, test_mae_per, test_r2_per):
        print(f"  {name:>14} | MAE={mae:8.2f} | R2={r2:6.3f}")

    epochs = np.array(history["epoch"])
    train_mse = np.array(history["train_mse"])
    val_mse = np.array(history["val_mse"])
    val_mae_per = np.stack(history["val_mae_per"], axis=0)  # (epochs, 4)
    val_r2_per = np.stack(history["val_r2_per"], axis=0)  # (epochs, 4)

    # plot mse curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mse, label="train MSE")
    plt.plot(epochs, val_mse, label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (standardized)")
    plt.title("Training vs Validation MSE")
    plt.legend()
    plt.grid(True)
    mse_filename = f"mse_curves_{timestamp}.png"
    plt.savefig(os.path.join(figures_dir, mse_filename))
    plt.close()
    print(f"MSE curves saved to reports/figures/{mse_filename}")

    # plot per-target mae curves
    plt.figure(figsize=(10, 6))
    for j, name in enumerate(TARGET_COLS):
        plt.plot(epochs, val_mae_per[:, j], label=f"val MAE {name}")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (original units)")
    plt.title("Validation MAE per target")
    plt.legend()
    plt.grid(True)
    mae_filename = f"mae_curves_{timestamp}.png"
    plt.savefig(os.path.join(figures_dir, mae_filename))
    plt.close()
    print(f"MAE curves saved to reports/figures/{mae_filename}")

    # plot per-target r2 curves
    plt.figure(figsize=(10, 6))
    for j, name in enumerate(TARGET_COLS):
        plt.plot(epochs, val_r2_per[:, j], label=f"val R2 {name}")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.title("Validation R2 per target")
    plt.legend()
    plt.grid(True)
    r2_filename = f"r2_curves_{timestamp}.png"
    plt.savefig(os.path.join(figures_dir, r2_filename))
    plt.close()
    print(f"R2 curves saved to reports/figures/{r2_filename}")

    # create scatter plots for each target
    for j, name in enumerate(TARGET_COLS):
        yt = y_true_test[:, j]
        yp = y_pred_test[:, j]
        plt.figure(figsize=(8, 6))
        plt.scatter(yt, yp, s=10, alpha=0.6)
        mn = min(yt.min(), yp.min())
        mx = max(yt.max(), yp.max())
        plt.plot([mn, mx], [mn, mx], linewidth=1)
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Pred {name}")
        plt.title(f"Test predictions: {name}")
        plt.grid(True)
        scatter_filename = f"scatter_{name}_{timestamp}.png"
        plt.savefig(os.path.join(figures_dir, scatter_filename))
        plt.close()
        print(f"Scatter plot for {name} saved to reports/figures/{scatter_filename}")

    # generate random sample visualizations
    model.eval()

    def predict_one(pil_img):
        x = img_tfm(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            use_amp = DEVICE == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_scaled = model(x).squeeze(0).float().detach().cpu().numpy()
        pred = pred_scaled * y_std + y_mean  # unscale
        return pred

    idxs = random.sample(range(len(test_raw)), 5)
    for i, idx in enumerate(idxs):
        row = test_raw[idx]  # raw has 'image' + target cols
        img = row["image"]
        gt = np.array([float(row[c]) for c in TARGET_COLS], dtype=np.float32)
        pred = predict_one(img)

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")

        # create title with predictions
        title_lines = [f"Timestamp: {timestamp}"]
        for k, name in enumerate(TARGET_COLS):
            title_lines.append(f"{name}: GT={gt[k]:.1f} | Pred={pred[k]:.1f}")
        title = "\n".join(title_lines)

        # add text overlay
        plt.text(
            0.5,
            0.02,
            title,
            ha="center",
            va="bottom",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        random_filename = f"random_sample_{i + 1}_{timestamp}.png"
        plt.savefig(os.path.join(random_dir, random_filename), bbox_inches="tight")
        plt.close()
        print(
            f"Random sample {i + 1} saved to reports/figures/Random/{random_filename}"
        )


if __name__ == "__main__":
    # this would need to be called with the trained model and data
    pass
