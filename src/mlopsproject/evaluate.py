import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unscale(y_scaled, y_mean, y_std):
    # convert scaled predictions back to original units
    y_mean_t = torch.tensor(y_mean, device=DEVICE, dtype=torch.float32)
    y_std_t = torch.tensor(y_std, device=DEVICE, dtype=torch.float32)
    return y_scaled * y_std_t + y_mean_t


@torch.no_grad()
def evaluate(model, loader, y_mean, y_std):
    # computes:
    # - mean squared error (mse)
    # - mean absolute error per target (mae_per)
    # - r-squared score per target (r2_per)

    # evaluate model on dataset, compute metrics
    model.eval()
    mse_loss = nn.MSELoss()
    mse_sum, n = 0.0, 0
    all_true, all_pred = [], []

    for x, y_scaled in loader:
        x = x.to(DEVICE, non_blocking=True)
        y_scaled = y_scaled.to(DEVICE, non_blocking=True)

        pred_scaled = model(x)
        loss = mse_loss(pred_scaled, y_scaled).item()

        y_true = unscale(y_scaled, y_mean, y_std)
        y_pred = unscale(pred_scaled, y_mean, y_std)

        bs = x.size(0)
        mse_sum += loss * bs
        n += bs

        all_true.append(y_true.detach().cpu().numpy())
        all_pred.append(y_pred.detach().cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    # compute per-target mae
    mae_per = np.mean(np.abs(all_pred - all_true), axis=0)

    # compute per-target r2 score
    r2_per = []
    for j in range(all_true.shape[1]):
        yt = all_true[:, j]
        yp = all_pred[:, j]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2) + 1e-12
        r2_per.append(1.0 - ss_res / ss_tot)
    r2_per = np.array(r2_per, dtype=np.float32)

    return (mse_sum / n), mae_per, r2_per, all_true, all_pred
