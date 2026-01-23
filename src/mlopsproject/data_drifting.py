import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig
import glob
import os

# Import your project modules
from mlopsproject.data import load_data
from mlopsproject.model import load_model, DEVICE

def check_robustness(model_path: str, noise_lvl: float = 0.5):
    """
    Checks if the regression model's MSE worsens when data quality degrades.
    """
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    # load_data() returns 8 values. We strictly need the test_loader (index 2).
    # We ignore the rest (_)
    _, _, test_loader, _, _, _, _, _ = load_data()

    # 2. Initialize Model Architecture
    print(f"Initializing model architecture...")

    # Your load_model function expects a DictConfig, so we pass an empty one.
    dummy_cfg = DictConfig({})
    model = load_model(dummy_cfg)

    # 3. Load Trained Weights
    print(f"Loading trained weights from {model_path}...")
    try:
        # Load the weights you saved during training
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Ensure you are pointing to a model trained with this architecture.")
        return

    model.to(DEVICE)
    model.eval()

    # 4. Metric: Mean Squared Error (Standard for Regression)
    criterion = nn.MSELoss()

    # 5. Run Experiment
    print("\n--- Starting Robustness Check ---")

    # A. Baseline (Clean Data)
    baseline_error = evaluate_loss(model, test_loader, criterion, noise=0.0)
    print(f"Baseline MSE (Clean Data): {baseline_error:.4f}")

    # B. Drifted (Noisy Data)
    drifted_error = evaluate_loss(model, test_loader, criterion, noise=noise_lvl)
    print(f"Drifted MSE (Noisy Data):  {drifted_error:.4f}")

    # 6. Save Report
    output_path = Path("reports/data_drift_robustness.txt")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    degradation = drifted_error - baseline_error

    with open(output_path, "w") as f:
        f.write("MLOps Data Drift Robustness Report\n")
        f.write("==================================\n")
        f.write(f"Model Path:   {model_path}\n")
        f.write(f"Noise Level:  {noise_lvl} (Gaussian std)\n\n")
        f.write(f"Baseline MSE: {baseline_error:.4f}\n")
        f.write(f"Drifted MSE:  {drifted_error:.4f}\n")
        f.write(f"Degradation:  {degradation:.4f}\n")

        if degradation > 0:
            f.write("\nRESULT: PASSED. Model performance degrades with drift, verifying sensitivity.")
        else:
            f.write("\nRESULT: UNCERTAIN. Model performance did not drop. Noise might be too low.")

    print(f"\nReport saved to {output_path}")

def evaluate_loss(model, loader, criterion, noise=0.0):
    """Calculate average MSE over the loader with optional noise injection."""
    total_loss = 0
    batches = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # --- DRIFT INJECTION ---
            if noise > 0:
                # Add noise to normalized images
                corruption = torch.randn_like(images) * noise
                images = images + corruption
            # -----------------------

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            batches += 1

    return total_loss / batches

if __name__ == "__main__":
    import argparse
    import os

    # 1. Parse arguments passed from tasks.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the specific model file")
    args = parser.parse_args()

    # 2. Logic: Use provided path OR auto-detect
    if args.model_path:
        # If the user provided a path, use it explicitly
        print(f"Using provided model path: {args.model_path}")
        check_robustness(args.model_path)
    else:
        # Auto-detect the latest model if no path provided
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models"

        # Check if directory exists
        if not models_dir.exists():
            print(f"Directory not found: {models_dir}")
            print("Running in CI? Make sure you pull the model via DVC or train it first.")
            exit(1)

        list_of_files = list(models_dir.glob('*.pth'))

        if list_of_files:
            latest_model = max(list_of_files, key=os.path.getctime)
            print(f"Automatically found latest model: {latest_model}")
            check_robustness(str(latest_model))
        else:
            print(f"No .pth models found in {models_dir}.")
            print("Please run 'uv run invoke train' first or pull your models with DVC.")
