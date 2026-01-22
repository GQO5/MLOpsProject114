import os
import torch
import numpy as np
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from io import BytesIO
from google.cloud import storage

from PIL import Image as PILImage

app = FastAPI(title="Food Nutrients Prediction API")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# TODO: These should be loaded from the trained model metadata
Y_MEAN = np.array([253.70776, 12.700077, 19.36578, 17.732618], dtype=np.float32)
Y_STD = np.array([219.31146, 13.548113, 22.665058, 19.38393], dtype=np.float32)
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "models/model_20260118_135410_FT_True.pth"
)  # Example model (update as needed)

# Support for GCS paths
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI")  # e.g. gs://group114-bucket/models/final_model.pth
def download_from_gcs(gcs_uri: str, dst_path: str) -> None:
    """
    Provisional helper: download a model file from GCS.
    Expected gcs_uri format: gs://bucket-name/path/to/file.pth
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    no_scheme = gcs_uri[len("gs://") :]
    bucket_name, blob_path = no_scheme.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(dst_path) 



# Function to unscale predictions
def unscale(y_scaled: torch.Tensor) -> torch.Tensor:
    y_mean_t = torch.tensor(Y_MEAN, device=DEVICE, dtype=torch.float32)
    y_std_t = torch.tensor(Y_STD, device=DEVICE, dtype=torch.float32)
    return y_scaled * y_std_t + y_mean_t


# Image preprocessing
IMG_TFM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TARGET_COLS = ["total_calories", "total_fat", "total_carb", "total_protein"]


@app.on_event("startup")
def load_model_on_startup():

    model_path = MODEL_PATH
    # Download model from GCS if needed 
    # Provisional: if a GCS URI is provided, download it to /tmp and use that file.
    if MODEL_GCS_URI:
        tmp_path = "/tmp/model.pth"
        print(f"Downloading model from GCS: {MODEL_GCS_URI} -> {tmp_path}")
        download_from_gcs(MODEL_GCS_URI, tmp_path)
        model_path = tmp_path   

    # Load model weights once
    state = torch.load(model_path, map_location=DEVICE)

    model = models.resnet50(weights=None, num_classes=len(TARGET_COLS))
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(TARGET_COLS))
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    app.state.model = model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_path": MODEL_PATH,
        "model_gcs_uri": MODEL_GCS_URI,
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) read image bytes
    content = await file.read()

    # 2) open with PIL
    img = PILImage.open(BytesIO(content)).convert("RGB")

    # 3) preprocess -> tensor
    x = IMG_TFM(img).unsqueeze(0).to(DEVICE)

    # 4) predict
    model = app.state.model
    with torch.no_grad():
        y_scaled = model(x)
        y = unscale(y_scaled).detach().cpu().numpy()[0]

    # 5) format response
    return {
        "total_calories": float(y[0]),
        "total_fat": float(y[1]),
        "total_carb": float(y[2]),
        "total_protein": float(y[3]),
    }
