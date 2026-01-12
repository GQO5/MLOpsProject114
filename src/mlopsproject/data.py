import os

import numpy as np
import torch
from datasets import Image, load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# data paths and configuration
DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../data/food-nutrients")
METADATA = os.path.join(DATA_ROOT, "metadata.jsonl")
MODEL_PATH = os.path.join(DATA_ROOT, "food101_model.pth")

# target nutritional columns to predict
TARGET_COLS = ["total_calories", "total_fat", "total_carb", "total_protein"]


def load_data():
    # 1 load raw dataset from jsonl
    # 2 split into train/val/test sets
    # 3 normalization
    # 4 setup image transformations
    # 5 create dataloaders

    # verify data files exist
    assert os.path.exists(METADATA), f"metadata.jsonl not found: {METADATA}"

    # load dataset from jsonl file
    ds = load_dataset("json", data_files=METADATA, split="train")
    print(ds)
    print("Columns:", ds.column_names)

    # split into train/val/test sets
    ds_tmp = ds.train_test_split(test_size=0.10, seed=42)
    train_val = ds_tmp["train"].train_test_split(test_size=0.10, seed=42)
    train_ds = train_val["train"]
    val_ds = train_val["test"]
    test_ds = ds_tmp["test"]
    print("Sizes:", len(train_ds), len(val_ds), len(test_ds))

    # check for missing target columns
    missing = [c for c in TARGET_COLS if c not in train_ds.column_names]
    if missing:
        raise KeyError(f"Missing targets: {missing}\nAvailable: {train_ds.column_names}")

    # compute normalization stats from training data
    y = np.stack([np.array(train_ds[c], dtype=np.float32) for c in TARGET_COLS], axis=1)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-6
    print("Target mean:", dict(zip(TARGET_COLS, y_mean)))
    print("Target std: ", dict(zip(TARGET_COLS, y_std)))

    # add absolute image paths
    def add_image_path(ex):
        ex["image"] = os.path.join(DATA_ROOT, ex["file_name"])
        return ex

    train_ds = train_ds.map(add_image_path)
    val_ds = val_ds.map(add_image_path)
    test_ds = test_ds.map(add_image_path)

    print("First train image path:", train_ds[0]["image"])
    print("Exists?", os.path.exists(train_ds[0]["image"]))

    # cast image column to pil image type
    train_ds = train_ds.cast_column("image", Image())
    val_ds = val_ds.cast_column("image", Image())
    test_ds = test_ds.cast_column("image", Image())

    # keep raw copies for visualization
    train_raw, val_raw, test_raw = train_ds, val_ds, test_ds

    # define image preprocessing for resnet
    img_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # transform function for batch processing
    def hf_transform(batch):
        pixel_values = [img_tfm(img.convert("RGB")) for img in batch["image"]]
        labels = []
        for i in range(len(pixel_values)):
            y = np.array([float(batch[c][i]) for c in TARGET_COLS], dtype=np.float32)
            y = (y - y_mean) / y_std
            labels.append(y)
        return {"pixel_values": pixel_values, "labels": labels}

    # create transformed dataset views
    train_tf = train_raw.with_transform(hf_transform)
    val_tf = val_raw.with_transform(hf_transform)
    test_tf = test_raw.with_transform(hf_transform)

    # collate function for batching
    def collate_fn(samples):
        x = torch.stack([s["pixel_values"] for s in samples])
        y = torch.from_numpy(np.stack([s["labels"] for s in samples]).astype(np.float32))
        return x, y

    # create pytorch dataloaders
    train_loader = DataLoader(
        train_tf, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(val_tf, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_tf, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_raw, val_raw, test_raw, y_mean, y_std


if __name__ == "__main__":
    load_data()
