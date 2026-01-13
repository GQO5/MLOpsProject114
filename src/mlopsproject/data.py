from pathlib import Path
import json
import numpy as np
import pandas as pd

import typer


TARGET_COLS = ["total_calories", "total_protein", "total_carb", "total_fat"]


def load_food_nutrients_dataset():
    """
    Load food-nutrients from the auto-converted Parquet artifacts on the HuggingFace Hub.
    """
    from huggingface_hub import HfFileSystem
    from datasets import load_dataset, Image

    fs = HfFileSystem()
    base = "datasets/mmathys/food-nutrients@refs/convert/parquet/default/test"

    files = fs.ls(base, detail=False)
    parquet_files = [f"hf://{p}" for p in files if p.endswith(".parquet")]

    if not parquet_files:
        raise RuntimeError(f"No parquet files found under: {base}")

    ds = load_dataset("parquet", data_files={"test": parquet_files}, split="test")

    if "image" in ds.column_names:
        ds = ds.cast_column("image", Image())

    print("Columns (parquet-direct):", ds.column_names)
    return ds


def make_split(n: int, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    split = np.array(["test"] * n, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "val"
    return split.tolist()


def extract_targets(sample: dict) -> dict:
    return {
        "total_calories": float(sample["total_calories"]),
        "total_protein": float(sample["total_protein"]),
        "total_carb": float(sample["total_carb"]),
        "total_fat": float(sample["total_fat"]),
    }


def build_manifest(ds, output_csv: Path, seed: int = 42):
    n = len(ds)
    splits = make_split(n, seed)

    rows = []
    for i in range(n):
        s = ds[i]
        rows.append(
            {
                "sample_id": i,
                "dish_id": str(s["id"]),
                "split": splits[i],
                "image_path": "",
                **extract_targets(s),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote manifest: {output_csv} (rows={n})")


def export_images_and_update_manifest(ds, manifest_csv: Path, images_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_csv)

    paths = []
    for i, row in df.iterrows():
        out_path = images_dir / f"{int(row.sample_id):06d}.jpg"
        if not out_path.exists():
            img = ds[int(row.sample_id)]["image"]
            img.save(out_path, format="JPEG")
        paths.append(str(out_path))

        if (i + 1) % 200 == 0:
            print(f"Exported {i+1}/{len(df)} images...")

    df["image_path"] = paths
    df.to_csv(manifest_csv, index=False)
    print(f"Updated manifest with image_path: {manifest_csv}")
    print(f"Images exported to: {images_dir}")


def compute_and_save_target_normalization(manifest_csv: Path, out_json: Path):
    df = pd.read_csv(manifest_csv)
    train_df = df[df["split"] == "train"]

    stats = {}
    for c in TARGET_COLS:
        mean = float(train_df[c].mean())
        std = float(train_df[c].std(ddof=0)) or 1.0
        stats[c] = {"mean": mean, "std": std}

    payload = {
        "computed_from_split": "train",
        "target_cols": TARGET_COLS,
        "stats": stats,
    }

    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote normalization artifact: {out_json}")


def preprocess(data_path: Path, output_folder: Path):
    print("Preprocessing data...")

    ds = load_food_nutrients_dataset()

    manifest = output_folder / "manifest.csv"
    images = output_folder / "images"
    norm_json = output_folder / "normalization.json"

    build_manifest(ds, manifest)
    export_images_and_update_manifest(ds, manifest, images)
    compute_and_save_target_normalization(manifest, norm_json)


if __name__ == "__main__":
    typer.run(preprocess)
