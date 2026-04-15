from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split


@dataclass
class WindowConfig:
    center: float = 300.0
    width: float = 900.0


class CocaVolumeDataset:
    def __init__(self, manifest: pd.DataFrame, augment: bool = False, window: Optional[WindowConfig] = None):
        self.manifest = manifest.reset_index(drop=True)
        self.augment = augment
        self.window = window or WindowConfig()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        image = sitk.ReadImage(row.image_path)
        mask = sitk.ReadImage(row.mask_path)
        x = sitk.GetArrayFromImage(image).astype(np.float32)
        y = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
        x = apply_hu_window(x, self.window.center, self.window.width)
        if self.augment:
            x, y = mild_augment_3d(x, y)
        return {
            "case_id": row.case_id,
            "image": x[None, ...],
            "mask": y[None, ...],
            "spacing_xyz": image.GetSpacing(),
        }


def apply_hu_window(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    arr = np.clip(arr, low, high)
    arr = (arr - low) / max(high - low, 1e-6)
    return arr.astype(np.float32)


def mild_augment_3d(x: np.ndarray, y: np.ndarray):
    if random.random() < 0.5:
        x = np.flip(x, axis=2).copy()
        y = np.flip(y, axis=2).copy()
    if random.random() < 0.5:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=1).copy()
    if random.random() < 0.5:
        x = x + np.random.normal(0.0, 0.01, size=x.shape).astype(np.float32)
    if random.random() < 0.5:
        x = x + np.random.uniform(-0.02, 0.02)
    x = np.clip(x, 0.0, 1.0)
    return x, y


def read_case_stats(image_path: str, mask_path: str) -> dict:
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    m = sitk.GetArrayFromImage(mask) > 0
    spacing = image.GetSpacing()  # x, y, z
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    pos_voxels = int(m.sum())
    total_voxels = int(np.prod(m.shape))
    return {
        "size_z": int(arr.shape[0]),
        "size_y": int(arr.shape[1]),
        "size_x": int(arr.shape[2]),
        "spacing_x": float(spacing[0]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[2]),
        "hu_min": float(arr.min()),
        "hu_max": float(arr.max()),
        "hu_mean": float(arr.mean()),
        "positive_voxels": pos_voxels,
        "positive_fraction": float(pos_voxels / max(total_voxels, 1)),
        "calcium_volume_mm3": float(pos_voxels * voxel_vol),
        "has_calcium": int(pos_voxels > 0),
    }


def make_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    strata_col = "agatston_category" if "agatston_category" in df.columns else "has_calcium"
    temp_size = 0.30
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=seed,
        stratify=df[strata_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[strata_col],
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], axis=0).sort_values("case_id").reset_index(drop=True)


def try_torch_sanity_check(manifest: pd.DataFrame, batch_size: int) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader

        ds = CocaVolumeDataset(manifest, augment=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        return {
            "torch_ok": True,
            "image_batch_shape": list(batch["image"].shape),
            "mask_batch_shape": list(batch["mask"].shape),
        }
    except Exception as exc:  # pragma: no cover
        return {"torch_ok": False, "torch_error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Common task pipeline for PrediCT COCA preprocessing and dataloader sanity checks.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--score-csv", type=Path, default=None, help="Optional Agatston CSV to stratify by categories")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    required = {"case_id", "image_path", "mask_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    rows = []
    for row in manifest.itertuples(index=False):
        stats = read_case_stats(row.image_path, row.mask_path)
        stats["case_id"] = row.case_id
        stats["image_path"] = row.image_path
        stats["mask_path"] = row.mask_path
        rows.append(stats)

    stats_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)

    if args.score_csv and args.score_csv.exists():
        score_df = pd.read_csv(args.score_csv)
        if {"case_id", "agatston_category"} <= set(score_df.columns):
            stats_df = stats_df.merge(score_df[["case_id", "agatston_category"]], on="case_id", how="left")

    splits_df = make_splits(stats_df)

    case_stats_path = args.out_dir / "case_level_stats.csv"
    splits_path = args.out_dir / "splits.csv"
    summary_path = args.out_dir / "summary_stats.json"

    stats_df.to_csv(case_stats_path, index=False)
    splits_df.to_csv(splits_path, index=False)

    summary = {
        "n_cases": int(len(stats_df)),
        "n_with_calcium": int(stats_df["has_calcium"].sum()),
        "spacing_xyz_mean": {
            "x": float(stats_df["spacing_x"].mean()),
            "y": float(stats_df["spacing_y"].mean()),
            "z": float(stats_df["spacing_z"].mean()),
        },
        "shape_median_zyx": {
            "z": int(stats_df["size_z"].median()),
            "y": int(stats_df["size_y"].median()),
            "x": int(stats_df["size_x"].median()),
        },
        "calcium_volume_mm3_median": float(stats_df["calcium_volume_mm3"].median()),
        "split_counts": splits_df["split"].value_counts().to_dict(),
        "dataloader_sanity": try_torch_sanity_check(manifest[["case_id", "image_path", "mask_path"]], args.batch_size),
        "windowing": {"center": 300.0, "width": 900.0},
        "augmentation_note": "Mild flips, gaussian noise, and intensity shift only; kept conservative for radiomics compatibility.",
        "sampling_strategy_note": "Primary split is stratified. Class-imbalance sampling is not forced at extraction time; if Project 1 is pursued later, use weighted sampling on the same split file.",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] case stats -> {case_stats_path}")
    print(f"[OK] splits -> {splits_path}")
    print(f"[OK] summary -> {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
