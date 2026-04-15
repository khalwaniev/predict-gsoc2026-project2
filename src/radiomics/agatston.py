from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi


def agatston_weight(max_hu: float) -> int:
    if max_hu < 130:
        return 0
    if max_hu < 200:
        return 1
    if max_hu < 300:
        return 2
    if max_hu < 400:
        return 3
    return 4


def categorize(score: float) -> str:
    if score <= 0:
        return "0"
    if score < 100:
        return "1-99"
    if score < 400:
        return "100-399"
    return ">=400"


def compute_agatston(image: sitk.Image, mask: sitk.Image) -> dict:
    img = sitk.GetArrayFromImage(image).astype(np.float32)   # z, y, x
    msk = sitk.GetArrayFromImage(mask) > 0
    sx, sy, sz = image.GetSpacing()  # x, y, z
    pixel_area_mm2 = float(sx * sy)

    score = 0.0
    lesion_count = 0
    thresholded_voxels = 0

    for z in range(img.shape[0]):
        lesion_map = (img[z] >= 130) & msk[z]
        thresholded_voxels += int(lesion_map.sum())
        if not lesion_map.any():
            continue
        labeled, n_comp = ndi.label(lesion_map)
        for comp_idx in range(1, n_comp + 1):
            lesion = labeled == comp_idx
            area_mm2 = float(lesion.sum() * pixel_area_mm2)
            if area_mm2 < 1.0:
                continue
            lesion_max = float(img[z][lesion].max())
            w = agatston_weight(lesion_max)
            if w == 0:
                continue
            score += area_mm2 * w
            lesion_count += 1

    return {
        "agatston_score": float(score),
        "agatston_category": categorize(score),
        "agatston_lesion_count": int(lesion_count),
        "thresholded_voxels": int(thresholded_voxels),
        "spacing_x": float(sx),
        "spacing_y": float(sy),
        "spacing_z": float(sz),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Agatston scores from image/mask pairs using original spacing.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    rows = []
    for row in df.itertuples(index=False):
        image = sitk.ReadImage(row.image_path)
        mask = sitk.ReadImage(row.mask_path)
        result = compute_agatston(image, mask)
        result["case_id"] = row.case_id
        rows.append(result)

    out_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[OK] wrote {len(out_df)} scores -> {args.output}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
