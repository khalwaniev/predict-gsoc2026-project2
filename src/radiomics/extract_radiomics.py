from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


FEATURE_ALIASES = {
    "shape_sphericity": ["original_shape_Sphericity"],
    "shape_surface_volume_ratio": ["original_shape_SurfaceVolumeRatio"],
    "shape_maximum_3d_diameter": ["original_shape_Maximum3DDiameter"],
    "glcm_contrast": ["original_glcm_Contrast"],
    "glcm_correlation": ["original_glcm_Correlation"],
    "glcm_inverse_difference_moment": ["original_glcm_Idm", "original_glcm_InverseDifferenceMoment"],
    "glszm_small_area_emphasis": ["original_glszm_SmallAreaEmphasis"],
    "glszm_large_area_emphasis": ["original_glszm_LargeAreaEmphasis"],
    "glszm_zone_percentage": ["original_glszm_ZonePercentage"],
    "glrlm_short_run_emphasis": ["original_glrlm_ShortRunEmphasis"],
    "glrlm_long_run_emphasis": ["original_glrlm_LongRunEmphasis"],
    "glrlm_run_percentage": ["original_glrlm_RunPercentage"],
}


def first_match(result: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        if key in result:
            try:
                return float(result[key])
            except Exception:
                return np.nan
    return np.nan


def optional_hu_stats(image: sitk.Image, mask: sitk.Image) -> dict[str, float]:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    m = sitk.GetArrayFromImage(mask) > 0
    if not m.any():
        return {
            "calcium_hu_mean": np.nan,
            "calcium_hu_max": np.nan,
            "calcium_volume_mm3": 0.0,
        }
    sx, sy, sz = image.GetSpacing()
    voxel_vol = float(sx * sy * sz)
    vals = arr[m]
    return {
        "calcium_hu_mean": float(vals.mean()),
        "calcium_hu_max": float(vals.max()),
        "calcium_volume_mm3": float(m.sum() * voxel_vol),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract required Project 2 radiomics features with PyRadiomics.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    extractor = featureextractor.RadiomicsFeatureExtractor(str(args.config))

    rows = []
    for row in df.itertuples(index=False):
        image = sitk.ReadImage(row.image_path)
        mask = sitk.ReadImage(row.mask_path)
        mask_arr = sitk.GetArrayFromImage(mask)
        if not np.any(mask_arr > 0):
            result = {}
        else:
            result = extractor.execute(row.image_path, row.mask_path)

        out = {"case_id": row.case_id}
        for out_name, candidates in FEATURE_ALIASES.items():
            out[out_name] = first_match(result, candidates)
        out.update(optional_hu_stats(image, mask))
        rows.append(out)

    out_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[OK] wrote {len(out_df)} feature rows -> {args.output}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
