#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

CATEGORY_ORDER = ["0", "1-99", "100-399", ">=400"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a balanced Project 2 subset from full Agatston results.")
    ap.add_argument("--usable-cases", required=True, help="usable_cases.csv produced by make_project2_full_manifests.py")
    ap.add_argument("--agatston-csv", required=True, help="agatston scores csv produced on all usable cases")
    ap.add_argument("--n-per-group", type=int, default=6, help="Target cases per Agatston category")
    ap.add_argument("--out-subset-csv", required=True, help="Output subset CSV with canonical and resampled paths")
    ap.add_argument("--out-manifest-dir", required=True, help="Output directory for balanced manifests")
    args = ap.parse_args()

    usable = pd.read_csv(args.usable_cases).copy()
    ag = pd.read_csv(args.agatston_csv).copy()
    usable["scan_id"] = usable["scan_id"].astype(str)
    ag["case_id"] = ag["case_id"].astype(str)

    merged = usable.merge(ag, left_on="scan_id", right_on="case_id", how="inner")
    if merged.empty:
        raise SystemExit("No overlap between usable cases and Agatston results.")

    counts = merged["agatston_category"].value_counts(dropna=False).to_dict()
    print("[INFO] category counts in all usable cases:", counts)

    picks = []
    for cat in CATEGORY_ORDER:
        g = merged[merged["agatston_category"] == cat].copy()
        if cat == "0":
            # Prefer clean true-zero masks with zero annotated voxels; then smallest patient_id for determinism.
            g = g.sort_values(["voxels", "num_slices", "patient_id"], ascending=[True, True, True])
        else:
            # Prefer stronger positive examples with richer masks and larger Agatston within the category.
            g = g.sort_values(["agatston_score", "voxels", "num_slices", "patient_id"], ascending=[False, False, False, True])
        take = g.head(args.n_per_group)
        if len(take) < args.n_per_group:
            print(f"[WARN] category {cat}: requested {args.n_per_group}, available {len(take)}")
        picks.append(take)

    subset = pd.concat(picks, ignore_index=True)
    subset = subset.drop_duplicates(subset=["scan_id"]).copy()

    # Keep a useful, explicit column order.
    cols = [
        "patient_id", "scan_id", "voxels", "num_slices",
        "agatston_score", "agatston_category", "agatston_lesion_count",
        "thresholded_voxels", "spacing_x", "spacing_y", "spacing_z",
        "canonical_img", "canonical_seg", "resampled_img", "resampled_seg",
    ]
    for c in cols:
        if c not in subset.columns:
            subset[c] = pd.NA
    subset = subset[cols].copy()

    out_subset = Path(args.out_subset_csv)
    out_subset.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_subset, index=False)

    out_dir = Path(args.out_manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agatston_manifest = subset[["scan_id", "canonical_img", "canonical_seg"]].rename(columns={
        "scan_id": "case_id", "canonical_img": "image_path", "canonical_seg": "mask_path"
    })
    radiomics_manifest = subset[["scan_id", "resampled_img", "resampled_seg"]].rename(columns={
        "scan_id": "case_id", "resampled_img": "image_path", "resampled_seg": "mask_path"
    })
    agatston_manifest.to_csv(out_dir / "agatston_manifest.csv", index=False)
    radiomics_manifest.to_csv(out_dir / "radiomics_manifest.csv", index=False)

    print(f"[OK] balanced subset -> {out_subset} ({len(subset)} rows)")
    print(subset[["patient_id", "scan_id", "agatston_category", "agatston_score", "voxels"]].to_string(index=False))
    print("\n[OK] balanced manifests written to", out_dir)
    print(subset["agatston_category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0).to_string())


if __name__ == "__main__":
    main()
