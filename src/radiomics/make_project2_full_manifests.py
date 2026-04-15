#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Build full canonical/resampled manifests for all usable Project 2 cases.")
    ap.add_argument("--scan-index", required=True, help="Path to scan_index.csv from COCA_processor")
    ap.add_argument("--canonical-root", required=True, help="Root of canonical images dir")
    ap.add_argument("--resampled-root", required=True, help="Root of resampled images dir")
    ap.add_argument("--out-dir", required=True, help="Output directory for manifests")
    args = ap.parse_args()

    scan_index = Path(args.scan_index)
    canonical_root = Path(args.canonical_root)
    resampled_root = Path(args.resampled_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scan_index).copy()
    df["scan_id"] = df["scan_id"].astype(str)

    def canonical_paths(scan_id: str) -> tuple[str, str]:
        base = canonical_root / scan_id
        return str(base / f"{scan_id}_img.nii.gz"), str(base / f"{scan_id}_seg.nii.gz")

    def resampled_paths(scan_id: str) -> tuple[str, str]:
        base = resampled_root / scan_id
        return str(base / f"{scan_id}_img.nii.gz"), str(base / f"{scan_id}_seg.nii.gz")

    df[["canonical_img", "canonical_seg"]] = df["scan_id"].apply(lambda s: pd.Series(canonical_paths(s)))
    df[["resampled_img", "resampled_seg"]] = df["scan_id"].apply(lambda s: pd.Series(resampled_paths(s)))

    df["canonical_ok"] = df["canonical_img"].map(lambda p: Path(p).exists()) & df["canonical_seg"].map(lambda p: Path(p).exists())
    df["resampled_ok"] = df["resampled_img"].map(lambda p: Path(p).exists()) & df["resampled_seg"].map(lambda p: Path(p).exists())
    usable = df[df["canonical_ok"] & df["resampled_ok"]].copy()

    agatston_manifest = usable[["scan_id", "canonical_img", "canonical_seg"]].rename(columns={
        "scan_id": "case_id", "canonical_img": "image_path", "canonical_seg": "mask_path"
    })
    radiomics_manifest = usable[["scan_id", "resampled_img", "resampled_seg"]].rename(columns={
        "scan_id": "case_id", "resampled_img": "image_path", "resampled_seg": "mask_path"
    })

    usable_csv = out_dir / "usable_cases.csv"
    agatston_csv = out_dir / "agatston_manifest_all.csv"
    radiomics_csv = out_dir / "radiomics_manifest_all.csv"

    usable.to_csv(usable_csv, index=False)
    agatston_manifest.to_csv(agatston_csv, index=False)
    radiomics_manifest.to_csv(radiomics_csv, index=False)

    print(f"[OK] usable cases -> {usable_csv} ({len(usable)} rows)")
    print(f"[OK] agatston manifest -> {agatston_csv}")
    print(f"[OK] radiomics manifest -> {radiomics_csv}")
    print(usable[["patient_id", "scan_id", "voxels", "num_slices"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
