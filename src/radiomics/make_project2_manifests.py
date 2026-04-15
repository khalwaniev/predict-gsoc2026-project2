from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Create separate Agatston and radiomics manifests from subset CSV.")
    ap.add_argument("--subset-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.subset_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    required = {"scan_id", "canonical_img", "canonical_seg", "resampled_img", "resampled_seg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"subset csv missing columns: {sorted(missing)}")

    agat = df[["scan_id", "canonical_img", "canonical_seg"]].copy()
    agat.columns = ["case_id", "image_path", "mask_path"]
    rad = df[["scan_id", "resampled_img", "resampled_seg"]].copy()
    rad.columns = ["case_id", "image_path", "mask_path"]

    agat_path = args.out_dir / "agatston_manifest.csv"
    rad_path = args.out_dir / "radiomics_manifest.csv"
    agat.to_csv(agat_path, index=False)
    rad.to_csv(rad_path, index=False)

    print(f"[OK] agatston manifest -> {agat_path}")
    print(f"[OK] radiomics manifest -> {rad_path}")
    print(agat.head(10).to_string(index=False))
    print(rad.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
