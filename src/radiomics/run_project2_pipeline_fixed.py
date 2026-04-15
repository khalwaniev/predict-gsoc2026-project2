from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run core Project 2 pipeline end-to-end with separate manifests.")
    ap.add_argument("--agatston-manifest", required=True, type=Path)
    ap.add_argument("--radiomics-manifest", required=True, type=Path)
    ap.add_argument("--radiomics-config", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    agatston_csv = args.out_dir / "agatston_scores.csv"
    radiomics_csv = args.out_dir / "radiomics_features.csv"
    merged_csv = args.out_dir / "project2_merged.csv"
    stats_dir = args.out_dir / "analysis"

    base = Path(__file__).resolve().parent

    run([sys.executable, str(base / "agatston.py"), "--manifest", str(args.agatston_manifest), "--output", str(agatston_csv)])
    run([sys.executable, str(base / "extract_radiomics.py"), "--manifest", str(args.radiomics_manifest), "--config", str(args.radiomics_config), "--output", str(radiomics_csv)])

    scores = pd.read_csv(agatston_csv)
    feats = pd.read_csv(radiomics_csv)
    merged = scores.merge(feats, on="case_id", how="inner").sort_values("case_id").reset_index(drop=True)
    merged.to_csv(merged_csv, index=False)
    print(f"[OK] merged -> {merged_csv}")

    run([sys.executable, str(base / "analyze_project2.py"), "--input-csv", str(merged_csv), "--out-dir", str(stats_dir)])


if __name__ == "__main__":
    main()
