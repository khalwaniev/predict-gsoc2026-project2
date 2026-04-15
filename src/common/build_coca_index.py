from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

IMAGE_EXTS = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"}
MASK_HINTS_DEFAULT = ["mask", "seg", "label", "lesion", "calcium", "cac"]
IMAGE_HINTS_DEFAULT = ["image", "img", "ct", "volume", "scan"]


def normalize_stem(path: Path) -> str:
    name = path.name
    for suffix in [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"(?i)([_\-]?(mask|seg|label|lesion|calcium|cac|image|img|ct|volume|scan))+$", "", name)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name.lower()


def has_any_token(text: str, tokens: Iterable[str]) -> bool:
    text = text.lower()
    return any(tok.lower() in text for tok in tokens)


def list_candidate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS or p.name.lower().endswith(".nii.gz"):
            files.append(p)
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a manifest pairing COCA image volumes and masks.")
    ap.add_argument("--root", required=True, type=Path, help="Root directory containing processed NIfTI-like files")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV manifest path")
    ap.add_argument("--image-contains", nargs="*", default=IMAGE_HINTS_DEFAULT)
    ap.add_argument("--mask-contains", nargs="*", default=MASK_HINTS_DEFAULT)
    args = ap.parse_args()

    files = list_candidate_files(args.root)
    if not files:
        raise FileNotFoundError(f"No supported volumetric files found under: {args.root}")

    images: dict[str, list[Path]] = {}
    masks: dict[str, list[Path]] = {}

    for p in files:
        key = normalize_stem(p)
        lower = str(p).lower()
        if has_any_token(lower, args.mask_contains):
            masks.setdefault(key, []).append(p)
        elif has_any_token(lower, args.image_contains):
            images.setdefault(key, []).append(p)
        else:
            images.setdefault(key, []).append(p)

    rows = []
    all_keys = sorted(set(images) | set(masks))
    for key in all_keys:
        image_candidates = sorted(images.get(key, []))
        mask_candidates = sorted(masks.get(key, []))
        if not image_candidates or not mask_candidates:
            continue
        rows.append(
            {
                "case_id": key,
                "image_path": str(image_candidates[0].resolve()),
                "mask_path": str(mask_candidates[0].resolve()),
                "n_image_candidates": len(image_candidates),
                "n_mask_candidates": len(mask_candidates),
            }
        )

    if not rows:
        raise RuntimeError(
            "No image/mask pairs were found. Rerun with explicit --image-contains and --mask-contains hints "
            "or inspect the processed COCA folder naming."
        )

    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[OK] wrote {len(df)} paired cases -> {args.output}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
