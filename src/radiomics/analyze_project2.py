from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

CATEGORY_ORDER = ["0", "1-99", "100-399", ">=400"]
NON_FEATURE_COLS = {
    "case_id",
    "agatston_score",
    "agatston_category",
    "agatston_lesion_count",
    "thresholded_voxels",
    "spacing_x",
    "spacing_y",
    "spacing_z",
}


def candidate_features(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan, np.nan
    if tmp.iloc[:, 0].nunique() < 2 or tmp.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan
    rho, p = spearmanr(tmp.iloc[:, 0], tmp.iloc[:, 1])
    return float(rho), float(p)


def safe_kruskal(df: pd.DataFrame, feature: str) -> tuple[float, float, int]:
    groups = []
    used = 0
    for cat in CATEGORY_ORDER:
        vals = df.loc[df["agatston_category"] == cat, feature].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            used += 1
    if used < 2:
        return np.nan, np.nan, used
    stat, p = kruskal(*groups)
    return float(stat), float(p), used


def save_heatmap(corr_df: pd.DataFrame, out_png: Path) -> None:
    if corr_df.empty:
        return
    plot_df = corr_df.sort_values("spearman_abs", ascending=False).head(20).copy()
    vals = plot_df[["spearman_rho"]].to_numpy()
    labels = plot_df["feature"].tolist()

    fig_h = max(6, 0.35 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    im = ax.imshow(vals, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["Agatston score"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Top |Spearman| radiomics associations")
    for i, v in enumerate(plot_df["spearman_rho"]):
        ax.text(0, i, f"{v:.3f}" if pd.notna(v) else "nan", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, feature: str, out_png: Path) -> None:
    tmp = df[[feature, "agatston_score"]].dropna()
    if len(tmp) < 3:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(tmp[feature], tmp["agatston_score"], alpha=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel("Agatston score")
    ax.set_title(f"{feature} vs Agatston score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_boxplot(df: pd.DataFrame, feature: str, out_png: Path) -> None:
    groups = []
    labels = []
    for cat in CATEGORY_ORDER:
        vals = df.loc[df["agatston_category"] == cat, feature].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(cat)
    if len(groups) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_xlabel("Agatston category")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} across Agatston categories")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Project 2 merged radiomics + Agatston table.")
    ap.add_argument("--input-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df["agatston_category"] = pd.Categorical(df["agatston_category"], categories=CATEGORY_ORDER, ordered=True)

    features = candidate_features(df)
    corr_rows = []
    kw_rows = []

    for feature in features:
        rho, p = safe_spearman(df[feature], df["agatston_score"])
        corr_rows.append({
            "feature": feature,
            "spearman_rho": rho,
            "spearman_p": p,
            "spearman_abs": abs(rho) if pd.notna(rho) else np.nan,
            "n_non_null": int(df[feature].notna().sum()),
        })
        stat, kw_p, used = safe_kruskal(df, feature)
        kw_rows.append({
            "feature": feature,
            "kruskal_stat": stat,
            "kruskal_p": kw_p,
            "groups_used": used,
            "n_non_null": int(df[feature].notna().sum()),
        })

    corr_df = pd.DataFrame(corr_rows).sort_values(["spearman_abs", "spearman_p"], ascending=[False, True])
    kw_df = pd.DataFrame(kw_rows).sort_values(["kruskal_p", "kruskal_stat"], ascending=[True, False])

    corr_df.to_csv(args.out_dir / "spearman_correlations.csv", index=False)
    kw_df.to_csv(args.out_dir / "kruskal_wallis.csv", index=False)

    sig = corr_df.merge(kw_df, on=["feature", "n_non_null"], how="outer")
    sig = sig[(sig["spearman_p"] < 0.05) | (sig["kruskal_p"] < 0.05)].copy()
    sig = sig.sort_values(["spearman_p", "kruskal_p"], na_position="last")
    sig.to_csv(args.out_dir / "significant_associations.csv", index=False)

    summary = {
        "n_cases": int(len(df)),
        "n_features_tested": int(len(features)),
        "agatston_category_counts": df["agatston_category"].value_counts(dropna=False).sort_index().to_dict(),
        "top_positive_spearman": corr_df.head(10).to_dict(orient="records"),
        "top_kw": kw_df.head(10).to_dict(orient="records"),
    }
    pd.Series(summary).to_json(args.out_dir / "analysis_summary.json", indent=2)

    save_heatmap(corr_df, figures_dir / "top_spearman_heatmap.png")
    top_corr_features = corr_df["feature"].head(3).tolist()
    for feat in top_corr_features:
        save_scatter(df, feat, figures_dir / f"scatter_{feat}.png")
    top_kw_features = kw_df["feature"].head(3).tolist()
    for feat in top_kw_features:
        save_boxplot(df, feat, figures_dir / f"boxplot_{feat}.png")

    print(f"[OK] analyzed {len(df)} cases and {len(features)} features -> {args.out_dir}")
    print(corr_df.head(10).to_string(index=False))
    print(kw_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
