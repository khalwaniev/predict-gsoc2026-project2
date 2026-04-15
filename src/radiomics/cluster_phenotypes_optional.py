from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in {"case_id", "agatston_category", "agatston_score"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Optional clustering / phenotype discovery for Project 2.")
    ap.add_argument("--input-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    cols = feature_cols(df)
    if len(cols) < 3:
        raise ValueError("Need at least 3 numeric features for clustering")

    prep = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X = prep.fit_transform(df[cols])

    silhouette_rows = []
    best_model = None
    best_score = -1.0
    best_k = None
    for k in range(args.k_min, args.k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X, labels)
        silhouette_rows.append({"k": k, "silhouette": sil})
        if sil > best_score:
            best_score = sil
            best_model = model
            best_k = k

    if best_model is None:
        raise RuntimeError("Could not fit a valid clustering model")

    labels = best_model.predict(X)
    out_df = df.copy()
    out_df["cluster"] = labels
    out_df.to_csv(args.out_dir / "cluster_assignments.csv", index=False)
    pd.DataFrame(silhouette_rows).to_csv(args.out_dir / "kmeans_silhouette_scan.csv", index=False)

    pca = PCA(n_components=2, random_state=42)
    pca_xy = pca.fit_transform(X)

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        emb_name = "umap"
    else:
        reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
        emb = reducer.fit_transform(X)
        emb_name = "tsne"

    plt.figure(figsize=(6, 5))
    plt.scatter(pca_xy[:, 0], pca_xy[:, 1], c=labels)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"KMeans clusters in PCA space (k={best_k})")
    plt.tight_layout()
    plt.savefig(args.out_dir / "clusters_pca.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels)
    plt.xlabel(f"{emb_name.upper()} 1")
    plt.ylabel(f"{emb_name.upper()} 2")
    plt.title(f"KMeans clusters in {emb_name.upper()} space (k={best_k})")
    plt.tight_layout()
    plt.savefig(args.out_dir / f"clusters_{emb_name}.png", dpi=160)
    plt.close()

    summary = (
        out_df.groupby("cluster")
        .agg(
            n_cases=("case_id", "count"),
            agatston_mean=("agatston_score", "mean"),
            agatston_median=("agatston_score", "median"),
        )
        .reset_index()
    )
    summary.to_csv(args.out_dir / "cluster_summary.csv", index=False)

    print(f"[OK] best k = {best_k}, silhouette = {best_score:.4f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
