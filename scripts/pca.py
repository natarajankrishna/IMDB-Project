"""
PCA — Module 2 Part 2
=======================
Applies Principal Component Analysis to the same numeric, unlabeled feature set
used for clustering, to see how much of the variation between films can be
captured in a small number of underlying dimensions.

INPUT:  output/data/unsupervised_features.csv   (built by clustering.py — run that first,
        or this script will build it itself from clean_movies.csv + akas_region_counts.csv)

OUTPUT: output/samples/unsupervised-data-sample.png     — small image of the data (shared with Clustering tab)
        output/charts/pca-overview-eigenvectors.png
        output/charts/pca-overview-dimensionality.png
        output/charts/pca-scree-plot.png
        output/charts/pca-biplot.png
        output/charts/pca-scatter-by-era.png

HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib seaborn
    python pca.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = "output/data"
SAMPLES_DIR = "output/samples"
CHARTS_DIR = "output/charts"
for d in [DATA_DIR, SAMPLES_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURES = ["averageRating", "numVotes_log", "runtimeMinutes", "startYear", "num_genres", "region_count"]
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", font_scale=0.9)
GOLD, SLATE, RED = "#C9A05C", "#9CA3AF", "#A8442E"


# ============================================================
# STEP 1 — LOAD (or build) THE SHARED NUMERIC FEATURE SET
# ============================================================
# PCA requires the same kind of input as clustering: unlabeled, numeric, and —
# importantly for PCA specifically — standardized, since PCA is sensitive to
# the scale of each variable (a feature measured in the thousands, like vote
# count, would otherwise dominate the components purely because of its units).

def load_feature_set():
    print("=== STEP 1: Loading numeric feature set ===")
    feature_path = os.path.join(DATA_DIR, "unsupervised_features.csv")
    if os.path.exists(feature_path):
        feature_df = pd.read_csv(feature_path)
        print(f"  Loaded existing feature set: {len(feature_df):,} rows")
        return feature_df

    # fallback: build it the same way clustering.py does, if it hasn't been run yet
    clean = pd.read_csv(os.path.join(DATA_DIR, "clean_movies.csv"))
    akas = pd.read_csv(os.path.join(DATA_DIR, "akas_region_counts.csv"))
    df = clean.merge(akas, on="tconst", how="left")
    df["region_count"] = df["region_count"].fillna(0)
    df["numVotes_log"] = np.log1p(df["numVotes"])
    feature_df = df[["tconst"] + FEATURES + ["era"]].dropna()
    feature_df.to_csv(feature_path, index=False)
    print(f"  Built feature set: {len(feature_df):,} rows")
    return feature_df


# ============================================================
# STEP 2 — OVERVIEW ILLUSTRATIONS (synthetic, for explaining the concept)
# ============================================================

def overview_illustrations():
    print("\n=== STEP 2: Generating overview illustration images ===")

    # (a) classic 2D PCA illustration: correlated synthetic data + eigenvector arrows
    rng = np.random.default_rng(RANDOM_STATE)
    x = rng.normal(0, 1.5, 200)
    y = 0.8 * x + rng.normal(0, 0.6, 200)
    data = np.column_stack([x, y])
    data_centered = data - data.mean(axis=0)
    pca_demo = PCA(n_components=2).fit(data_centered)

    plt.figure(figsize=(6, 5.5))
    plt.scatter(data_centered[:, 0], data_centered[:, 1], alpha=0.4, color=SLATE, s=18)
    origin = np.zeros(2)
    for i, (comp, var) in enumerate(zip(pca_demo.components_, pca_demo.explained_variance_)):
        vec = comp * np.sqrt(var) * 2.5
        plt.annotate("", xy=vec, xytext=origin,
                     arrowprops=dict(arrowstyle="->", color=[GOLD, RED][i], lw=2.5))
        plt.text(vec[0] * 1.1, vec[1] * 1.1, f"PC{i+1}", color=[GOLD, RED][i], fontweight="bold")
    plt.title("Eigenvectors of a 2D Dataset — illustrative example")
    plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/pca-overview-eigenvectors.png", dpi=160); plt.close()

    # (b) dimensionality reduction illustration: 3D synthetic data -> 2D projection, side by side
    z = 0.5 * x + 0.3 * y + rng.normal(0, 0.4, 200)
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(x, y, z, alpha=0.4, color=SLATE, s=14)
    ax1.set_title("Original: 3 dimensions")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    data3d = np.column_stack([x, y, z])
    data3d_centered = data3d - data3d.mean(axis=0)
    proj = PCA(n_components=2).fit_transform(data3d_centered)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(proj[:, 0], proj[:, 1], alpha=0.4, color=GOLD, s=14)
    ax2.set_title("Reduced: 2 principal components")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    plt.suptitle("Dimensionality Reduction — illustrative example")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/pca-overview-dimensionality.png", dpi=160); plt.close()


# ============================================================
# STEP 3 — RUN PCA ON THE REAL DATA
# ============================================================

def run_pca(feature_df):
    print("\n=== STEP 3: Running PCA ===")
    X = feature_df[FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=len(FEATURES), random_state=RANDOM_STATE)
    scores = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    for i, (e, c) in enumerate(zip(explained, cumulative), start=1):
        print(f"  PC{i}: explains {e*100:.1f}% of variance (cumulative: {c*100:.1f}%)")

    # --- Scree plot (bar = individual variance, line = cumulative) ---
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.bar(range(1, len(explained) + 1), explained * 100, color=GOLD, alpha=0.85, label="Individual")
    ax1.set_xlabel("Principal component"); ax1.set_ylabel("% variance explained")
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative) + 1), cumulative * 100, color=RED, marker="o", label="Cumulative")
    ax2.set_ylabel("Cumulative % variance explained")
    ax2.axhline(80, color=SLATE, linestyle="--", alpha=0.6)
    plt.title("Scree Plot — Variance Explained by Each Principal Component")
    fig.tight_layout(); plt.savefig(f"{CHARTS_DIR}/pca-scree-plot.png", dpi=160); plt.close()

    # --- Biplot: PC1/PC2 scores + loading vectors for each original variable ---
    plt.figure(figsize=(8, 7))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.12, s=10, color=SLATE)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    scale = 3.0  # purely visual scaling so arrows are visible against the point cloud
    for i, feat in enumerate(FEATURES):
        plt.annotate("", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
        plt.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15, feat,
                  color=RED, fontsize=9, fontweight="bold")
    plt.title("PCA Biplot — Movies Projected onto PC1/PC2, with Variable Loadings")
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/pca-biplot.png", dpi=160); plt.close()

    # --- PC1/PC2 scatter colored by era, for interpretability ---
    if "era" in feature_df.columns:
        plt.figure(figsize=(7.5, 6))
        for era, color in zip(feature_df["era"].unique(), [GOLD, RED]):
            mask = feature_df["era"].values == era
            plt.scatter(scores[mask, 0], scores[mask, 1], alpha=0.25, s=10, label=era, color=color)
        plt.title("Movies in PC1/PC2 Space, Colored by Era")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
        plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/pca-scatter-by-era.png", dpi=160); plt.close()

    return pca, scores, explained


def main():
    feature_df = load_feature_set()
    overview_illustrations()
    pca, scores, explained = run_pca(feature_df)
    print("\nDone. Charts saved to output/charts/.")


if __name__ == "__main__":
    main()
