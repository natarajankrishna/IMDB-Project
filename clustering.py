"""
Clustering — Module 2 Part 2
=============================
Applies partitional (k-means) and hierarchical (cosine-distance) clustering to a
numeric, unlabeled feature set derived from the cleaned movie dataset, to explore
whether films group into distinct structural "types."

INPUT:  output/data/clean_movies.csv          (from imdb_pipeline.py, Part 2 EDA)
        output/data/akas_region_counts.csv    (from imdb_pipeline.py, Part 2 EDA)

OUTPUT: output/data/unsupervised_features.csv          — the shared numeric dataset
        output/samples/unsupervised-data-sample.png     — small image of that data
        output/charts/clustering-overview-partitional.png
        output/charts/clustering-overview-hierarchical.png
        output/charts/clustering-dendrogram.png
        output/charts/clustering-k-comparison.png
        output/charts/clustering-silhouette-curve.png
        output/charts/clustering-silhouette-plot.png
        output/charts/clustering-scatter-pca2d.png
        output/charts/clustering-hclust-scatter.png

HOW TO RUN:
    pip install pandas numpy scikit-learn scipy matplotlib seaborn
    python clustering.py
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "output/data"
SAMPLES_DIR = "output/samples"
CHARTS_DIR = "output/charts"
for d in [DATA_DIR, SAMPLES_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURES = ["averageRating", "numVotes_log", "runtimeMinutes", "startYear", "num_genres", "region_count"]
K_VALUES_TO_COMPARE = [3, 4, 5]          # at least 3 k values, as required
SILHOUETTE_K_RANGE = range(2, 11)        # used to find the "best k"
CLUSTER_SAMPLE_N = 1000                   # subsample for both k-means & hclust (fair comparison + hclust is O(n^2))
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", font_scale=0.9)
GOLD, SLATE, RED = "#C9A05C", "#9CA3AF", "#A8442E"


def save_dataframe_image(df, path, title, n_rows=8):
    sample = df.head(n_rows)
    fig, ax = plt.subplots(figsize=(min(16, 1.4 * len(sample.columns)), 0.5 * n_rows + 1))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=12)
    tbl = ax.table(cellText=sample.round(3).astype(str).values, colLabels=sample.columns,
                    cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  [saved image] {path}")


# ============================================================
# STEP 1 — BUILD THE SHARED NUMERIC FEATURE SET (unlabeled, numeric-only)
# ============================================================
# Clustering (and PCA) require ONLY unlabeled numeric data — no text, no IDs used
# as inputs, no target/label column. Below, tconst is kept ONLY as a row identifier
# for joining back to results later; it is explicitly excluded from the feature
# matrix used by the models themselves.

def build_feature_set():
    print("=== STEP 1: Building numeric feature set ===")
    clean = pd.read_csv(os.path.join(DATA_DIR, "clean_movies.csv"))
    akas = pd.read_csv(os.path.join(DATA_DIR, "akas_region_counts.csv"))

    df = clean.merge(akas, on="tconst", how="left")
    df["region_count"] = df["region_count"].fillna(0)
    df["numVotes_log"] = np.log1p(df["numVotes"])

    # 'era' is carried along for later interpretation/plotting only — it is NOT
    # part of the numeric feature matrix (FEATURES) used by the clustering models.
    feature_df = df[["tconst"] + FEATURES + ["era"]].dropna()
    feature_df.to_csv(os.path.join(DATA_DIR, "unsupervised_features.csv"), index=False)
    print(f"  Feature set built: {len(feature_df):,} rows, {len(FEATURES)} numeric features")

    # image shows ONLY the numeric feature columns (plus the row identifier) —
    # 'era' is excluded here since it's categorical and not part of the actual
    # unlabeled numeric input the models receive
    save_dataframe_image(feature_df[["tconst"] + FEATURES], f"{SAMPLES_DIR}/unsupervised-data-sample.png",
                          "Unlabeled numeric feature set (used for Clustering & PCA)")
    return feature_df


# ============================================================
# STEP 2 — OVERVIEW ILLUSTRATIONS (synthetic examples, not the real data —
# these exist purely to visually explain the two clustering approaches)
# ============================================================

def overview_illustrations():
    print("\n=== STEP 2: Generating overview illustration images ===")
    X, _ = make_blobs(n_samples=180, centers=4, cluster_std=0.9, random_state=RANDOM_STATE)
    labels = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="flare", s=25, alpha=0.8)
    plt.title("Partitional Clustering (k-means) — illustrative example")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/clustering-overview-partitional.png", dpi=160); plt.close()

    X2, _ = make_blobs(n_samples=25, centers=3, cluster_std=0.6, random_state=1)
    Z = linkage(X2, method="average", metric="euclidean")
    plt.figure(figsize=(6, 5))
    dendrogram(Z, color_threshold=0, above_threshold_color=GOLD)
    plt.title("Hierarchical Clustering — illustrative dendrogram")
    plt.xlabel("Sample index"); plt.ylabel("Distance")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/clustering-overview-hierarchical.png", dpi=160); plt.close()


# ============================================================
# STEP 3 — K-MEANS (partitional clustering)
# ============================================================

def run_kmeans(X_scaled, X_full_scaled_index):
    print("\n=== STEP 3: K-means clustering ===")

    # (a) Compare at least 3 different k values
    fig, axes = plt.subplots(1, len(K_VALUES_TO_COMPARE), figsize=(5 * len(K_VALUES_TO_COMPARE), 4.5))
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_scaled)
    X_2d = pca2.transform(X_scaled)
    k_results = {}
    for ax, k in zip(axes, K_VALUES_TO_COMPARE):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        k_results[k] = {"model": km, "silhouette": sil}
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=km.labels_, cmap="flare", s=10, alpha=0.6)
        ax.set_title(f"k = {k}  (silhouette = {sil:.3f})")
        ax.set_xticks([]); ax.set_yticks([])
        print(f"  k={k}: silhouette score = {sil:.4f}")
    plt.suptitle("K-means Results Across Different k Values (PCA-projected for display)")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/clustering-k-comparison.png", dpi=160); plt.close()

    # (b) Silhouette curve across a wider k range to find the "best k"
    sil_scores = []
    for k in SILHOUETTE_K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))
    best_k = list(SILHOUETTE_K_RANGE)[int(np.argmax(sil_scores))]
    print(f"  Best k by silhouette score across k=2..10: k={best_k} (score={max(sil_scores):.4f})")

    plt.figure(figsize=(7, 4.5))
    plt.plot(list(SILHOUETTE_K_RANGE), sil_scores, marker="o", color=GOLD)
    plt.axvline(best_k, color=RED, linestyle="--", alpha=0.7, label=f"best k = {best_k}")
    plt.title("Silhouette Score by k (k-means)")
    plt.xlabel("k"); plt.ylabel("Average silhouette score"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/clustering-silhouette-curve.png", dpi=160); plt.close()

    # (c) Per-sample silhouette plot for the best k (classic silhouette visualization)
    best_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    sample_sil_values = silhouette_samples(X_scaled, best_km.labels_)
    palette = sns.color_palette("flare", best_k)
    plt.figure(figsize=(7, 5))
    y_lower = 10
    for i in range(best_k):
        cluster_sil = np.sort(sample_sil_values[best_km.labels_ == i])
        size = cluster_sil.shape[0]
        y_upper = y_lower + size
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                           facecolor=palette[i], alpha=0.8)
        plt.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10
    plt.axvline(x=silhouette_score(X_scaled, best_km.labels_), color=RED, linestyle="--",
                label="average score")
    plt.title(f"Per-Sample Silhouette Plot (k = {best_k})")
    plt.xlabel("Silhouette coefficient"); plt.ylabel("Cluster"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/clustering-silhouette-plot.png", dpi=160); plt.close()

    # (d) Non-dendrogram cluster visualization for the best k, PCA-projected to 2D
    X_2d_best = pca2.transform(X_scaled)
    plt.figure(figsize=(7, 5.5))
    plt.scatter(X_2d_best[:, 0], X_2d_best[:, 1], c=best_km.labels_, cmap="flare", s=14, alpha=0.65)
    plt.title(f"K-means Clusters (k = {best_k}), projected onto first 2 principal components")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/clustering-scatter-pca2d.png", dpi=160); plt.close()

    return best_k, best_km, k_results, sil_scores, pca2


# ============================================================
# STEP 4 — HIERARCHICAL CLUSTERING (cosine distance)
# ============================================================

def run_hierarchical(X_scaled, pca2, kmeans_best_k, kmeans_labels):
    print("\n=== STEP 4: Hierarchical clustering (cosine distance) ===")

    # scipy linkage with a precomputed cosine distance condensed matrix
    cosine_dist = pdist(X_scaled, metric="cosine")
    Z = linkage(cosine_dist, method="average")

    plt.figure(figsize=(10, 5.5))
    dendrogram(Z, truncate_mode="lastp", p=30, color_threshold=0.7 * max(Z[:, 2]))
    plt.title("Hierarchical Clustering Dendrogram (cosine distance, average linkage)")
    plt.xlabel("Cluster size (truncated view)"); plt.ylabel("Cosine distance")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/clustering-dendrogram.png", dpi=160); plt.close()

    # suggested k: largest gap in the last 10 merge distances (a standard dendrogram heuristic)
    last_merges = Z[-10:, 2]
    gaps = np.diff(last_merges)
    suggested_k = len(last_merges) - int(np.argmax(gaps))
    print(f"  Dendrogram-suggested k (largest gap heuristic): k={suggested_k}")

    # fit AgglomerativeClustering at both the hclust-suggested k AND the k-means best k,
    # so we can directly compare labelings at the same k
    hclust_at_kmeans_k = AgglomerativeClustering(n_clusters=kmeans_best_k, metric="cosine",
                                                  linkage="average").fit(X_scaled)
    ari = adjusted_rand_score(kmeans_labels, hclust_at_kmeans_k.labels_)
    print(f"  Adjusted Rand Index between k-means and hclust labels (both at k={kmeans_best_k}): {ari:.4f}")
    print("  (ARI ranges -1 to 1; close to 0 = little agreement, close to 1 = near-identical labelings)")

    X_2d = pca2.transform(X_scaled)
    plt.figure(figsize=(7, 5.5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=hclust_at_kmeans_k.labels_, cmap="flare", s=14, alpha=0.65)
    plt.title(f"Hierarchical Clustering Labels (k = {kmeans_best_k}), projected onto PC1/PC2")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/clustering-hclust-scatter.png", dpi=160); plt.close()

    return suggested_k, ari


# ============================================================
# MAIN
# ============================================================

def main():
    feature_df = build_feature_set()
    overview_illustrations()

    # subsample for clustering — same subsample used for BOTH k-means and hclust
    # so the part (d) comparison is apples-to-apples, and so cosine-distance hclust
    # (which needs a full pairwise distance matrix) stays computationally feasible
    sample_df = feature_df.sample(n=min(CLUSTER_SAMPLE_N, len(feature_df)), random_state=RANDOM_STATE)
    X = sample_df[FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)
    print(f"\nUsing a subsample of {len(sample_df):,} rows for k-means + hierarchical clustering "
          f"(full feature set has {len(feature_df):,} rows — hierarchical clustering's pairwise "
          f"distance matrix doesn't scale to the full set).")

    best_k, best_km, k_results, sil_scores, pca2 = run_kmeans(X_scaled, sample_df.index)
    suggested_k, ari = run_hierarchical(X_scaled, pca2, best_k, best_km.labels_)

    print("\n=== SUMMARY ===")
    print(f"K-means best k (by silhouette): {best_k}")
    print(f"Hierarchical clustering suggested k (dendrogram gap heuristic): {suggested_k}")
    print(f"Agreement between the two methods at matched k (Adjusted Rand Index): {ari:.4f}")
    print("\nDone. Charts saved to output/charts/, data sample to output/samples/, "
          "feature CSV to output/data/unsupervised_features.csv")


if __name__ == "__main__":
    main()
