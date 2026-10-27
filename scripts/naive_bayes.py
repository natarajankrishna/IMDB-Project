"""
Naive Bayes -- Module 3
========================
Predicts a film's rating tier (Low / Medium / High) from structural features using:
  * Multinomial NB  -- the required, standard version (non-negative features scaled to [0, 1])
  * Bernoulli NB    -- binary features (genre flags, franchise flag, numeric features split at the median)
  * Gaussian NB     -- extra comparison, works directly on raw continuous features

Run order:  imdb_pipeline.py  ->  naive_bayes.py   (data prep is built into this file)

HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib seaborn
    python naive_bayes.py

OUTPUT: output/charts/nb-label-distribution.png, nb-class-feature-means.png,
        nb-confusion-matrix.png, nb-confusion-matrix-bernoulli.png, nb-model-comparison.png
        output/data/nb_metrics.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# SHARED DATA PREP -- labels, features, and the disjoint train/test split.
# This block is identical in naive_bayes.py and decision_trees.py. Whichever script
# runs first saves labeled_data.csv / train_set.csv / test_set.csv; the other then
# reloads those exact files, so both methods use the SAME train and test films.
# Label: rating_class = tertiles of averageRating (Low / Medium / High).
# averageRating itself is NOT a model input.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_DIR = "output/data"
SAMPLES_DIR = "output/samples"
CHARTS_DIR = "output/charts"
for d in [DATA_DIR, SAMPLES_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_TOP_GENRES = 12
LABEL_COL = "rating_class"
CLASS_ORDER = ["Low", "Medium", "High"]
NUMERIC_FEATURES = ["numVotes_log", "runtimeMinutes", "startYear", "num_genres", "region_count"]


def save_dataframe_image(df, path, title, columns=None, n_rows=8):
    """Small image of a few rows -- never the whole dataset."""
    sample = df if columns is None else df[columns]
    sample = sample.head(n_rows)
    fig, ax = plt.subplots(figsize=(min(16, 1.3 * len(sample.columns)), 0.5 * n_rows + 1))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=12)
    tbl = ax.table(cellText=sample.astype(str).values, colLabels=sample.columns,
                    cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved image] {path}")


def build_labeled_dataset():
    print("=== Building labeled dataset ===")
    clean = pd.read_csv(os.path.join(DATA_DIR, "clean_movies.csv"))
    akas = pd.read_csv(os.path.join(DATA_DIR, "akas_region_counts.csv"))

    df = clean.merge(akas, on="tconst", how="left")
    df["region_count"] = df["region_count"].fillna(0)
    df["numVotes_log"] = np.log1p(df["numVotes"]).round(4)
    df["is_franchise"] = df["is_franchise_heuristic"].astype(int)

    # one 0/1 indicator column per frequent genre (numeric-only data for sklearn)
    genre_sets = df["genres"].str.split(",").apply(set)
    top_genres = (df["genres"].str.split(",").explode().value_counts().head(N_TOP_GENRES).index)
    for g in top_genres:
        df[f"genre_{g}"] = genre_sets.apply(lambda s, g=g: int(g in s))

    # LABEL: tertiles of averageRating -> Low / Medium / High
    df[LABEL_COL], cutoffs = pd.qcut(df["averageRating"], q=3, labels=CLASS_ORDER, retbins=True)
    print(f"  Label cutoffs (rating): Low <= {cutoffs[1]:.2f} < Medium <= {cutoffs[2]:.2f} < High")
    print(f"  Class counts:\n{df[LABEL_COL].value_counts().reindex(CLASS_ORDER).to_string()}")

    genre_cols = [f"genre_{g}" for g in top_genres]
    keep = ["tconst"] + NUMERIC_FEATURES + ["is_franchise"] + genre_cols + [LABEL_COL]
    labeled = df[keep].dropna().reset_index(drop=True)
    labeled[LABEL_COL] = labeled[LABEL_COL].astype(str)
    return labeled


def split_train_test(labeled):
    train, test = train_test_split(labeled, test_size=TEST_SIZE, stratify=labeled[LABEL_COL],
                                   random_state=RANDOM_STATE)
    # The two sets MUST be disjoint -- verify it rather than assume it.
    overlap = set(train["tconst"]) & set(test["tconst"])
    assert len(overlap) == 0, f"Train/test overlap found: {len(overlap)} rows"
    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows | Overlap: {len(overlap)} (must be 0)")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _build_and_save_splits(paths):
    labeled = build_labeled_dataset()
    train, test = split_train_test(labeled)
    labeled.to_csv(paths["labeled_data"], index=False)
    train.to_csv(paths["train_set"], index=False)
    test.to_csv(paths["test_set"], index=False)

    genre_cols = [c for c in labeled.columns if c.startswith("genre_") and c != "genre_list"]
    show_cols = ["tconst"] + NUMERIC_FEATURES + ["is_franchise"] + genre_cols[:3] + [LABEL_COL]
    save_dataframe_image(labeled.sample(8, random_state=1), f"{SAMPLES_DIR}/labeled-data-sample.png",
                          "Labeled data (sample of columns shown)", columns=show_cols)
    save_dataframe_image(train.sample(6, random_state=1), f"{SAMPLES_DIR}/train-set-sample.png",
                          f"Training set ({len(train):,} rows)", columns=show_cols, n_rows=6)
    save_dataframe_image(test.sample(6, random_state=1), f"{SAMPLES_DIR}/test-set-sample.png",
                          f"Testing set ({len(test):,} rows)", columns=show_cols, n_rows=6)
    return train, test


def load_or_build_splits():
    paths = {n: os.path.join(DATA_DIR, f"{n}.csv") for n in ["labeled_data", "train_set", "test_set"]}
    train = test = None
    if all(os.path.exists(p) for p in paths.values()):
        train = pd.read_csv(paths["train_set"])
        test = pd.read_csv(paths["test_set"])
        # Validate the saved files: every model input must be numeric. If an older/bad version of the
        # files is found (for example one containing a text column), rebuild instead of trusting it.
        inputs = [c for c in train.columns if c not in ("tconst", LABEL_COL)]
        non_numeric = [c for c in inputs if not pd.api.types.is_numeric_dtype(train[c])]
        if non_numeric or LABEL_COL not in train.columns:
            print(f"=== Saved split has non-numeric input column(s) {non_numeric} -> rebuilding it ===")
            train = test = None
        else:
            print("=== Loading existing labeled data and train/test split ===")
            assert len(set(train["tconst"]) & set(test["tconst"])) == 0
    if train is None:
        train, test = _build_and_save_splits(paths)

    feature_cols = [c for c in train.columns if c not in ("tconst", LABEL_COL)]
    return train, test, feature_cols

# ===================== END SHARED DATA PREP =====================

sns.set_theme(style="whitegrid", font_scale=0.95)
GOLD, SLATE, RED = "#C9A05C", "#9CA3AF", "#A8442E"


def plot_confusion(cm, title, path):
    """Confusion matrix annotated with counts and row percentages (recall per class)."""
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.0f}%)" for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])
    plt.figure(figsize=(6.2, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="flare", cbar=False,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    train, test, feature_cols = load_or_build_splits()
    X_train, y_train = train[feature_cols], train[LABEL_COL]
    X_test, y_test = test[feature_cols], test[LABEL_COL]
    results = {}

    # ---- class balance in train vs test ----
    dist = pd.DataFrame({"Training": y_train.value_counts(normalize=True),
                         "Testing": y_test.value_counts(normalize=True)}).reindex(CLASS_ORDER) * 100
    dist.plot(kind="bar", color=[GOLD, SLATE], figsize=(6.5, 4.5))
    plt.title("Class Balance: Training vs. Testing Set")
    plt.ylabel("% of films")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/nb-label-distribution.png", dpi=160)
    plt.close()

    baseline = y_test.value_counts(normalize=True).max()
    results["baseline_majority_class_accuracy"] = round(float(baseline), 4)
    print(f"\nBaseline (always guess the most common class): {baseline:.4f}")

    # ---- what each class looks like (helps explain NB's learned probabilities) ----
    scaler = MinMaxScaler().fit(X_train)          # fit on TRAINING data only (no leakage)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=feature_cols)
    means = X_train_s.groupby(y_train.values).mean().loc[CLASS_ORDER].T
    plt.figure(figsize=(6.5, 7))
    sns.heatmap(means, annot=True, fmt=".2f", cmap="flare", cbar_kws={"label": "Mean (scaled 0-1)"})
    plt.title("Average Feature Value by Rating Class (training set)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/nb-class-feature-means.png", dpi=160)
    plt.close()

    # ---- 1) MULTINOMIAL NB (required) ----
    print("\n=== Multinomial Naive Bayes ===")
    X_test_s = scaler.transform(X_test)
    mnb = MultinomialNB(alpha=1.0).fit(scaler.transform(X_train), y_train)
    pred = mnb.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
    print(f"Accuracy: {acc:.4f}")
    print(cm)
    print(classification_report(y_test, pred, labels=CLASS_ORDER, digits=3))
    plot_confusion(cm, f"Multinomial NB  (accuracy = {acc:.3f})", f"{CHARTS_DIR}/nb-confusion-matrix.png")
    results["multinomial"] = {"accuracy": round(float(acc), 4), "confusion_matrix": cm.tolist(),
                              "report": classification_report(y_test, pred, labels=CLASS_ORDER,
                                                              output_dict=True)}

    # ---- 2) BERNOULLI NB: every feature becomes 0/1 ----
    print("\n=== Bernoulli Naive Bayes ===")
    medians = X_train[NUMERIC_FEATURES].median()          # thresholds from TRAINING data only
    def binarize(X):
        B = X.copy()
        for c in NUMERIC_FEATURES:
            B[c] = (X[c] > medians[c]).astype(int)
        return B
    bnb = BernoulliNB(alpha=1.0).fit(binarize(X_train), y_train)
    pred_b = bnb.predict(binarize(X_test))
    acc_b = accuracy_score(y_test, pred_b)
    cm_b = confusion_matrix(y_test, pred_b, labels=CLASS_ORDER)
    print(f"Accuracy: {acc_b:.4f}")
    print(cm_b)
    plot_confusion(cm_b, f"Bernoulli NB  (accuracy = {acc_b:.3f})",
                   f"{CHARTS_DIR}/nb-confusion-matrix-bernoulli.png")
    results["bernoulli"] = {"accuracy": round(float(acc_b), 4), "confusion_matrix": cm_b.tolist()}

    # ---- 3) GAUSSIAN NB: raw continuous features ----
    print("\n=== Gaussian Naive Bayes (extra comparison) ===")
    gnb = GaussianNB().fit(X_train, y_train)
    pred_g = gnb.predict(X_test)
    acc_g = accuracy_score(y_test, pred_g)
    print(f"Accuracy: {acc_g:.4f}")
    results["gaussian"] = {"accuracy": round(float(acc_g), 4),
                           "confusion_matrix": confusion_matrix(y_test, pred_g, labels=CLASS_ORDER).tolist()}

    # ---- model comparison chart ----
    names = ["Always guess\nmost common", "Multinomial\nNB", "Bernoulli\nNB", "Gaussian\nNB"]
    accs = [baseline, acc, acc_b, acc_g]
    plt.figure(figsize=(7, 4.8))
    bars = plt.bar(names, accs, color=[SLATE, GOLD, GOLD, GOLD])
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.3f}", ha="center")
    plt.ylim(0, max(accs) + 0.1)
    plt.ylabel("Test-set accuracy")
    plt.title("Naive Bayes Variants vs. Baseline")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/nb-model-comparison.png", dpi=160)
    plt.close()

    with open(f"{DATA_DIR}/nb_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Charts -> output/charts/, metrics -> output/data/nb_metrics.json")


if __name__ == "__main__":
    main()
