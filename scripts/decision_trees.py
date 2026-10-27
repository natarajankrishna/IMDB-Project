"""
Decision Trees -- Module 3
===========================
Predicts a film's rating tier (Low / Medium / High) from structural features using three
different decision trees, so their structure and accuracy can be compared:
  Tree 1: Gini impurity,  max depth 3   (small and readable)
  Tree 2: Entropy,        max depth 3   (same size, different split criterion)
  Tree 3: Gini impurity,  max depth 5, min 500 films per leaf (bigger, more detailed)

Run order:  imdb_pipeline.py  ->  decision_trees.py   (data prep is built into this file; if naive_bayes.py already ran,
the exact same train/test files are reused.)

HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib seaborn
    python decision_trees.py

OUTPUT: output/charts/dt-overview-structure.png, dt-overview-impurity.png,
        dt-tree-1-gini-depth3.png, dt-tree-2-entropy-depth3.png, dt-tree-3-gini-depth5.png,
        dt-confusion-matrices.png, dt-depth-curve.png, dt-feature-importance.png
        output/data/dt_metrics.json, dt_tree1_rules.txt, dt_worked_example.txt
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

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


# ============================================================
# IMPURITY MATH (used by the worked example on the website)
# ============================================================
def gini(counts):
    p = np.array(counts) / sum(counts)
    return 1 - np.sum(p ** 2)


def entropy(counts):
    p = np.array(counts) / sum(counts)
    p = p[p > 0]
    return float(max(0.0, -np.sum(p * np.log2(p))))   # max() avoids printing "-0.0" for pure nodes


def split_quality(parent, left, right, impurity):
    n = sum(parent)
    weighted = sum(left) / n * impurity(left) + sum(right) / n * impurity(right)
    return impurity(parent), weighted, impurity(parent) - weighted


def worked_example():
    """Small, hand-checkable example: 10 films, 6 High and 4 Low (hypothetical numbers)."""
    parent = [6, 4]                                   # [High, Low]
    splits = {"Split A: runtime > 100 min": ([4, 1], [2, 3]),
              "Split B: 2 or more genres":  ([4, 0], [2, 4])}
    lines = [f"Parent node: 6 High, 4 Low | Gini = {gini(parent):.4f} | Entropy = {entropy(parent):.4f}"]
    for name, (left, right) in splits.items():
        _, wg, gg = split_quality(parent, left, right, gini)
        _, we, ig = split_quality(parent, left, right, entropy)
        lines.append(f"{name}: left={left} right={right}")
        lines.append(f"   Gini:    left={gini(left):.4f} right={gini(right):.4f} "
                     f"weighted={wg:.4f}  Gini gain={gg:.4f}")
        lines.append(f"   Entropy: left={entropy(left):.4f} right={entropy(right):.4f} "
                     f"weighted={we:.4f}  Information gain={ig:.4f}")
    text = "\n".join(lines)
    print(text)
    with open(f"{DATA_DIR}/dt_worked_example.txt", "w") as f:
        f.write(text)


# ============================================================
# OVERVIEW ILLUSTRATIONS (toy data, not the movie data)
# ============================================================
def overview_illustrations():
    print("\n=== Overview illustrations ===")
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=3, random_state=3)
    toy = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    plt.figure(figsize=(9, 5))
    plot_tree(toy, feature_names=["feature 1", "feature 2"], class_names=["A", "B", "C"],
              filled=True, rounded=True, fontsize=10)
    plt.title("Anatomy of a Decision Tree: root, internal nodes, leaves (illustrative example)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/dt-overview-structure.png", dpi=160)
    plt.close()

    p = np.linspace(0.001, 0.999, 400)
    g = 2 * p * (1 - p)
    e = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    plt.figure(figsize=(6.5, 4.8))
    plt.plot(p, g, color=GOLD, lw=2.5, label="Gini impurity")
    plt.plot(p, e, color=RED, lw=2.5, label="Entropy")
    plt.axvline(0.5, color=SLATE, ls="--", alpha=0.6)
    plt.xlabel("Proportion of class 1 in the node")
    plt.ylabel("Impurity")
    plt.title("Impurity of a Two-Class Node (0 = pure)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/dt-overview-impurity.png", dpi=160)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    worked_example()
    overview_illustrations()

    train, test, feature_cols = load_or_build_splits()
    X_train, y_train = train[feature_cols], train[LABEL_COL]
    X_test, y_test = test[feature_cols], test[LABEL_COL]
    baseline = y_test.value_counts(normalize=True).max()
    print(f"\nBaseline (always guess the most common class): {baseline:.4f}")

    configs = [
        ("Tree 1: Gini, depth 3", dict(criterion="gini", max_depth=3), "dt-tree-1-gini-depth3.png", (16, 7)),
        ("Tree 2: Entropy, depth 3", dict(criterion="entropy", max_depth=3), "dt-tree-2-entropy-depth3.png", (16, 7)),
        ("Tree 3: Gini, depth 5", dict(criterion="gini", max_depth=5, min_samples_leaf=500),
         "dt-tree-3-gini-depth5.png", (34, 12)),
    ]
    results = {"baseline_majority_class_accuracy": round(float(baseline), 4)}
    models, cms = [], []

    for title, params, fname, figsize in configs:
        clf = DecisionTreeClassifier(random_state=RANDOM_STATE, **params).fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc_tr = accuracy_score(y_train, clf.predict(X_train))
        acc_te = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
        print(f"\n=== {title} ===  train acc={acc_tr:.4f}  test acc={acc_te:.4f}  leaves={clf.get_n_leaves()}")
        print(cm)

        plt.figure(figsize=figsize)
        plot_tree(clf, feature_names=feature_cols, class_names=list(clf.classes_), filled=True,
                  rounded=True, fontsize=9 if figsize[0] < 20 else 8, impurity=True)
        plt.title(f"{title}   (test accuracy = {acc_te:.3f})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{CHARTS_DIR}/{fname}", dpi=110 if figsize[0] > 20 else 130)
        plt.close()

        results[title] = {"train_accuracy": round(float(acc_tr), 4), "test_accuracy": round(float(acc_te), 4),
                          "leaves": int(clf.get_n_leaves()), "confusion_matrix": cm.tolist(),
                          "report": classification_report(y_test, pred, labels=CLASS_ORDER, output_dict=True)}
        models.append((title, clf))
        cms.append((title, cm, acc_te))

        if title.startswith("Tree 1"):
            with open(f"{DATA_DIR}/dt_tree1_rules.txt", "w") as f:
                f.write(export_text(clf, feature_names=feature_cols))

    # ---- confusion matrices for all three trees ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (title, cm, acc) in zip(axes, cms):
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.0f}%)" for j in range(3)] for i in range(3)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="flare", cbar=False, ax=ax,
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
        ax.set_title(f"{title}\naccuracy = {acc:.3f}")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/dt-confusion-matrices.png", dpi=150)
    plt.close()

    # ---- depth vs accuracy: where does the tree start memorizing? ----
    depths = list(range(1, 16))
    tr_acc, te_acc = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(X_train, y_train)
        tr_acc.append(accuracy_score(y_train, m.predict(X_train)))
        te_acc.append(accuracy_score(y_test, m.predict(X_test)))
    best_depth = depths[int(np.argmax(te_acc))]
    print(f"\nBest max_depth by test accuracy (gini): {best_depth} ({max(te_acc):.4f})")
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(depths, tr_acc, marker="o", color=SLATE, label="Training accuracy")
    plt.plot(depths, te_acc, marker="o", color=GOLD, label="Testing accuracy")
    plt.axvline(best_depth, color=RED, ls="--", alpha=0.7, label=f"best depth = {best_depth}")
    plt.xlabel("Maximum tree depth")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Testing Accuracy by Tree Depth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/dt-depth-curve.png", dpi=160)
    plt.close()
    results["best_depth_by_test_accuracy"] = best_depth

    # ---- feature importance of the best of the three trees ----
    best_title, best_clf = max(models, key=lambda t: accuracy_score(y_test, t[1].predict(X_test)))
    imp = pd.Series(best_clf.feature_importances_, index=feature_cols).sort_values().tail(10)
    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh", color=GOLD)
    plt.title(f"Top Feature Importances ({best_title})")
    plt.xlabel("Importance (share of total impurity reduction)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/dt-feature-importance.png", dpi=160)
    plt.close()

    with open(f"{DATA_DIR}/dt_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Charts -> output/charts/, metrics -> output/data/dt_metrics.json")


if __name__ == "__main__":
    main()
