"""
AdaBoost -- Module 4
=====================
Predicts a film's rating tier (Low / Medium / High) with AdaBoost built from decision stumps, and compares it
with a single decision tree. Uses the SAME training and testing films as Part 3 so the comparison is fair.

Tuning is done on a VALIDATION set carved out of the training data, so the testing set is used exactly once,
at the very end:
      training set (80% of films)  ->  inner-train (80%)  +  validation (20%)     <- tune here
      testing set  (20% of films)                                                  <- score once, here

Run order:  imdb_pipeline.py -> naive_bayes.py (creates the Part 3 train/test files) -> adaboost.py

HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib seaborn
    python adaboost.py          (takes a few minutes: 4 learning rates x 300 boosting rounds)

OUTPUT: output/charts/ada-overview-rounds.png, ada-overview-alpha.png, ada-overview-bagging-vs-boosting.png,
                       ada-validation-curves.png, ada-estimator-weights.png, ada-confusion-matrices.png,
                       ada-feature-importance.png, ada-model-comparison.png
        output/samples/ada-onehot-step.png
        output/data/ada_metrics.json, ada_first_stumps.txt
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Harmless message from some scikit-learn versions (AdaBoost hands plain arrays to its stumps); hide it.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

DATA_DIR = "output/data"
SAMPLES_DIR = "output/samples"
CHARTS_DIR = "output/charts"
for d in [DATA_DIR, SAMPLES_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

LABEL_COL = "rating_class"
CLASS_ORDER = ["Low", "Medium", "High"]
RANDOM_STATE = 42
N_MAX = 300                                  # maximum boosting rounds tried
LEARNING_RATES = [0.05, 0.1, 0.5, 1.0]       # learning rates tried

sns.set_theme(style="whitegrid", font_scale=0.95)
GOLD, SLATE, RED = "#C9A05C", "#9CA3AF", "#A8442E"


def make_ada(**kw):
    """AdaBoost on decision STUMPS (trees with one split). algorithm='SAMME' is requested explicitly on
    older scikit-learn versions so that each stump gets a weight (alpha); newer versions only have SAMME."""
    stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    try:
        return AdaBoostClassifier(estimator=stump, algorithm="SAMME", random_state=RANDOM_STATE, **kw)
    except TypeError:
        return AdaBoostClassifier(estimator=stump, random_state=RANDOM_STATE, **kw)


def plot_confusion_ax(ax, cm, title):
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.0f}%)" for j in range(3)] for i in range(3)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="flare", cbar=False, ax=ax, xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")


# ============================================================
# STEP 1 -- LOAD THE PART 3 SPLIT (already numeric: genres are one-hot encoded)
# ============================================================
def load_data():
    print("=== STEP 1: Loading the Part 3 train/test split ===")
    try:
        train = pd.read_csv(os.path.join(DATA_DIR, "train_set.csv"))
        test = pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"))
    except FileNotFoundError as e:
        raise SystemExit(f"Missing file: {e.filename}\nRun imdb_pipeline.py and naive_bayes.py first.")
    feature_cols = [c for c in train.columns if c not in ("tconst", LABEL_COL)]
    bad = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]
    if bad:
        raise SystemExit(f"Non-numeric input column(s) {bad}. Re-run naive_bayes.py (it rebuilds the split).")
    assert len(set(train["tconst"]) & set(test["tconst"])) == 0
    print(f"  Train: {len(train):,} | Test: {len(test):,} | Overlap: 0 | Inputs: {len(feature_cols)} numeric features")
    return train, test, feature_cols


# ============================================================
# STEP 2 -- SHOW THE ONE-HOT ENCODING STEP (categorical -> numeric)
# ============================================================
def onehot_step_image(train, feature_cols):
    """Side-by-side: the original categorical columns, and the 0/1 numeric columns they became."""
    print("\n=== STEP 2: One-hot encoding illustration ===")
    path = os.path.join(DATA_DIR, "clean_movies.csv")
    if not os.path.exists(path):
        print("  (clean_movies.csv not found, skipping this image)")
        return
    raw = pd.read_csv(path, usecols=["tconst", "genres", "is_franchise_heuristic"])
    # 4 random films plus 1 franchise film, so both kinds of conversion (genre text, True/False) are visible
    franchise = train[train["is_franchise"] == 1]
    pick = train.drop(franchise.index).sample(4, random_state=3)["tconst"].tolist()
    if len(franchise):
        pick += franchise.sample(1, random_state=3)["tconst"].tolist()
    raw = raw.set_index("tconst").loc[pick].reset_index()
    enc_cols = ["tconst"] + [c for c in feature_cols if c.startswith("genre_")] + ["is_franchise"]
    enc = train.set_index("tconst").loc[pick].reset_index()[enc_cols]
    enc.columns = [c.replace("genre_", "") if c.startswith("genre_") else c for c in enc.columns]

    fig, axes = plt.subplots(2, 1, figsize=(15, 5.4), gridspec_kw={"height_ratios": [1, 1]})
    for ax, df, title in [(axes[0], raw, "BEFORE: categorical columns (text and True/False)"),
                          (axes[1], enc, "AFTER: one-hot encoded -> every column is a number (1 = present, 0 = absent)")]:
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        tbl = ax.table(cellText=df.astype(str).values, colLabels=df.columns, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(f"{SAMPLES_DIR}/ada-onehot-step.png", dpi=170, bbox_inches="tight")
    plt.close()
    print(f"  [saved image] {SAMPLES_DIR}/ada-onehot-step.png")


# ============================================================
# STEP 3 -- OVERVIEW ILLUSTRATIONS (toy data, not the movie data)
# ============================================================
def overview_rounds_image():
    """A hand-built AdaBoost on toy data: marker size = row weight."""
    X, y = make_classification(n_samples=40, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, class_sep=0.9, flip_y=0.05, random_state=7)
    y = np.where(y == 1, 1, -1)
    n = len(y)
    w = np.ones(n) / n
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 0.8, X[:, 0].max() + 0.8, 200),
                         np.linspace(X[:, 1].min() - 0.8, X[:, 1].max() + 0.8, 200))
    grid = np.c_[gx.ravel(), gy.ravel()]
    stumps, alphas = [], []
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for t in range(3):
        stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
        pred = stump.predict(X)
        err = float(np.clip(w[pred != y].sum(), 1e-10, 1 - 1e-10))
        alpha = 0.5 * np.log((1 - err) / err)
        ax = axes[t]
        ax.contourf(gx, gy, stump.predict(grid).reshape(gx.shape), levels=[-2, 0, 2], colors=["#e6c9c0", "#ead9b6"], alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], s=30 + 2600 * w, c=np.where(y == 1, GOLD, RED), edgecolor="black", linewidth=0.6)
        wrong = pred != y
        ax.scatter(X[wrong, 0], X[wrong, 1], s=30 + 2600 * w[wrong] + 90, facecolors="none", edgecolors="black", linewidth=2)
        ttl = "all rows weighted equally" if t == 0 else "rows the last stump missed are now larger"
        ax.set_title(f"Round {t + 1}: {ttl}\nerror = {err:.2f},  alpha = {alpha:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        stumps.append(stump); alphas.append(alpha)
        w = w * np.exp(-alpha * y * pred)       # raise weight of mistakes, lower weight of correct rows
        w = w / w.sum()
    ax = axes[3]
    score = sum(a * s.predict(grid) for a, s in zip(alphas, stumps)).reshape(gx.shape)
    ax.contourf(gx, gy, score, levels=[-100, 0, 100], colors=["#e6c9c0", "#ead9b6"], alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], s=60, c=np.where(y == 1, GOLD, RED), edgecolor="black", linewidth=0.6)
    final_pred = np.sign(sum(a * s.predict(X) for a, s in zip(alphas, stumps)))
    ax.set_title(f"Final: weighted vote of the 3 stumps\n(accuracy on these rows = {np.mean(final_pred == y):.0%})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("AdaBoost on toy data  (marker size = row weight; circled = misclassified by that round's stump)", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/ada-overview-rounds.png", dpi=150, bbox_inches="tight")
    plt.close()


def overview_alpha_image():
    e = np.linspace(0.01, 0.99, 300)
    alpha = 0.5 * np.log((1 - e) / e)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(e, alpha, color=GOLD, lw=3)
    ax.axhline(0, color=SLATE, ls="--")
    ax.axvline(0.5, color=SLATE, ls="--")
    for ev in (0.1, 0.2, 0.5, 0.8):
        av = 0.5 * np.log((1 - ev) / ev)
        ax.scatter([ev], [av], color=RED, zorder=3)
        ax.annotate(f"error {ev:.1f}\nalpha {av:.2f}", (ev, av), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Weighted error rate of the stump")
    ax.set_ylabel("Stump weight  alpha = 1/2 ln((1 - error) / error)")
    ax.set_title("Accurate stumps get a large say; coin-flip stumps get none")
    ax = axes[1]
    ax.plot(e, np.sqrt((1 - e) / e), color=RED, lw=3, label="Rows it got WRONG  (x e^alpha)")
    ax.plot(e, np.sqrt(e / (1 - e)), color=SLATE, lw=3, label="Rows it got RIGHT  (x e^-alpha)")
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.scatter([0.2, 0.2], [2, 0.5], color="black", zorder=3)
    ax.annotate("error 0.2: wrong rows x2,\ncorrect rows x0.5", (0.2, 2), textcoords="offset points", xytext=(12, 10), fontsize=9)
    ax.set_xlabel("Weighted error rate of the stump")
    ax.set_ylabel("Weight multiplier (before re-normalizing)")
    ax.set_title("How the row weights change after each round")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/ada-overview-alpha.png", dpi=160)
    plt.close()


def overview_bagging_vs_boosting():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    def box(ax, xy, text, w=2.6, h=0.9, color="#f3ead7"):
        ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle="round,pad=0.05",
                                    fc=color, ec="#555", lw=1.2))
        ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=9)

    def arrow(ax, a, b, label=None):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))
        if label:
            ax.text((a[0] + b[0]) / 2 + 0.15, (a[1] + b[1]) / 2, label, fontsize=8, color=RED, va="center")

    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("BAGGING: trees built independently, in parallel", fontsize=11, fontweight="bold")
    box(ax, (5, 9), "Training data")
    for i, x in enumerate((1.8, 5, 8.2), start=1):
        box(ax, (x, 6.5), f"Random resample {i}", w=2.5)
        box(ax, (x, 4.2), f"Full tree {i}", w=2.5, color="#e9dcc3")
        arrow(ax, (5, 8.55), (x, 6.95))
        arrow(ax, (x, 6.05), (x, 4.65))
        arrow(ax, (x, 3.75), (5, 1.95))
    box(ax, (5, 1.5), "Average / majority vote\n(every tree counts equally)", w=4.4, h=1.0, color="#ead9b6")
    ax.text(5, 0.2, "Goal: reduce variance", ha="center", fontsize=9, style="italic")

    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("BOOSTING (AdaBoost): stumps built one after another", fontsize=11, fontweight="bold")
    box(ax, (5, 9.0), "Training data, all rows weighted equally", w=5.4)
    xs = (1.7, 5.0, 8.3)
    for i, x in enumerate(xs, start=1):
        box(ax, (x, 5.6), f"Stump {i}\n(one split)", w=2.3, color="#e9dcc3")
    arrow(ax, (3.2, 8.55), (1.9, 6.1))
    for a_x, b_x in ((xs[0] + 1.2, xs[1] - 1.2), (xs[1] + 1.2, xs[2] - 1.2)):
        arrow(ax, (a_x, 5.6), (b_x, 5.6))
        ax.text((a_x + b_x) / 2, 6.55, "raise weight\nof its mistakes", ha="center", fontsize=8, color=RED)
    box(ax, (5, 1.5), "Weighted vote\n(accurate stumps get a larger say, alpha)", w=5.8, h=1.0, color="#ead9b6")
    for x in xs:
        arrow(ax, (x, 5.1), (5 + (x - 5) * 0.35, 2.05))
    ax.text(5, 0.2, "Goal: reduce bias (many weak rules make a strong one)", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/ada-overview-bagging-vs-boosting.png", dpi=160)
    plt.close()


# ============================================================
# STEP 4 -- TUNE ON A VALIDATION SET
# ============================================================
def tune(X_tr, y_tr, X_val, y_val):
    print("\n=== STEP 4: Tuning n_estimators and learning_rate on the validation set ===")
    curves, models = {}, {}
    for lr in LEARNING_RATES:
        t0 = time.time()
        ada = make_ada(n_estimators=N_MAX, learning_rate=lr).fit(X_tr, y_tr)
        curves[lr] = np.array([accuracy_score(y_val, p) for p in ada.staged_predict(X_val)])
        models[lr] = ada
        n_best = int(np.argmax(curves[lr])) + 1
        print(f"  learning_rate={lr:<5} best validation accuracy {curves[lr].max():.4f} at {n_best} rounds "
              f"(rounds actually built: {len(curves[lr])}, {time.time() - t0:.0f}s)")
    best_lr = max(curves, key=lambda k: curves[k].max())
    best_n = int(np.argmax(curves[best_lr])) + 1
    print(f"  BEST: learning_rate = {best_lr}, n_estimators = {best_n} (validation accuracy {curves[best_lr].max():.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("flare", len(LEARNING_RATES))
    ax = axes[0]
    for color, lr in zip(palette, LEARNING_RATES):
        ax.plot(np.arange(1, len(curves[lr]) + 1), curves[lr], color=color, lw=2, label=f"learning rate = {lr}")
    ax.scatter([best_n], [curves[best_lr].max()], color="black", zorder=3, label=f"best: {best_n} rounds, lr {best_lr}")
    ax.set_xlabel("Boosting rounds (number of stumps)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy as rounds are added")
    ax.legend(fontsize=9)
    ax = axes[1]
    tr_curve = np.array([accuracy_score(y_tr, p) for p in models[best_lr].staged_predict(X_tr)])
    ax.plot(np.arange(1, len(tr_curve) + 1), tr_curve, color=SLATE, lw=2, label="Inner-training accuracy")
    ax.plot(np.arange(1, len(curves[best_lr]) + 1), curves[best_lr], color=GOLD, lw=2, label="Validation accuracy")
    ax.axvline(best_n, color=RED, ls="--", label=f"chosen: {best_n} rounds")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Training vs. validation (learning rate {best_lr})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/ada-validation-curves.png", dpi=160)
    plt.close()
    return best_lr, best_n, curves


def tune_single_tree(X_tr, y_tr, X_val, y_val):
    """The Part 3 decision tree again, but its depth is now chosen on the validation set (not the test set)."""
    accs = []
    for d in range(1, 16):
        m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(X_tr, y_tr)
        accs.append(accuracy_score(y_val, m.predict(X_val)))
    best_d = int(np.argmax(accs)) + 1
    print(f"  Decision tree: best depth on validation = {best_d} (validation accuracy {max(accs):.4f})")
    return best_d, max(accs)


# ============================================================
# MAIN
# ============================================================
def main():
    train, test, feature_cols = load_data()
    X_train, y_train = train[feature_cols], train[LABEL_COL]
    X_test, y_test = test[feature_cols], test[LABEL_COL]
    baseline = float(y_test.value_counts(normalize=True).max())
    results = {"baseline_majority_class_accuracy": round(baseline, 4)}
    print(f"  Baseline (always guess the most common class): {baseline:.4f}")

    onehot_step_image(train, feature_cols)
    print("\n=== STEP 3: Overview illustrations ===")
    overview_rounds_image(); overview_alpha_image(); overview_bagging_vs_boosting()

    # validation split carved from the TRAINING data only
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train,
                                                random_state=RANDOM_STATE)
    assert len(set(X_tr.index) & set(X_val.index)) == 0
    print(f"\n  Inner-train: {len(X_tr):,} | Validation: {len(X_val):,} | Test (untouched): {len(X_test):,}")
    results["split_sizes"] = {"inner_train": len(X_tr), "validation": len(X_val), "test": len(X_test)}

    best_lr, best_n, curves = tune(X_tr, y_tr, X_val, y_val)
    best_depth, dt_val_acc = tune_single_tree(X_tr, y_tr, X_val, y_val)

    # ---- final models: refit on the FULL training set with the chosen settings, score on test ONCE ----
    print("\n=== STEP 5: Final models, scored once on the testing set ===")
    ada = make_ada(n_estimators=best_n, learning_rate=best_lr).fit(X_train, y_train)
    pred_ada = ada.predict(X_test)
    acc_ada = accuracy_score(y_test, pred_ada)
    cm_ada = confusion_matrix(y_test, pred_ada, labels=CLASS_ORDER)
    print(f"  AdaBoost ({len(ada.estimators_)} stumps, lr={best_lr}): test accuracy = {acc_ada:.4f}")
    print(cm_ada)
    print(classification_report(y_test, pred_ada, labels=CLASS_ORDER, digits=3))

    tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
    pred_tree = tree.predict(X_test)
    acc_tree = accuracy_score(y_test, pred_tree)
    cm_tree = confusion_matrix(y_test, pred_tree, labels=CLASS_ORDER)
    print(f"  Decision tree (depth {best_depth}, chosen on validation): test accuracy = {acc_tree:.4f}")
    print(cm_tree)

    stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE).fit(X_train, y_train)
    acc_stump = accuracy_score(y_test, stump.predict(X_test))
    print(f"  A single stump (one question): test accuracy = {acc_stump:.4f}")
    print(f"  AdaBoost vs tuned decision tree: {100 * (acc_ada - acc_tree):+.2f} percentage points")

    # ---- confusion matrices ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_ax(axes[0], cm_tree, f"Decision tree, depth {best_depth}\naccuracy = {acc_tree:.3f}")
    plot_confusion_ax(axes[1], cm_ada, f"AdaBoost, {len(ada.estimators_)} stumps\naccuracy = {acc_ada:.3f}")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/ada-confusion-matrices.png", dpi=160)
    plt.close()

    # ---- stump weights (alpha) and weighted errors per round ----
    k = len(ada.estimators_)
    alphas, errs = ada.estimator_weights_[:k], ada.estimator_errors_[:k]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(np.arange(1, k + 1), alphas, color=GOLD, lw=2)
    axes[0].set_xlabel("Boosting round")
    axes[0].set_ylabel("Stump weight (alpha)")
    axes[0].set_title("How much say each stump gets in the final vote")
    axes[1].plot(np.arange(1, k + 1), errs, color=RED, lw=2)
    axes[1].axhline(2 / 3, color=SLATE, ls="--", label="2/3: with 3 classes, a stump with higher error gets zero or negative weight")
    axes[1].axhline(0.5, color=SLATE, ls=":")
    axes[1].set_xlabel("Boosting round")
    axes[1].set_ylabel("Weighted error of the stump")
    axes[1].set_title("Each new stump faces a harder, re-weighted problem")
    axes[1].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/ada-estimator-weights.png", dpi=160)
    plt.close()

    # ---- what the first stumps ask ----
    lines = []
    for i in range(min(8, k)):
        st = ada.estimators_[i]
        f, thr = feature_cols[st.tree_.feature[0]], st.tree_.threshold[0]
        left = st.classes_[int(np.argmax(st.tree_.value[1]))]
        right = st.classes_[int(np.argmax(st.tree_.value[2]))]
        lines.append(f"Round {i + 1}: is {f} <= {thr:.3f}?  yes -> {left}, no -> {right}   "
                     f"(weight alpha = {alphas[i]:.3f}, weighted error = {errs[i]:.3f})")
    open(f"{DATA_DIR}/ada_first_stumps.txt", "w").write("\n".join(lines))
    print("\n  First stumps:\n   " + "\n   ".join(lines))

    # ---- feature importance ----
    imp = pd.Series(ada.feature_importances_, index=feature_cols).sort_values().tail(10)
    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh", color=GOLD)
    plt.title("AdaBoost: Top Feature Importances")
    plt.xlabel("Importance (weighted across all stumps)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/ada-feature-importance.png", dpi=160)
    plt.close()

    # ---- comparison chart ----
    names, vals, colors = ["Always guess\nmost common", "Single stump", f"Decision tree\n(depth {best_depth})", "AdaBoost"], \
                          [baseline, acc_stump, acc_tree, acc_ada], [SLATE, SLATE, GOLD, RED]
    dt_path = os.path.join(DATA_DIR, "dt_metrics.json")
    if os.path.exists(dt_path):
        t3 = json.load(open(dt_path)).get("Tree 3: Gini, depth 5", {}).get("test_accuracy")
        if t3:
            names.insert(2, "Part 3 Tree 3\n(depth 5)")
            vals.insert(2, t3)
            colors.insert(2, GOLD)
            results["part3_tree3_test_accuracy"] = t3
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center")
    plt.ylim(0, max(vals) + 0.1)
    plt.ylabel("Test-set accuracy (3 rating tiers)")
    plt.title("AdaBoost vs. Decision Trees (same test films)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/ada-model-comparison.png", dpi=160)
    plt.close()

    results.update({
        "best_learning_rate": best_lr, "best_n_estimators": best_n,
        "validation_accuracy_best": round(float(curves[best_lr].max()), 4),
        "adaboost": {"test_accuracy": round(float(acc_ada), 4), "rounds_built": k, "confusion_matrix": cm_ada.tolist(),
                     "report": classification_report(y_test, pred_ada, labels=CLASS_ORDER, output_dict=True)},
        "decision_tree_tuned_on_validation": {"best_depth": best_depth, "validation_accuracy": round(float(dt_val_acc), 4),
                                              "test_accuracy": round(float(acc_tree), 4), "confusion_matrix": cm_tree.tolist()},
        "single_stump_test_accuracy": round(float(acc_stump), 4),
        "adaboost_minus_tree_points": round(100 * float(acc_ada - acc_tree), 2),
    })
    with open(f"{DATA_DIR}/ada_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Charts -> output/charts/, metrics -> output/data/ada_metrics.json")


if __name__ == "__main__":
    main()
