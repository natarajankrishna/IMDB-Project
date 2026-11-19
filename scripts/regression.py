"""
Regression -- Module 4 (linear regression + logistic regression)
==================================================================
Uses the SAME training and testing films as Part 3 (Naive Bayes / Decision Trees), so results
can be compared directly.

  LINEAR regression    target: averageRating (numeric, 1-10)         -> predicts a rating
  LOGISTIC regression  target: is_high (1 = "High" rating tier, 0 = not)  -> predicts a probability/class
  (extra) multinomial logistic regression on the 3 tiers, for a like-for-like comparison with Part 3

Run order:  imdb_pipeline.py -> naive_bayes.py (creates the Part 3 train/test files) -> regression.py

HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib seaborn
    python regression.py

OUTPUT: output/data/regression_data.csv  (full data with a 'split' column: train / test)
        output/data/regression_metrics.json, reg_linear_coefficients.csv, reg_logistic_coefficients.csv
        output/samples/regression-data-sample.png, regression-train-sample.png, regression-test-sample.png
        output/charts/reg-overview-linear.png, reg-overview-sigmoid.png, reg-pred-vs-actual.png,
                      reg-linear-coefficients.png, reg-confusion-matrix-logistic.png,
                      reg-logistic-probabilities.png, reg-logistic-odds-ratios.png,
                      reg-confusion-matrix-3class.png, reg-model-comparison.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
                             confusion_matrix, classification_report, roc_auc_score)

DATA_DIR = "output/data"
SAMPLES_DIR = "output/samples"
CHARTS_DIR = "output/charts"
for d in [DATA_DIR, SAMPLES_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

LABEL_COL = "rating_class"
CLASS_ORDER = ["Low", "Medium", "High"]
NUMERIC_FEATURES = ["numVotes_log", "runtimeMinutes", "startYear", "num_genres", "region_count"]
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", font_scale=0.95)
GOLD, SLATE, RED = "#C9A05C", "#9CA3AF", "#A8442E"


def save_dataframe_image(df, path, title, columns=None, n_rows=8):
    sample = (df if columns is None else df[columns]).head(n_rows)
    fig, ax = plt.subplots(figsize=(min(16, 1.3 * len(sample.columns)), 0.5 * n_rows + 1))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=12)
    tbl = ax.table(cellText=sample.astype(str).values, colLabels=sample.columns, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved image] {path}")


# ============================================================
# STEP 1 -- LOAD THE PART 3 SPLIT AND ADD THE TWO REGRESSION TARGETS
# ============================================================
def load_data():
    print("=== STEP 1: Loading the Part 3 train/test split ===")
    try:
        train = pd.read_csv(os.path.join(DATA_DIR, "train_set.csv"))
        test = pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"))
        clean = pd.read_csv(os.path.join(DATA_DIR, "clean_movies.csv"), usecols=["tconst", "averageRating"])
    except FileNotFoundError as e:
        raise SystemExit(f"Missing file: {e.filename}\nRun imdb_pipeline.py and naive_bayes.py first "
                         "(they create the train/test files this script reuses).")
    bad = [c for c in train.columns if c not in ("tconst", LABEL_COL) and not pd.api.types.is_numeric_dtype(train[c])]
    if bad:
        raise SystemExit(f"train_set.csv has non-numeric column(s) {bad}. Re-run naive_bayes.py (it rebuilds the split).")

    train = train.merge(clean, on="tconst", how="left")
    test = test.merge(clean, on="tconst", how="left")
    for df in (train, test):
        df["is_high"] = (df[LABEL_COL] == "High").astype(int)        # binary target for logistic regression
    assert len(set(train["tconst"]) & set(test["tconst"])) == 0, "train/test overlap!"
    assert train["averageRating"].notna().all() and test["averageRating"].notna().all()
    print(f"  Train: {len(train):,} films | Test: {len(test):,} films | Overlap: 0 (verified)")

    feature_cols = [c for c in train.columns if c not in ("tconst", LABEL_COL, "averageRating", "is_high")]
    print(f"  {len(feature_cols)} input features (the rating itself is NOT an input)")

    # save one combined file with a 'split' column, plus small sample images
    both = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    both = both[["tconst", "split", "averageRating", "is_high", LABEL_COL] + feature_cols]
    both["averageRating"] = both["averageRating"].round(1)
    both.to_csv(os.path.join(DATA_DIR, "regression_data.csv"), index=False)
    show = ["tconst", "averageRating", "is_high"] + NUMERIC_FEATURES + ["is_franchise"] + \
           [c for c in feature_cols if c.startswith("genre_")][:3]
    save_dataframe_image(both.sample(8, random_state=1), f"{SAMPLES_DIR}/regression-data-sample.png",
                          "Regression data: numeric target (averageRating), binary target (is_high), inputs",
                          columns=show + ["split"])
    save_dataframe_image(train.assign(averageRating=train["averageRating"].round(1)).sample(6, random_state=1),
                          f"{SAMPLES_DIR}/regression-train-sample.png", f"Training set ({len(train):,} films)",
                          columns=show, n_rows=6)
    save_dataframe_image(test.assign(averageRating=test["averageRating"].round(1)).sample(6, random_state=1),
                          f"{SAMPLES_DIR}/regression-test-sample.png", f"Testing set ({len(test):,} films)",
                          columns=show, n_rows=6)
    return train, test, feature_cols


# ============================================================
# STEP 2 -- OVERVIEW ILLUSTRATIONS (toy data, not the movie data)
# ============================================================
def overview_illustrations():
    print("\n=== STEP 2: Overview illustrations ===")
    rng = np.random.default_rng(3)

    # (1) linear regression: best-fit line and the residuals it minimizes
    x = rng.uniform(0, 10, 40)
    y = 2 + 0.6 * x + rng.normal(0, 1.1, 40)
    b1, b0 = np.polyfit(x, y, 1)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color=SLATE, s=35, zorder=3, label="Observed values")
    xs = np.linspace(0, 10, 50)
    plt.plot(xs, b0 + b1 * xs, color=GOLD, lw=3, label=f"Best-fit line:  y = {b0:.1f} + {b1:.2f}x", zorder=2)
    for xi, yi in zip(x, y):
        plt.plot([xi, xi], [yi, b0 + b1 * xi], color=RED, lw=1, alpha=0.6, zorder=1)
    plt.plot([], [], color=RED, lw=1, label="Residuals (errors that least squares minimizes)")
    plt.xlabel("Feature (x)")
    plt.ylabel("Numeric target (y)")
    plt.title("Linear Regression: fit the line that minimizes squared errors")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/reg-overview-linear.png", dpi=160)
    plt.close()

    # (2) sigmoid: linear score -> probability -> class; and why a straight line fails on 0/1 labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    z = np.linspace(-7, 7, 300)
    p = 1 / (1 + np.exp(-z))
    ax = axes[0]
    ax.plot(z, p, color=GOLD, lw=3)
    ax.axhline(0.5, color=SLATE, ls="--")
    ax.axvline(0, color=SLATE, ls="--")
    for zi in (-4, -2, 0, 2, 4):
        pi = 1 / (1 + np.exp(-zi))
        ax.scatter([zi], [pi], color=RED, zorder=3)
        ax.annotate(f"z={zi}\np={pi:.2f}", (zi, pi), textcoords="offset points",
                    xytext=(8, -26 if zi >= 0 else 10), fontsize=9)
    ax.set_xlabel("Linear score  z = b0 + b1x1 + b2x2 + ...   (the log-odds)")
    ax.set_ylabel("Probability of the positive class")
    ax.set_title("Sigmoid: p = 1 / (1 + e^(-z)).  Class = 1 when p >= 0.5")
    ax.text(-6.8, 0.56, "predict class 0 below this line,\nclass 1 above it", fontsize=9, color="gray")

    ax = axes[1]
    xx = rng.uniform(-6, 6, 60)
    yy = (rng.uniform(0, 1, 60) < 1 / (1 + np.exp(-1.3 * xx))).astype(int)
    b1, b0 = np.polyfit(xx, yy, 1)
    ax.scatter(xx, yy, color=SLATE, s=30, label="Observed 0 / 1 labels")
    ax.plot(z, b0 + b1 * z, color=RED, lw=2, label="Straight line (linear regression)")
    ax.plot(z, p, color=GOLD, lw=3, label="Sigmoid (logistic regression)")
    ax.axhspan(-0.6, 0, color="red", alpha=0.07)
    ax.axhspan(1, 1.6, color="red", alpha=0.07)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Predicted value")
    ax.set_title("A straight line escapes [0, 1]; the sigmoid cannot")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/reg-overview-sigmoid.png", dpi=160)
    plt.close()


def plot_confusion(cm, labels, title, path):
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.0f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    plt.figure(figsize=(6, 4.8))
    sns.heatmap(cm, annot=annot, fmt="", cmap="flare", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ============================================================
# STEP 3 -- LINEAR REGRESSION (numeric target: averageRating)
# ============================================================
def run_linear(train, test, feature_cols, results):
    print("\n=== STEP 3: Linear regression (target = averageRating) ===")
    X_train, y_train = train[feature_cols], train["averageRating"]
    X_test, y_test = test[feature_cols], test["averageRating"]

    lin = LinearRegression().fit(X_train, y_train)
    pred = lin.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    r2_train = float(r2_score(y_train, lin.predict(X_train)))
    base_rmse = float(np.sqrt(np.mean((y_test - y_train.mean()) ** 2)))   # "always predict the average rating"
    base_mae = float(np.mean(np.abs(y_test - y_train.mean())))
    print(f"  Test RMSE = {rmse:.4f} | MAE = {mae:.4f} | R^2 = {r2:.4f}   (train R^2 = {r2_train:.4f})")
    print(f"  Baseline (always predict the training mean): RMSE = {base_rmse:.4f} | MAE = {base_mae:.4f}")

    # coefficients in natural units + standardized (coefficient x feature's std) for comparing sizes
    coefs = pd.DataFrame({"feature": feature_cols, "coefficient": lin.coef_,
                          "std_of_feature": X_train.std().values})
    coefs["standardized_coef"] = coefs["coefficient"] * coefs["std_of_feature"]
    coefs = coefs.reindex(coefs["standardized_coef"].abs().sort_values(ascending=False).index)
    coefs.round(5).to_csv(f"{DATA_DIR}/reg_linear_coefficients.csv", index=False)
    print(f"  Intercept = {lin.intercept_:.4f}")
    print(coefs.round(4).head(10).to_string(index=False))

    c = dict(zip(coefs["feature"], coefs["coefficient"]))
    plain = {"per_doubling_of_votes": c["numVotes_log"] * np.log(2),
             "per_extra_10_minutes": c["runtimeMinutes"] * 10,
             "per_extra_10_years": c["startYear"] * 10}
    print("  Plain-language effects on predicted rating (rating points):")
    for k, v in plain.items():
        print(f"     {k}: {v:+.3f}")

    # predicted vs actual
    plt.figure(figsize=(6.6, 6))
    plt.hexbin(y_test, pred, gridsize=45, cmap="flare", mincnt=1)
    lims = [1, 10]
    plt.plot(lims, lims, color=GOLD, lw=2.5, label="Perfect prediction (y = x)")
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Actual rating")
    plt.ylabel("Predicted rating")
    plt.title(f"Linear Regression: Predicted vs. Actual   (R$^2$ = {r2:.3f}, RMSE = {rmse:.2f})")
    plt.colorbar(label="Number of test films")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/reg-pred-vs-actual.png", dpi=160)
    plt.close()

    # coefficient chart (standardized so bars are comparable)
    top = coefs.head(12).iloc[::-1]
    plt.figure(figsize=(7.5, 5.5))
    plt.barh(top["feature"], top["standardized_coef"], color=[GOLD if v >= 0 else RED for v in top["standardized_coef"]])
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Change in predicted rating for a 1-standard-deviation increase (rating points)")
    plt.title("Linear Regression: Largest Coefficients (standardized)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/reg-linear-coefficients.png", dpi=160)
    plt.close()

    results["linear"] = {"test_rmse": round(rmse, 4), "test_mae": round(mae, 4), "test_r2": round(r2, 4),
                         "train_r2": round(r2_train, 4), "baseline_rmse": round(base_rmse, 4),
                         "baseline_mae": round(base_mae, 4), "intercept": round(float(lin.intercept_), 4),
                         "plain_language_effects": {k: round(float(v), 4) for k, v in plain.items()}}


# ============================================================
# STEP 4 -- LOGISTIC REGRESSION (binary target: is_high)
# ============================================================
def run_logistic(train, test, feature_cols, results):
    print("\n=== STEP 4: Logistic regression (target = is_high: High tier vs. not High) ===")
    num_cols = NUMERIC_FEATURES
    scaler = StandardScaler().fit(train[num_cols])          # fit on TRAINING data only

    def prep(df):
        X = df[feature_cols].copy()
        X[num_cols] = scaler.transform(df[num_cols])        # numeric -> z-scores; 0/1 flags left as they are
        return X

    X_train, X_test = prep(train), prep(test)
    y_train, y_test = train["is_high"], test["is_high"]
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"  Share of High films in test set: {y_test.mean():.4f} -> baseline (always say 'Not High') = {baseline:.4f}")

    logit = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    proba = logit.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    rep = classification_report(y_test, pred, target_names=["Not High", "High"], output_dict=True)
    print(f"  Accuracy = {acc:.4f} | ROC AUC = {auc:.4f}")
    print(cm)
    print(classification_report(y_test, pred, target_names=["Not High", "High"], digits=3))
    plot_confusion(cm, ["Not High", "High"], f"Logistic Regression (accuracy = {acc:.3f})",
                   f"{CHARTS_DIR}/reg-confusion-matrix-logistic.png")

    # how a probability becomes a class
    plt.figure(figsize=(7.5, 4.8))
    plt.hist(proba[y_test == 0], bins=40, alpha=0.7, color=SLATE, label="Actually Not High")
    plt.hist(proba[y_test == 1], bins=40, alpha=0.7, color=GOLD, label="Actually High")
    plt.axvline(0.5, color=RED, ls="--", lw=2, label="Threshold = 0.5")
    plt.xlabel("Predicted probability of 'High'")
    plt.ylabel("Number of test films")
    plt.title("From Probability to Class: Predicted Probabilities by Actual Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/reg-logistic-probabilities.png", dpi=160)
    plt.close()

    # coefficients -> odds ratios
    coefs = pd.DataFrame({"feature": feature_cols, "coefficient": logit.coef_[0]})
    coefs["odds_ratio"] = np.exp(coefs["coefficient"])
    coefs = coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index)
    coefs.round(5).to_csv(f"{DATA_DIR}/reg_logistic_coefficients.csv", index=False)
    print(coefs.round(4).head(10).to_string(index=False))

    raw = {n: logit.coef_[0][feature_cols.index(n)] / scaler.scale_[num_cols.index(n)] for n in num_cols}
    odds_plain = {"odds_multiplier_per_doubling_of_votes": float(np.exp(raw["numVotes_log"] * np.log(2))),
                  "odds_multiplier_per_extra_10_minutes": float(np.exp(raw["runtimeMinutes"] * 10)),
                  "odds_multiplier_per_extra_10_years": float(np.exp(raw["startYear"] * 10))}
    print("  Plain-language effects on the ODDS of being rated High:")
    for k, v in odds_plain.items():
        print(f"     {k}: x{v:.3f}")

    top = coefs.head(12).iloc[::-1]
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(top["odds_ratio"], top["feature"], color=GOLD, s=70, zorder=3)
    plt.axvline(1, color=RED, ls="--", label="Odds ratio = 1 (no effect)")
    plt.xscale("log")
    plt.xlabel("Odds ratio (log scale): per +1 SD for numeric features, or 'has this genre / flag'")
    plt.title("Logistic Regression: Largest Effects on the Odds of a High Rating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/reg-logistic-odds-ratios.png", dpi=160)
    plt.close()

    # same binary task, Naive Bayes (Gaussian) for a like-for-like comparison
    gnb = GaussianNB().fit(train[feature_cols], y_train)
    nb_proba = gnb.predict_proba(test[feature_cols])[:, 1]
    nb_pred = (nb_proba >= 0.5).astype(int)
    nb_acc, nb_auc = accuracy_score(y_test, nb_pred), roc_auc_score(y_test, nb_proba)
    nb_recall = float(((nb_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum())
    print(f"  Gaussian Naive Bayes on the same binary task: accuracy = {nb_acc:.4f} | AUC = {nb_auc:.4f} | recall(High) = {nb_recall:.4f}")

    results["logistic_binary"] = {"accuracy": round(float(acc), 4), "roc_auc": round(float(auc), 4),
                                  "baseline_accuracy": round(float(baseline), 4),
                                  "confusion_matrix": cm.tolist(), "report": rep,
                                  "odds_plain_language": {k: round(v, 4) for k, v in odds_plain.items()}}
    results["naive_bayes_binary"] = {"accuracy": round(float(nb_acc), 4), "roc_auc": round(float(nb_auc), 4),
                                     "recall_high": round(nb_recall, 4)}


# ============================================================
# STEP 5 -- 3-CLASS LOGISTIC REGRESSION vs. PART 3 NAIVE BAYES
# ============================================================
def run_three_class(train, test, feature_cols, results):
    print("\n=== STEP 5: Multinomial logistic regression on the 3 tiers (compare with Part 3) ===")
    num_cols = NUMERIC_FEATURES
    scaler = StandardScaler().fit(train[num_cols])

    def prep(df):
        X = df[feature_cols].copy()
        X[num_cols] = scaler.transform(df[num_cols])
        return X

    model = LogisticRegression(max_iter=3000).fit(prep(train), train[LABEL_COL])
    pred = model.predict(prep(test))
    acc = accuracy_score(test[LABEL_COL], pred)
    cm = confusion_matrix(test[LABEL_COL], pred, labels=CLASS_ORDER)
    baseline = test[LABEL_COL].value_counts(normalize=True).max()
    print(f"  3-class accuracy = {acc:.4f}   (baseline {baseline:.4f})")
    print(cm)
    plot_confusion(cm, CLASS_ORDER, f"Logistic Regression, 3 tiers (accuracy = {acc:.3f})",
                   f"{CHARTS_DIR}/reg-confusion-matrix-3class.png")

    # pull Part 3 Naive Bayes results if naive_bayes.py saved them
    names, vals = ["Always guess\nmost common", "Logistic\nregression"], [baseline, acc]
    nb_path = os.path.join(DATA_DIR, "nb_metrics.json")
    nb_part3 = {}
    if os.path.exists(nb_path):
        nb = json.load(open(nb_path))
        for key, label in [("multinomial", "Multinomial\nNB"), ("bernoulli", "Bernoulli\nNB"), ("gaussian", "Gaussian\nNB")]:
            if key in nb:
                names.append(label)
                vals.append(nb[key]["accuracy"])
                nb_part3[key] = nb[key]["accuracy"]
    else:
        print("  (nb_metrics.json not found -- run naive_bayes.py to include the Part 3 bars)")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, vals, color=[SLATE, RED] + [GOLD] * (len(vals) - 2))
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center")
    plt.ylim(0, max(vals) + 0.1)
    plt.ylabel("Test-set accuracy (3 rating tiers)")
    plt.title("Logistic Regression vs. Part 3 Naive Bayes (same test films)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/reg-model-comparison.png", dpi=160)
    plt.close()

    results["logistic_3class"] = {"accuracy": round(float(acc), 4), "baseline_accuracy": round(float(baseline), 4),
                                  "confusion_matrix": cm.tolist(), "part3_naive_bayes_accuracy": nb_part3}


def main():
    train, test, feature_cols = load_data()
    overview_illustrations()
    results = {}
    run_linear(train, test, feature_cols, results)
    run_logistic(train, test, feature_cols, results)
    run_three_class(train, test, feature_cols, results)
    with open(f"{DATA_DIR}/regression_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Charts -> output/charts/, samples -> output/samples/, metrics -> output/data/regression_metrics.json")


if __name__ == "__main__":
    main()
