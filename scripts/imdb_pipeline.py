"""
IMDb Rating Study — Data Collection & EDA Pipeline
====================================================

WHAT THIS SCRIPT DOES (maps directly to the DataPrep/EDA tab requirements):
  1. Downloads raw IMDb Non-Commercial Dataset files (free, official, no key needed)
  2. Builds a RAW merged dataset and saves a small image of it (pre-cleaning)
  3. Cleans the data and saves a small image of the CLEANED dataset
  4. Enriches a sample of titles via the OMDb API (free tier, GET endpoint) —
     this is the "API used" requirement, with a real endpoint + GET example
  5. Pulls supporting signals from title.akas (international release breadth)
     and title.crew/name.basics (director info)
  6. Produces 13 EDA visualizations, each tied to one of the project's
     research questions, saved as individual PNG files
  7. Produces summary statistics (describe()) as a CSV

HOW TO RUN:
  1. Install dependencies:
       pip install pandas numpy matplotlib seaborn requests
  2. Get a free OMDb API key: https://www.omdbapi.com/apikey.aspx
     (free tier = 1,000 requests/day)
  3. Set it below in CONFIG, or as an environment variable OMDB_API_KEY
  4. Run:
       python imdb_pipeline.py
  5. Everything is written to ./output/  — raw/clean sample images go in
     output/samples/, all charts go in output/charts/, data files go in
     output/data/

NOTES ON SCALE:
  title.basics.tsv.gz is large (millions of rows across all title types,
  not just movies), so it's read in CHUNKS and filtered down as it's read,
  rather than loaded fully into memory. title.akas and name.basics are
  handled the same way, since we only need rows tied to titles we already
  kept. Expect the first run to take several minutes purely for the
  IMDb downloads and the chunked read of title.basics.
"""

import os
import re
import io
import gc
import json
import time
import gzip
import shutil
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG — adjust these before running
# ============================================================

OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "PUT_YOUR_FREE_OMDB_KEY_HERE")
OMDB_ENDPOINT = "https://www.omdbapi.com/"          # core OMDb endpoint
IMDB_DATASETS_BASE = "https://datasets.imdbws.com"  # official IMDb dataset host

MIN_VOTES = 100          # drop obscure titles with very few votes
MIN_YEAR = 1970          # ignore very old/sparse-era titles for cleaner trend lines
OMDB_SAMPLE_SIZE = 3000  # how many titles to enrich via the API (stays under 1,000/day cap if split across days)
STREAMING_CUTOFF_YEAR = 2015  # used for Q10 (theatrical era vs streaming era)

# --- MEMORY / KERNEL-STABILITY SETTINGS ---
# If your notebook kernel died on a previous run, this is almost always a RAM issue —
# title.basics, title.akas, and name.basics each contain tens of millions of rows.
# SAMPLE_LIMIT caps the CLEANED dataset size (via stratified sampling across decades)
# BEFORE the expensive full-file scans (akas region counts, director lookups, OMDb
# enrichment) run — those scans are the most likely place a low-RAM kernel dies,
# since they check every row of a huge file against your kept title IDs.
# Set SAMPLE_LIMIT = None to disable sampling and use the full cleaned dataset
# (only recommended if you have 16GB+ RAM and are running as a plain script, not
# inside a notebook).
SAMPLE_LIMIT = 30_000
CHUNK_SIZE = 100_000     # smaller = less peak memory per chunk, more iterations

RAW_DIR = "output/raw_downloads"
DATA_DIR = "output/data"
SAMPLES_DIR = "output/samples"
CHARTS_DIR = "output/charts"

for d in [RAW_DIR, DATA_DIR, SAMPLES_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.9)
FIGSIZE = (8, 5)


# ============================================================
# STEP 1 — DOWNLOAD RAW IMDB DATASETS
# ============================================================
# Source website: https://developer.imdb.com/non-commercial-datasets/
# Direct file host: https://datasets.imdbws.com/
# License: free for personal / non-commercial use, refreshed daily by IMDb

IMDB_FILES = [
    "title.basics.tsv.gz",
    "title.ratings.tsv.gz",
    "title.akas.tsv.gz",
    "title.crew.tsv.gz",
    "name.basics.tsv.gz",
]


def download_imdb_datasets():
    print("\n=== STEP 1: Downloading IMDb Non-Commercial Datasets ===")
    for fname in IMDB_FILES:
        local_path = os.path.join(RAW_DIR, fname)
        if os.path.exists(local_path):
            print(f"  [skip] {fname} already downloaded")
            continue
        url = f"{IMDB_DATASETS_BASE}/{fname}"
        print(f"  Downloading {url} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        print(f"  [done] saved to {local_path}")


# ============================================================
# STEP 2 — LOAD + FILTER title.basics IN CHUNKS, MERGE WITH RATINGS
# ============================================================

def load_and_filter_basics():
    """title.basics contains ALL title types (movies, tv series, episodes,
    shorts, etc) — tens of millions of rows. We only want feature-length
    movies from MIN_YEAR onward, filtered chunk-by-chunk so the full file
    is never held in memory at once, and each chunk is discarded (gc'd)
    as soon as it's filtered down."""
    print("\n=== STEP 2: Loading + filtering title.basics.tsv.gz ===")
    path = os.path.join(RAW_DIR, "title.basics.tsv.gz")
    usecols = ["tconst", "titleType", "primaryTitle", "isAdult",
               "startYear", "runtimeMinutes", "genres"]
    kept_chunks = []
    reader = pd.read_csv(path, sep="\t", usecols=usecols, dtype=str,
                          na_values="\\N", chunksize=CHUNK_SIZE,
                          compression="gzip", quoting=3)
    for i, chunk in enumerate(reader):
        movies = chunk[(chunk["titleType"] == "movie") & (chunk["isAdult"] == "0")].copy()
        # filter by year here too (early), so old/obscure titles never make it
        # into the concatenated frame at all — this is the single biggest
        # memory saver in this step
        movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce")
        movies = movies[movies["startYear"] >= MIN_YEAR]
        if len(movies) > 0:
            kept_chunks.append(movies)
        del chunk
        if i % 20 == 0:
            gc.collect()
            print(f"  processed chunk {i+1} (running total kept: "
                  f"{sum(len(c) for c in kept_chunks):,} rows)")
    basics = pd.concat(kept_chunks, ignore_index=True)
    del kept_chunks
    gc.collect()
    print(f"  Total movie titles kept from title.basics: {len(basics):,}")
    return basics


def load_ratings():
    print("\n  Loading title.ratings.tsv.gz (small file, loaded fully)")
    path = os.path.join(RAW_DIR, "title.ratings.tsv.gz")
    ratings = pd.read_csv(path, sep="\t", dtype=str, na_values="\\N",
                           compression="gzip")
    ratings["averageRating"] = ratings["averageRating"].astype(float)
    ratings["numVotes"] = ratings["numVotes"].astype(int)
    return ratings


def build_raw_merged_dataset():
    cache_path = os.path.join(DATA_DIR, "raw_merged.csv")
    if os.path.exists(cache_path):
        print(f"\n=== STEP 2: Loading cached raw_merged.csv (skipping re-download/re-scan) ===")
        return pd.read_csv(cache_path)
    basics = load_and_filter_basics()
    ratings = load_ratings()
    raw_df = basics.merge(ratings, on="tconst", how="inner")
    del basics, ratings
    gc.collect()
    print(f"  Merged basics + ratings: {len(raw_df):,} movie rows "
          f"(this is the RAW dataset, before cleaning)")
    raw_df.to_csv(cache_path, index=False)
    return raw_df


# ============================================================
# STEP 3 — SAVE A SMALL IMAGE OF THE RAW DATA (as required)
# ============================================================

def save_dataframe_image(df, path, title, n_rows=8):
    """Renders a small sample of a dataframe as a PNG image — NOT the full
    dataset, just enough rows/columns for the reader to see its shape."""
    sample = df.head(n_rows)
    fig, ax = plt.subplots(figsize=(min(16, 1.4 * len(sample.columns)), 0.5 * n_rows + 1))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=12)
    tbl = ax.table(cellText=sample.astype(str).values,
                    colLabels=sample.columns,
                    cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved image] {path}")


# ============================================================
# STEP 4 — CLEAN THE DATA
# ============================================================

FRANCHISE_PATTERN = re.compile(
    r"(\bpart\s*\d+\b|\bchapter\s*\d+\b|\b(ii|iii|iv|v|vi|vii|viii)\b|\b\d+\s*$)",
    flags=re.IGNORECASE,
)


def clean_data(raw_df):
    cache_path = os.path.join(DATA_DIR, "clean_movies.csv")
    if os.path.exists(cache_path):
        print("\n=== STEP 4: Loading cached clean_movies.csv (skipping re-clean) ===")
        df = pd.read_csv(cache_path)
        df["genre_list"] = df["genres"].str.split(",")
        return df

    print("\n=== STEP 4: Cleaning data ===")
    df = raw_df.copy()

    # type casts — startYear/runtime arrive as strings with IMDb's "\N" null marker
    df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")

    # drop rows missing essentials for our analysis
    before = len(df)
    df = df.dropna(subset=["startYear", "runtimeMinutes", "genres"])
    df = df[df["numVotes"] >= MIN_VOTES]
    df = df[df["startYear"] >= MIN_YEAR]
    df = df[(df["runtimeMinutes"] >= 40) & (df["runtimeMinutes"] <= 300)]  # drop shorts/junk outliers
    df = df.drop_duplicates(subset="tconst")
    print(f"  Rows before cleaning: {before:,} -> after cleaning: {len(df):,}")

    df["startYear"] = df["startYear"].astype(int)

    # genre list + genre count (used for Q9)
    df["genre_list"] = df["genres"].str.split(",")
    df["num_genres"] = df["genre_list"].apply(len)

    # franchise heuristic flag (used for Q6) — approximate, see module docstring
    df["is_franchise_heuristic"] = df["primaryTitle"].str.contains(FRANCHISE_PATTERN, na=False)

    # era flag (used for Q10)
    df["era"] = np.where(df["startYear"] >= STREAMING_CUTOFF_YEAR, "Streaming era", "Theatrical era")

    # decade (used for Q2 line chart)
    df["decade"] = (df["startYear"] // 10) * 10

    # --- SAMPLE DOWN HERE, before the expensive full-file scans that follow ---
    # (akas region counts, director lookups, OMDb enrichment). This is the main
    # fix for kernel crashes: everything after this point operates on a much
    # smaller, but still temporally-representative, set of titles.
    if SAMPLE_LIMIT is not None and len(df) > SAMPLE_LIMIT:
        print(f"  Sampling down from {len(df):,} to {SAMPLE_LIMIT:,} rows "
              f"(stratified across decades) to keep the rest of the pipeline "
              f"memory-safe. Set SAMPLE_LIMIT = None in CONFIG to disable this.")
        per_decade = max(1, SAMPLE_LIMIT // df["decade"].nunique())
        df = (df.groupby("decade", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_decade), random_state=42)))
        df = df.reset_index(drop=True)
        print(f"  Final sampled dataset size: {len(df):,} rows")

    df.to_csv(os.path.join(DATA_DIR, "clean_movies.csv"), index=False)
    return df


def genre_exploded(df):
    """One row per (title, genre) pair — needed for genre-level boxplots."""
    return df.explode("genre_list").rename(columns={"genre_list": "genre"})


# ============================================================
# STEP 5 — SUPPORTING PULLS: title.akas (Q8) + title.crew/name.basics (bonus)
# ============================================================

def load_akas_region_counts(tconst_set):
    """Counts how many distinct regional title variants (akas) each film has —
    a proxy for how widely it was internationally released. Used for Q8."""
    print("\n=== STEP 5a: Loading title.akas.tsv.gz (international release breadth) ===")
    cache_path = os.path.join(DATA_DIR, "akas_region_counts.csv")
    if os.path.exists(cache_path):
        print("\n=== STEP 5a: Loading cached akas_region_counts.csv ===")
        return pd.read_csv(cache_path)

    print("\n=== STEP 5a: Loading title.akas.tsv.gz (international release breadth) ===")
    path = os.path.join(RAW_DIR, "title.akas.tsv.gz")
    counts = {}
    reader = pd.read_csv(path, sep="\t", usecols=["titleId", "region"], dtype=str,
                          na_values="\\N", chunksize=CHUNK_SIZE, compression="gzip", quoting=3)
    for i, chunk in enumerate(reader):
        chunk = chunk[chunk["titleId"].isin(tconst_set) & chunk["region"].notna()]
        for tconst, group in chunk.groupby("titleId"):
            counts[tconst] = counts.get(tconst, 0) + group["region"].nunique()
        del chunk
        if i % 20 == 0:
            gc.collect()
            print(f"  processed akas chunk {i+1}")
    result = pd.DataFrame({"tconst": list(counts.keys()), "region_count": list(counts.values())})
    result.to_csv(cache_path, index=False)
    return result


def load_director_info(tconst_set):
    """Pulls the primary director's name per title, for the bonus
    'director track record' visualization."""
    cache_path = os.path.join(DATA_DIR, "director_info.csv")
    if os.path.exists(cache_path):
        print("\n=== STEP 5b: Loading cached director_info.csv ===")
        return pd.read_csv(cache_path)

    print("\n=== STEP 5b: Loading title.crew.tsv.gz + name.basics.tsv.gz (director info) ===")
    crew_path = os.path.join(RAW_DIR, "title.crew.tsv.gz")
    # title.crew is read in chunks too — it's a full IMDb-scale file (millions of rows)
    crew_chunks = []
    reader = pd.read_csv(crew_path, sep="\t", usecols=["tconst", "directors"], dtype=str,
                          na_values="\\N", chunksize=CHUNK_SIZE, compression="gzip", quoting=3)
    for i, chunk in enumerate(reader):
        kept = chunk[chunk["tconst"].isin(tconst_set)].dropna()
        if len(kept) > 0:
            crew_chunks.append(kept)
        del chunk
        if i % 20 == 0:
            gc.collect()
    crew = pd.concat(crew_chunks, ignore_index=True) if crew_chunks else pd.DataFrame(columns=["tconst", "directors"])
    del crew_chunks
    crew["director_nconst"] = crew["directors"].str.split(",").str[0]  # first-listed director

    needed_nconsts = set(crew["director_nconst"].unique())
    names_path = os.path.join(RAW_DIR, "name.basics.tsv.gz")
    name_chunks = []
    reader = pd.read_csv(names_path, sep="\t", usecols=["nconst", "primaryName"], dtype=str,
                          na_values="\\N", chunksize=CHUNK_SIZE, compression="gzip", quoting=3)
    for i, chunk in enumerate(reader):
        kept = chunk[chunk["nconst"].isin(needed_nconsts)]
        if len(kept) > 0:
            name_chunks.append(kept)
        del chunk
        if i % 20 == 0:
            gc.collect()
    names = pd.concat(name_chunks, ignore_index=True) if name_chunks else pd.DataFrame(columns=["nconst", "primaryName"])
    del name_chunks
    gc.collect()

    result = crew.merge(names, left_on="director_nconst", right_on="nconst", how="left")
    result = result[["tconst", "primaryName"]].rename(columns={"primaryName": "director"})
    result.to_csv(cache_path, index=False)
    return result


# ============================================================
# STEP 6 — ENRICH A SAMPLE VIA THE OMDb API (satisfies the "API used" requirement)
# ============================================================
# Website: https://www.omdbapi.com/
# Core endpoint: https://www.omdbapi.com/
# Example GET request (lookup by IMDb ID, which is exactly what tconst is):
#   https://www.omdbapi.com/?i=tt0111161&apikey=YOUR_KEY
# Example GET request (lookup by title):
#   https://www.omdbapi.com/?t=Inception&y=2010&apikey=YOUR_KEY

def enrich_with_omdb(clean_df, sample_size=OMDB_SAMPLE_SIZE):
    print("\n=== STEP 6: Enriching a sample via the OMDb API ===")
    cache_path = os.path.join(DATA_DIR, "omdb_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
        print(f"  Loaded existing cache with {len(cache)} entries")

    # stratified sample across decades so enrichment isn't skewed toward one era
    sample = (clean_df.groupby("decade", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), max(1, sample_size // clean_df["decade"].nunique())),
                                         random_state=42)))
    sample = sample.head(sample_size)

    records = []
    for i, tconst in enumerate(sample["tconst"]):
        if tconst in cache:
            records.append(cache[tconst])
            continue
        try:
            resp = requests.get(OMDB_ENDPOINT, params={"i": tconst, "apikey": OMDB_API_KEY}, timeout=10)
            data = resp.json()
            if data.get("Response") == "True":
                record = {
                    "tconst": tconst,
                    "Country": data.get("Country"),
                    "Language": data.get("Language"),
                    "Rated": data.get("Rated"),
                    "Released": data.get("Released"),
                    "BoxOffice": data.get("BoxOffice"),
                    "Awards": data.get("Awards"),
                }
            else:
                record = {"tconst": tconst, "Country": None, "Language": None,
                          "Rated": None, "Released": None, "BoxOffice": None, "Awards": None}
            cache[tconst] = record
            records.append(record)
        except Exception as e:
            print(f"  [warn] request failed for {tconst}: {e}")

        if i % 50 == 0:
            print(f"  OMDb progress: {i}/{len(sample)}")
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        time.sleep(0.1)  # gentle pacing, well within free-tier limits

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    omdb_df = pd.DataFrame(records)
    merged = sample.merge(omdb_df, on="tconst", how="left")

    # parse first listed country/language and month from Released date
    merged["primary_country"] = merged["Country"].str.split(",").str[0]
    merged["primary_language"] = merged["Language"].str.split(",").str[0]
    merged["release_month"] = pd.to_datetime(merged["Released"], errors="coerce").dt.month_name()

    merged.to_csv(os.path.join(DATA_DIR, "omdb_enriched_sample.csv"), index=False)
    print(f"  Enriched {merged['Country'].notna().sum()} / {len(merged)} sampled titles successfully")
    return merged


# ============================================================
# STEP 7 — 13 EDA VISUALIZATIONS (each tied to a research question)
# ============================================================

def chart_1_rating_distribution(df):
    plt.figure(figsize=FIGSIZE)
    sns.histplot(df["averageRating"], bins=30, kde=True, color="#C9A05C")
    plt.title("Distribution of Average IMDb Ratings")
    plt.xlabel("Average rating"); plt.ylabel("Number of films")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-ratings-distribution.png", dpi=160); plt.close()


def chart_2_genre_boxplot(genres_df):
    top_genres = genres_df["genre"].value_counts().head(10).index
    plt.figure(figsize=(9, 5.5))
    sns.boxplot(data=genres_df[genres_df["genre"].isin(top_genres)],
                x="averageRating", y="genre", palette="flare")
    plt.title("Rating by Genre (Q1) — Top 10 Most Common Genres")
    plt.xlabel("Average rating"); plt.ylabel("")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-genre-boxplot.png", dpi=160); plt.close()


def chart_3_rating_by_year(df):
    yearly = df.groupby("startYear")["averageRating"].mean()
    plt.figure(figsize=FIGSIZE)
    yearly.plot(color="#C9A05C")
    plt.title("Mean Rating by Release Year (Q2)")
    plt.xlabel("Release year"); plt.ylabel("Mean average rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-rating-by-year.png", dpi=160); plt.close()


def chart_4_runtime_vs_rating(df):
    plt.figure(figsize=FIGSIZE)
    sns.regplot(data=df.sample(min(5000, len(df)), random_state=1),
                x="runtimeMinutes", y="averageRating",
                scatter_kws={"alpha": 0.15, "s": 10, "color": "#9CA3AF"},
                line_kws={"color": "#A8442E"})
    plt.title("Runtime vs. Rating (Q3)")
    plt.xlabel("Runtime (minutes)"); plt.ylabel("Average rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-runtime-vs-rating.png", dpi=160); plt.close()


def chart_5_release_month(omdb_df):
    month_order = ["January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"]
    monthly = omdb_df.dropna(subset=["release_month"]).groupby("release_month")["averageRating"].mean()
    monthly = monthly.reindex(month_order)
    plt.figure(figsize=(9, 5))
    monthly.plot(kind="bar", color="#C9A05C")
    plt.title("Mean Rating by Release Month (Q4) — OMDb-enriched sample")
    plt.xlabel("Release month"); plt.ylabel("Mean average rating"); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-rating-by-month.png", dpi=160); plt.close()


def chart_6_votes_vs_rating(df):
    plt.figure(figsize=FIGSIZE)
    plt.scatter(df["numVotes"], df["averageRating"], alpha=0.1, s=8, color="#9CA3AF")
    plt.xscale("log")
    plt.title("Vote Count vs. Rating (Q5)")
    plt.xlabel("Number of votes (log scale)"); plt.ylabel("Average rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-votes-vs-rating.png", dpi=160); plt.close()


def chart_7_franchise_violin(df):
    plt.figure(figsize=FIGSIZE)
    sns.violinplot(data=df, x="is_franchise_heuristic", y="averageRating", palette="flare")
    plt.title("Franchise (heuristic) vs. Standalone Films — Rating (Q6)")
    plt.xlabel("Detected as franchise/sequel title"); plt.ylabel("Average rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-franchise-violin.png", dpi=160); plt.close()


def chart_8_country_bar(omdb_df):
    top_countries = omdb_df["primary_country"].value_counts().head(10).index
    subset = omdb_df[omdb_df["primary_country"].isin(top_countries)]
    order = subset.groupby("primary_country")["averageRating"].mean().sort_values(ascending=False).index
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=subset, x="averageRating", y="primary_country", order=order, palette="flare")
    plt.title("Mean Rating by Country of Origin (Q7) — OMDb-enriched sample, top 10 countries")
    plt.xlabel("Mean average rating"); plt.ylabel("")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-country-bar.png", dpi=160); plt.close()


def chart_9_international_releases(df, akas_df):
    merged = df.merge(akas_df, on="tconst", how="left")
    merged["region_count"] = merged["region_count"].fillna(0)
    plt.figure(figsize=FIGSIZE)
    sns.regplot(data=merged.sample(min(5000, len(merged)), random_state=1),
                x="region_count", y="averageRating",
                scatter_kws={"alpha": 0.15, "s": 10, "color": "#9CA3AF"},
                line_kws={"color": "#A8442E"})
    plt.title("International Release Breadth vs. Rating (Q8)")
    plt.xlabel("Number of distinct regional title releases"); plt.ylabel("Average rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-international-releases.png", dpi=160); plt.close()
    return merged


def chart_10_num_genres_bar(df):
    grouped = df.groupby("num_genres")["averageRating"].mean()
    plt.figure(figsize=FIGSIZE)
    grouped.plot(kind="bar", color="#C9A05C")
    plt.title("Number of Genres Tagged vs. Mean Rating (Q9)")
    plt.xlabel("Number of genres tagged"); plt.ylabel("Mean average rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-genre-count-bar.png", dpi=160); plt.close()


def chart_11_era_boxplot(df):
    plt.figure(figsize=FIGSIZE)
    sns.boxplot(data=df, x="era", y="averageRating", palette="flare")
    plt.title(f"Theatrical Era vs. Streaming Era (cutoff {STREAMING_CUTOFF_YEAR}) — Q10")
    plt.xlabel(""); plt.ylabel("Average rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-era-boxplot.png", dpi=160); plt.close()


def chart_12_correlation_heatmap(df):
    numeric_cols = ["averageRating", "numVotes", "runtimeMinutes", "startYear", "num_genres"]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="flare", vmin=-1, vmax=1)
    plt.title("Correlation Between Numeric Features")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-correlations.png", dpi=160); plt.close()


def chart_13_missing_data(raw_df):
    missing = raw_df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    plt.figure(figsize=FIGSIZE)
    missing.plot(kind="barh", color="#A8442E")
    plt.title("Missing Values per Column — RAW Data (pre-cleaning)")
    plt.xlabel("Number of missing values")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-missing-data.png", dpi=160); plt.close()


def chart_14_director_filmography(df, director_df):
    merged = df.merge(director_df, on="tconst", how="left").dropna(subset=["director"])
    counts = merged.groupby("director").agg(films=("tconst", "count"),
                                              mean_rating=("averageRating", "mean"))
    counts = counts[counts["films"] >= 3]  # directors with at least 3 films in the dataset
    plt.figure(figsize=FIGSIZE)
    plt.scatter(counts["films"], counts["mean_rating"], alpha=0.4, color="#9CA3AF")
    plt.title("Director Filmography Size vs. Mean Rating (bonus)")
    plt.xlabel("Number of films in dataset"); plt.ylabel("Mean average rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/eda-director-filmography.png", dpi=160); plt.close()


# ============================================================
# STEP 8 — SUMMARY STATISTICS
# ============================================================

def save_summary_statistics(df):
    print("\n=== STEP 8: Saving summary statistics ===")
    numeric_cols = ["averageRating", "numVotes", "runtimeMinutes", "startYear", "num_genres"]
    summary = df[numeric_cols].describe().T
    summary.to_csv(os.path.join(DATA_DIR, "summary_statistics.csv"))
    print(summary)


# ============================================================
# MAIN
# ============================================================

def main():
    download_imdb_datasets()

    raw_df = build_raw_merged_dataset()
    save_dataframe_image(raw_df, f"{SAMPLES_DIR}/raw_data_sample.png",
                          "RAW merged data (title.basics + title.ratings) — before cleaning")

    clean_df = clean_data(raw_df)
    save_dataframe_image(clean_df.drop(columns=["genre_list"]), f"{SAMPLES_DIR}/clean_data_sample.png",
                          "CLEANED data — types cast, nulls dropped, features engineered")

    genres_df = genre_exploded(clean_df)
    tconst_set = set(clean_df["tconst"])

    akas_df = load_akas_region_counts(tconst_set)
    director_df = load_director_info(tconst_set)
    omdb_df = enrich_with_omdb(clean_df)

    print("\n=== STEP 7: Generating 13 EDA visualizations ===")
    chart_1_rating_distribution(clean_df)
    chart_2_genre_boxplot(genres_df)
    chart_3_rating_by_year(clean_df)
    chart_4_runtime_vs_rating(clean_df)
    chart_5_release_month(omdb_df)
    chart_6_votes_vs_rating(clean_df)
    chart_7_franchise_violin(clean_df)
    chart_8_country_bar(omdb_df)
    chart_9_international_releases(clean_df, akas_df)
    chart_10_num_genres_bar(clean_df)
    chart_11_era_boxplot(clean_df)
    chart_12_correlation_heatmap(clean_df)
    chart_13_missing_data(raw_df)
    chart_14_director_filmography(clean_df, director_df)

    save_summary_statistics(clean_df)

    print("\n=== DONE ===")
    print(f"Charts saved to: {CHARTS_DIR}/")
    print(f"Data samples saved to: {SAMPLES_DIR}/")
    print(f"Full data files saved to: {DATA_DIR}/")


if __name__ == "__main__":
    main()
