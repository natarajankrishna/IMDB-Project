This folder holds exactly one file: unsupervised_features.csv

This is the small, derived, numeric-only dataset used by both the Clustering and PCA tabs —
6 numeric columns (averageRating, numVotes_log, runtimeMinutes, startYear, num_genres,
region_count) plus a tconst identifier and an era label, for roughly 30,000 movies.

It is produced by running scripts/clustering.py (see output/data/unsupervised_features.csv in
that script's output). Copy that file here, replacing this README, and both clustering.html and
pca.html will link to it correctly via the relative path "data/unsupervised_features.csv".

This file is safe to publish: it contains only derived numeric features, not IMDb's original
text/title data, which keeps it consistent with IMDb's non-commercial data license.

Do NOT put the larger raw_merged.csv, clean_movies.csv, or any .tsv.gz files here — those are
too large for GitHub and are not needed by the website.
