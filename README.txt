This folder holds every image used across the website.

Expected filenames — these must match EXACTLY (case-sensitive) or the <img> tags in the HTML
won't find them. Rename your pipeline/script output files to match if they differ at all.

INTRODUCTION TAB (2 images):
- intro-hero.jpg
- intro-clapperboard.jpg

DATAPREP / EDA TAB (16 images):
  Raw/clean sample images (2):
  - raw_data_sample.png
  - clean_data_sample.png
  Chart images (14):
  - eda-ratings-distribution.png
  - eda-genre-boxplot.png
  - eda-rating-by-year.png
  - eda-runtime-vs-rating.png
  - eda-rating-by-month.png
  - eda-votes-vs-rating.png
  - eda-franchise-violin.png
  - eda-country-bar.png
  - eda-international-releases.png
  - eda-genre-count-bar.png
  - eda-era-boxplot.png
  - eda-correlations.png
  - eda-missing-data.png
  - eda-director-filmography.png

CLUSTERING TAB (9 images — from scripts/clustering.py's output/charts/ and output/samples/):
  - clustering-overview-partitional.png
  - clustering-overview-hierarchical.png
  - unsupervised-data-sample.png          (shared with the PCA tab — same underlying data)
  - clustering-k-comparison.png
  - clustering-silhouette-curve.png
  - clustering-silhouette-plot.png
  - clustering-scatter-pca2d.png
  - clustering-dendrogram.png
  - clustering-hclust-scatter.png

PCA TAB (5 images — from scripts/pca.py's output/charts/; unsupervised-data-sample.png is shared,
already listed above):
  - pca-overview-eigenvectors.png
  - pca-overview-dimensionality.png
  - pca-scree-plot.png
  - pca-biplot.png
  - pca-scatter-by-era.png

How to add these images:
1. Run scripts/imdb_pipeline.py, then scripts/clustering.py, then scripts/pca.py (in that order —
   clustering.py builds the shared numeric feature set that pca.py reuses).
2. Gather every file from each script's output/samples/ and output/charts/ folders.
3. Copy them all directly into this "assets" folder — flat, no subfolders.
4. Commit and push (or re-upload via GitHub's web UI). Each page will pick up its images
   automatically once the filenames match.

Note: Git does not track empty folders. As long as at least one real file sits in this folder,
it will upload correctly.

Note on large data files: do NOT put your raw or cleaned CSV/XLSX datasets in this folder — only
images belong here. The one exception referenced by the Clustering/PCA tabs,
data/unsupervised_features.csv, is small (just 6 numeric columns) and goes in a separate top-level
"data" folder, not here — see data/README.txt.
