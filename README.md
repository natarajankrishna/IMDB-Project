# IMDb Rating Prediction — Website

A 4-tab static website (Introduction, DataPrep/EDA, Model/Method, Conclusions) for predicting IMDb ratings using data from the IMDb developer API.

## Files

```
imdb-site/
├── index.html          Introduction
├── eda.html             DataPrep / EDA
├── clustering.html      Clustering
├── pca.html              PCA
├── conclusions.html     Conclusions
├── css/style.css        Shared styling
├── scripts/             Python scripts (imdb_pipeline.py, clustering.py, pca.py)
├── data/                 One small derived numeric CSV used by Clustering & PCA tabs
└── assets/              Chart images / screenshots — see assets/README.txt for the full list
```

## Local preview before publishing

Open `index.html` directly in a browser, or from a terminal in this folder run:
```
python3 -m http.server 8000
```
then visit `http://localhost:8000` to click through all four tabs before pushing changes live.
