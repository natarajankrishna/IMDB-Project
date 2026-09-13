# IMDb Rating Prediction — Website

A 4-tab static website (Introduction, DataPrep/EDA, Model/Method, Conclusions) for predicting IMDb ratings using data from the IMDb developer API.

## Files

```
imdb-site/
├── index.html          Introduction (home page)
├── eda.html             DataPrep / EDA
├── model.html           Model / Method
├── conclusions.html     Conclusions
├── css/style.css        Shared styling
└── assets/              Put chart images / screenshots here
```

## Local preview before publishing

Open `index.html` directly in a browser, or from a terminal in this folder run:
```
python3 -m http.server 8000
```
then visit `http://localhost:8000` to click through all four tabs before pushing changes live.
