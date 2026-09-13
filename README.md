# IMDb Rating Prediction — Website

A 4-tab static website (Introduction, DataPrep/EDA, Model/Method, Conclusions) for a data science
project predicting IMDb ratings using data from the IMDb developer API.

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

## How to get a shareable URL (GitHub Pages — free)

1. **Create a repo.** Go to github.com → New repository → name it e.g. `imdb-rating-project` →
   keep it Public → Create repository.
2. **Upload these files.** On the repo page, click "Add file" → "Upload files", drag in
   everything from this folder (keep the `css` and `assets` folders intact), then commit.
3. **Turn on Pages.** In the repo, go to Settings → Pages (left sidebar). Under "Build and
   deployment", set Source to "Deploy from a branch", branch = `main`, folder = `/ (root)`. Save.
4. **Get your URL.** After a minute, refresh that same Settings → Pages screen — it will show a
   live URL like:
   `https://yourusername.github.io/imdb-rating-project/`
5. **Share it.** Paste that URL into your Google Doc. Anyone with the doc can click it and land
   on the Introduction tab, then navigate the other three tabs from the nav bar.

## Updating it monthly

Each page has clearly marked placeholder boxes (dashed borders, labeled "Add this month") where
you should:
- Replace placeholder text with real writeups.
- Drop chart images into `assets/` and update the `<img src="assets/...">` paths (broken image
  references are hidden automatically until you add the file).
- Replace the example code blocks with your actual code snippets.
- Fill in the `—` placeholders in tables with real metrics.

To publish an update: edit the files (in GitHub's web editor, or locally and `git push`), commit,
and GitHub Pages redeploys automatically within a minute or two — the same URL always shows the
latest version, so you never need to re-share a new link.

## Local preview before publishing

Open `index.html` directly in a browser, or from a terminal in this folder run:
```
python3 -m http.server 8000
```
then visit `http://localhost:8000` to click through all four tabs before pushing changes live.
