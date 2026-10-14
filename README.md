# Books Recommender System

An end-to-end machine-learning project for book discovery. It includes a modular
training pipeline, item-item collaborative filtering with cosine nearest
neighbors, a Streamlit web app, and Docker support. A small sample dataset is
included so the project works without downloading external data.

## Architecture

```text
Books CSV + Ratings CSV
        |
  Data ingestion
        |
  Data validation
        |
Data transformation ----> sparse user-item matrix (Pandas, NumPy, SciPy)
        |
   Model trainer --------> artifacts/recommender.joblib
                                  |
                           Recommendation engine
                           /                 \
                 Streamlit app          Poster metadata
                                      (image URL in books CSV)
```

The recommender learns book-to-book similarity from readers' ratings. The app
offers popular books, similar-book lookup, and recommendations for a selected
reader. The image URLs are taken from the book metadata, so no external poster
API is required.

## Run locally

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m books_recommender.pipeline
streamlit run app.py
```

The Streamlit app also trains the sample-data model on first launch if the
artifact is missing.

## Use Book-Crossing or your own CSV data

Provide a books CSV with these columns:

`ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-M`

Provide a ratings CSV with:

`User-ID, ISBN, Book-Rating`

Ratings should be numeric from 0 through 10. Zero ratings (commonly used to
represent implicit/unrated records in Book-Crossing) are excluded from model
training. Invalid rows are filtered during validation; a missing required
column or empty usable dataset is reported as an error.

Train with explicit input paths:

```powershell
python -m books_recommender.pipeline --books data\Books.csv --ratings data\Ratings.csv --artifact artifacts\recommender.joblib
```

Point the app to the same files with environment variables:

```powershell
$env:BOOKS_CSV = "data\Books.csv"
$env:RATINGS_CSV = "data\Ratings.csv"
$env:MODEL_ARTIFACT = "artifacts\recommender.joblib"
streamlit run app.py
```

## Docker

Build and run the demo app:

```powershell
docker compose up --build
```

Open <http://localhost:8501>. Compose mounts `data` read-only and persists
trained artifacts in the local `artifacts` folder.

## Tests

```powershell
pytest
```
