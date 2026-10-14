from pathlib import Path

import pandas as pd


def ingest_data(books_path: str | Path, ratings_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load book metadata and ratings from CSV files."""
    books_file = Path(books_path)
    ratings_file = Path(ratings_path)
    # Fail before reading either file so a typo in a configured path is explicit.
    for path in (books_file, ratings_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required input CSV was not found: {path}")

    # Book-Crossing exports may contain accented names, hence the Latin-1 decoder.
    books = pd.read_csv(books_file, encoding="latin-1", low_memory=False)
    ratings = pd.read_csv(ratings_file, encoding="latin-1", low_memory=False)
    return books, ratings
