import pandas as pd


BOOK_COLUMNS = {
    "ISBN",
    "Book-Title",
    "Book-Author",
    "Year-Of-Publication",
    "Publisher",
    "Image-URL-M",
}
RATING_COLUMNS = {"User-ID", "ISBN", "Book-Rating"}


class DataValidationError(ValueError):
    """Raised when an input dataset cannot be used by the pipeline."""


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    # Missing schema fields cannot be safely inferred, so report them rather than silently guessing.
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(
            f"{name} CSV is missing required columns: {', '.join(sorted(missing))}"
        )


def validate_data(
    books: pd.DataFrame, ratings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check schemas and remove unusable records from both input tables."""
    _require_columns(books, BOOK_COLUMNS, "Books")
    _require_columns(ratings, RATING_COLUMNS, "Ratings")
    if books.empty:
        raise DataValidationError("Books CSV contains no rows.")
    if ratings.empty:
        raise DataValidationError("Ratings CSV contains no rows.")

    books = books.copy()
    ratings = ratings.copy()
    # Normalize identifiers before joining the two tables in the transformation step.
    for frame in (books, ratings):
        frame["ISBN"] = frame["ISBN"].astype("string").str.strip()

    # Rows without a usable book identity or title cannot be displayed or indexed.
    books["Book-Title"] = books["Book-Title"].astype("string").str.strip()
    books = books.dropna(subset=["ISBN", "Book-Title"])
    books = books[(books["ISBN"] != "") & (books["Book-Title"] != "")]

    ratings["User-ID"] = ratings["User-ID"].astype("string").str.strip()
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    # Coerce malformed scores to missing, then discard them and any out-of-range values.
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    ratings = ratings[(ratings["User-ID"] != "") & (ratings["ISBN"] != "")]
    ratings = ratings[ratings["Book-Rating"].between(0, 10)]

    if books.empty:
        raise DataValidationError("No books have a usable ISBN and title.")
    if ratings.empty:
        raise DataValidationError("No ratings have a valid user, ISBN, and score from 0 to 10.")
    return books, ratings
