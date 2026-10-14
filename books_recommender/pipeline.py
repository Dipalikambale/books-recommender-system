from pathlib import Path

from books_recommender.data_ingestion import ingest_data
from books_recommender.data_transformation import transform_data
from books_recommender.data_validation import validate_data
from books_recommender.model_trainer import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOKS = PROJECT_ROOT / "data" / "sample_books.csv"
DEFAULT_RATINGS = PROJECT_ROOT / "data" / "sample_ratings.csv"
DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "recommender.joblib"


def run_pipeline(
    books_path: str | Path = DEFAULT_BOOKS,
    ratings_path: str | Path = DEFAULT_RATINGS,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> dict:
    """Run ingestion, validation, transformation, and model training."""
    # Keep each lifecycle stage explicit so the steps can be tested and replaced independently.
    books, ratings = ingest_data(books_path, ratings_path)
    books, ratings = validate_data(books, ratings)
    metadata, interactions, user_item, item_ids, user_ids = transform_data(books, ratings)
    return train_model(metadata, interactions, user_item, item_ids, user_ids, artifact_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the book recommendation model.")
    parser.add_argument("--books", type=Path, default=DEFAULT_BOOKS, help="Path to books CSV.")
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS, help="Path to ratings CSV.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Output artifact path.")
    args = parser.parse_args()

    bundle = run_pipeline(args.books, args.ratings, args.artifact)
    print(
        f"Training complete: {len(bundle['item_ids'])} books, "
        f"{len(bundle['user_ids'])} users. Artifact: {args.artifact}"
    )
