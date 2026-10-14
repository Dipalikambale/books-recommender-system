import pandas as pd
import pytest

from books_recommender.data_transformation import transform_data
from books_recommender.data_validation import DataValidationError, validate_data
from books_recommender.model_trainer import train_model
from books_recommender.recommendation_engine import RecommendationEngine


def sample_frames():
    # Tiny in-memory dataset keeps the model test fast and independent of external files.
    books = pd.DataFrame(
        {
            "ISBN": ["a", "b", "c"],
            "Book-Title": ["Alpha", "Beta", "Gamma"],
            "Book-Author": ["A", "B", "C"],
            "Year-Of-Publication": [2001, 2002, 2003],
            "Publisher": ["P", "P", "Q"],
            "Image-URL-M": ["", "", ""],
        }
    )
    ratings = pd.DataFrame(
        {
            "User-ID": ["1", "1", "2", "2", "3", "3"],
            "ISBN": ["a", "b", "a", "b", "b", "c"],
            "Book-Rating": [9, 8, 8, 9, 7, 10],
        }
    )
    return books, ratings


def test_validate_rejects_missing_columns():
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_data(pd.DataFrame({"ISBN": ["a"]}), pd.DataFrame())


def test_pipeline_trains_and_recommends(tmp_path):
    books, ratings = sample_frames()
    books, ratings = validate_data(books, ratings)
    metadata, interactions, matrix, item_ids, user_ids = transform_data(books, ratings)
    artifact_path = tmp_path / "model.joblib"

    # Exercise artifact creation and inference together, as the app does at runtime.
    bundle = train_model(metadata, interactions, matrix, item_ids, user_ids, artifact_path)
    engine = RecommendationEngine(bundle)

    assert artifact_path.is_file()
    assert engine.similar_books("a", 2).iloc[0]["isbn"] == "b"
    assert engine.recommend_for_user("1")["isbn"].tolist() == ["c"]
