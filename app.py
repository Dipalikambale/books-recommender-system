import os
from pathlib import Path

import joblib
import streamlit as st

from books_recommender.pipeline import DEFAULT_ARTIFACT, DEFAULT_BOOKS, DEFAULT_RATINGS, run_pipeline
from books_recommender.recommendation_engine import RecommendationEngine


st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
st.title("📚 Books Recommender")
st.caption("Discover your next read with collaborative filtering and item-item nearest neighbors.")

artifact_path = Path(os.getenv("MODEL_ARTIFACT", str(DEFAULT_ARTIFACT)))
books_path = Path(os.getenv("BOOKS_CSV", str(DEFAULT_BOOKS)))
ratings_path = Path(os.getenv("RATINGS_CSV", str(DEFAULT_RATINGS)))


@st.cache_resource
def load_engine(path: str, books: str, ratings: str) -> RecommendationEngine:
    model_path = Path(path)
    if not model_path.is_file():
        # Make the included sample useful on first launch without requiring a CLI training step.
        with st.spinner("Preparing the recommendation model from the available CSV data..."):
            run_pipeline(books, ratings, model_path)
    # Cache the model across Streamlit reruns to avoid deserializing it on every interaction.
    artifacts = joblib.load(model_path)
    return RecommendationEngine(artifacts)


try:
    engine = load_engine(str(artifact_path), str(books_path), str(ratings_path))
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Could not load or train the recommender: {error}")
    st.info("Check that BOOKS_CSV and RATINGS_CSV point to valid CSV files.")
    st.stop()


def show_recommendations(recommendations) -> None:
    if recommendations.empty:
        st.info("No recommendations are available for this selection yet.")
        return
    columns = st.columns(min(5, len(recommendations)))
    for index, (_, book) in enumerate(recommendations.iterrows()):
        # Reuse the available columns so cards wrap cleanly for smaller result sets.
        with columns[index % len(columns)]:
            if book["image_url"]:
                st.image(book["image_url"], use_container_width=True)
            st.markdown(f"**{book['title']}**")
            st.caption(book["author"] or "Author not listed")
            if book["year"]:
                st.caption(str(book["year"]))


# Separate discovery workflows while sharing one loaded recommendation engine.
tab_discover, tab_similar, tab_personal = st.tabs(
    ["Popular books", "Find similar books", "For a reader"]
)

with tab_discover:
    st.subheader("Popular among readers")
    show_recommendations(engine.popular(10))

with tab_similar:
    st.subheader("Find books similar to one you enjoyed")
    books = engine.metadata
    options = sorted(
        books.index.tolist(),
        key=lambda isbn: str(books.loc[isbn, "title"]).casefold(),
    )
    selected_isbn = st.selectbox(
        "Choose a book",
        options,
        format_func=lambda isbn: f"{books.loc[isbn, 'title']} — {books.loc[isbn, 'author']}",
    )
    if st.button("Recommend similar books", type="primary"):
        show_recommendations(engine.similar_books(selected_isbn, 10))

with tab_personal:
    st.subheader("Personalized recommendations")
    user_id = st.selectbox("Choose a reader ID", engine.user_ids)
    st.caption("Unknown or new readers can start with the popular-books list.")
    if st.button("Recommend for this reader", type="primary"):
        show_recommendations(engine.recommend_for_user(user_id, 10))
