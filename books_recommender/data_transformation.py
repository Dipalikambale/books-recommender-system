import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def transform_data(
    books: pd.DataFrame, ratings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, csr_matrix, list[str], list[str]]:
    """Normalize metadata and build a sparse user-by-book rating matrix."""
    # Keep one metadata record per ISBN so every matrix column maps to one displayable book.
    metadata = books[
        ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-M"]
    ].copy()
    metadata["ISBN"] = metadata["ISBN"].astype(str).str.strip()
    metadata = metadata.drop_duplicates(subset="ISBN", keep="first").set_index("ISBN")
    metadata = metadata.rename(
        columns={
            "Book-Title": "title",
            "Book-Author": "author",
            "Year-Of-Publication": "year",
            "Publisher": "publisher",
            "Image-URL-M": "image_url",
        }
    )
    for column in ("author", "year", "publisher", "image_url"):
        metadata[column] = metadata[column].fillna("").astype(str).str.strip()

    interactions = ratings[["User-ID", "ISBN", "Book-Rating"]].copy()
    interactions["User-ID"] = interactions["User-ID"].astype(str).str.strip()
    interactions["ISBN"] = interactions["ISBN"].astype(str).str.strip()
    interactions["Book-Rating"] = pd.to_numeric(interactions["Book-Rating"])
    # A zero in Book-Crossing means "not rated", not a negative book preference.
    interactions = interactions[interactions["Book-Rating"] > 0]
    # Discard ratings for ISBNs absent from the metadata supplied to this run.
    interactions = interactions[interactions["ISBN"].isin(metadata.index)]
    # Average duplicate user/book ratings so each matrix coordinate occurs once.
    interactions = (
        interactions.groupby(["User-ID", "ISBN"], as_index=False)["Book-Rating"].mean()
    )
    if interactions.empty:
        raise ValueError("No positive ratings match the books in the metadata table.")

    item_ids = metadata.index.tolist()
    user_ids = sorted(interactions["User-ID"].unique().tolist())
    item_index = {book_id: index for index, book_id in enumerate(item_ids)}
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    # Build CSR directly from coordinates; unlike a dense pivot table, this stores only ratings.
    row_indices = interactions["User-ID"].map(user_index).to_numpy(dtype=np.int32)
    column_indices = interactions["ISBN"].map(item_index).to_numpy(dtype=np.int32)
    values = interactions["Book-Rating"].to_numpy(dtype=np.float32)
    user_item = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    return metadata, interactions, user_item, item_ids, user_ids
