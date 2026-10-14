from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


def train_model(
    metadata,
    interactions,
    user_item,
    item_ids: list[str],
    user_ids: list[str],
    artifact_path: str | Path,
) -> dict:
    """Fit item-item cosine nearest neighbors and persist inference artifacts."""
    # Transposing makes each book a row and each reader a feature for item-to-item search.
    item_user = user_item.transpose().tocsr()
    # The query book itself is included in neighbor results, so allow one extra neighbor.
    neighbor_count = min(11, len(item_ids))
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=neighbor_count)
    neighbors.fit(item_user)

    rating_stats = interactions.groupby("ISBN")["Book-Rating"].agg(["mean", "count"])
    # Balance average rating with the number of ratings without letting volume dominate.
    popularity = rating_stats["mean"] * np.log1p(rating_stats["count"])
    bundle = {
        "version": 1,
        "metadata": metadata,
        "item_ids": item_ids,
        "user_ids": user_ids,
        "user_item": user_item,
        "item_user": item_user,
        "neighbors": neighbors,
        "popularity": popularity.to_dict(),
    }
    destination = Path(artifact_path)
    # Persist the fitted model and mappings together so inference uses training-time indices.
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return bundle
