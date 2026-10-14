import numpy as np
import pandas as pd


class RecommendationEngine:
    """Generate item-similarity and user-personalized book recommendations."""

    def __init__(self, artifacts: dict):
        self.artifacts = artifacts
        self.metadata: pd.DataFrame = artifacts["metadata"]
        self.item_ids: list[str] = artifacts["item_ids"]
        self.item_index = {book_id: index for index, book_id in enumerate(self.item_ids)}
        self.user_ids: list[str] = artifacts["user_ids"]
        self.user_index = {user_id: index for index, user_id in enumerate(self.user_ids)}

    def _result(self, scored: dict[str, float], limit: int) -> pd.DataFrame:
        # Convert ranked ISBN scores into the metadata shape consumed by the app.
        ranked = sorted(scored.items(), key=lambda pair: pair[1], reverse=True)[:limit]
        if not ranked:
            return pd.DataFrame(columns=["isbn", "title", "author", "year", "image_url", "score"])
        rows = []
        for isbn, score in ranked:
            if isbn not in self.metadata.index:
                continue
            book = self.metadata.loc[isbn]
            rows.append(
                {
                    "isbn": isbn,
                    "title": book["title"],
                    "author": book["author"],
                    "year": book["year"],
                    "image_url": book["image_url"],
                    "score": float(score),
                }
            )
        return pd.DataFrame(rows)

    def popular(self, limit: int = 10) -> pd.DataFrame:
        popularity = self.artifacts["popularity"]
        ranked = sorted(popularity.items(), key=lambda pair: pair[1], reverse=True)
        return self._result(dict(ranked[:limit]), limit)

    def similar_books(self, isbn: str, limit: int = 10) -> pd.DataFrame:
        isbn = str(isbn).strip()
        # Cold-start requests use popularity because no learned item vector exists.
        if isbn not in self.item_index:
            return self.popular(limit)
        model = self.artifacts["neighbors"]
        distances, indices = model.kneighbors(
            self.artifacts["item_user"][self.item_index[isbn]],
            n_neighbors=min(limit + 1, len(self.item_ids)),
        )
        scored = {
            self.item_ids[index]: 1.0 - float(distance)
            for distance, index in zip(distances[0], indices[0])
            if self.item_ids[index] != isbn
        }
        return self._result(scored, limit)

    def recommend_for_user(self, user_id: str, limit: int = 10) -> pd.DataFrame:
        user_id = str(user_id).strip()
        # A reader unseen during training has no preference vector to personalize from.
        if user_id not in self.user_index:
            return self.popular(limit)

        row = self.artifacts["user_item"].getrow(self.user_index[user_id])
        rated_items = set(row.indices.tolist())
        # Use the reader's highest ratings as positive preference examples.
        favorite_positions = np.argsort(row.data)[::-1][: min(5, len(row.data))]
        scores: dict[str, float] = {}
        model = self.artifacts["neighbors"]

        for position in favorite_positions:
            item_position = int(row.indices[position])
            rating_weight = float(row.data[position]) / 10.0
            distances, indices = model.kneighbors(
                self.artifacts["item_user"][item_position],
                n_neighbors=min(limit + len(rated_items) + 1, len(self.item_ids)),
            )
            for distance, neighbor_index in zip(distances[0], indices[0]):
                # Never recommend books the reader has already rated.
                neighbor_index = int(neighbor_index)
                if neighbor_index in rated_items:
                    continue
                isbn = self.item_ids[neighbor_index]
                similarity = max(0.0, 1.0 - float(distance))
                # Accumulate similarity weighted by how strongly the reader liked the seed book.
                scores[isbn] = scores.get(isbn, 0.0) + similarity * rating_weight

        if not scores:
            # Sparse histories may have no unseen neighbors; popularity is a useful fallback.
            return self.popular(limit)
        return self._result(scores, limit)
