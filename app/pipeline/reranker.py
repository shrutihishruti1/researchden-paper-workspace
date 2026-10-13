"""Cross-encoder reranker (CPU-safe, opt-in)."""

from __future__ import annotations


class Reranker:
    """Cross-encoder reranker (CPU-safe).  Disabled by default; opt-in only.

    Parameters
    ----------
    model_name:
        A sentence-transformers CrossEncoder model name.  The lightweight
        ``cross-encoder/ms-marco-MiniLM-L-6-v2`` works well on CPU.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None, *, device: str = "cpu") -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self._model = None  # lazy

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(self, query: str, results: list[dict], *, k: int | None = None) -> list[dict]:
        """Score each (query, passage) pair and re-sort descending.

        Parameters
        ----------
        query:
            The natural language query string.
        results:
            A list of result dicts (each must have a ``text`` key).
        k:
            Truncate to top-k after reranking.  ``None`` returns all.

        Returns
        -------
        Re-sorted list with an additional ``rerank_score`` key on each dict.
        """
        if not results:
            return []
        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)
        reranked = sorted(
            zip(scores, results),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        out = []
        for score, entry in reranked:
            item = dict(entry)
            item["rerank_score"] = float(score)
            out.append(item)
        return out[:k] if k is not None else out
