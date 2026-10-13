"""BM25 sparse keyword index."""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.split(text) if t]


class BM25Index:
    """Keyword search index backed by rank_bm25.BM25Okapi.

    Build from a ``process_pdf`` result dict or from raw lists; then call
    ``query`` to get ranked chunk ids with BM25 scores.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._bm25 = None  # built lazily after add_*

    def add_result(self, result: dict) -> int:
        """Append chunks from a ``process_pdf`` result."""
        chunks = result.get("chunks", [])
        for c in chunks:
            self._ids.append(c["chunk_id"])
            self._texts.append(c["text"])
        self._bm25 = None  # invalidate cached index
        return len(chunks)

    def add_payload(self, payload: dict) -> int:
        """Append chunks from a ``vector_store_payload`` dict."""
        ids = payload.get("ids", [])
        docs = payload.get("documents", [])
        for cid, doc in zip(ids, docs):
            self._ids.append(cid)
            self._texts.append(doc)
        self._bm25 = None
        return len(ids)

    def _build(self) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore[import]
        corpus = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(corpus)

    def query(self, query_text: str, *, k: int = 5) -> list[dict]:
        """Return top-k results by BM25 score.

        Returns
        -------
        List of dicts with ``chunk_id``, ``text``, ``score`` (BM25 raw score).
        Results are sorted descending by score; zero-score entries are excluded.
        """
        if not self._ids:
            return []
        if self._bm25 is None:
            self._build()
        tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(tokens)  # type: ignore[union-attr]
        ranked = sorted(
            ((float(s), i) for i, s in enumerate(scores) if s > 0),
            reverse=True,
        )[:k]
        return [
            {"chunk_id": self._ids[i], "text": self._texts[i], "score": score}
            for score, i in ranked
        ]

    def __len__(self) -> int:
        return len(self._ids)
