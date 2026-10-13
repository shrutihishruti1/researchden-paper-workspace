"""ChromaDB-backed dense vector store."""

from __future__ import annotations

import json
from typing import Any


def _decode_meta(meta: dict) -> dict:
    """Deserialise JSON-encoded metadata values back to native Python types."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


class PaperStore:
    """Persistent ChromaDB-backed vector store for one or many papers.

    Parameters
    ----------
    path:
        Directory on disk where ChromaDB persists its files.  Pass ``":memory:"``
        for an in-memory ephemeral store (useful in tests).
    collection_name:
        ChromaDB collection to use.  One collection can hold chunks from many
        papers; ``paper_id`` is stored as metadata so you can filter by paper.
    """

    def __init__(self, path: str = "./chroma_store", collection_name: str = "papers") -> None:
        import chromadb
        if path == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_result(self, result: dict) -> int:
        """Ingest the dict returned by ``process_pdf`` or a saved JSON file.

        Pre-computed embeddings are passed directly to Chroma; the embedding
        model is *never* called again.  Duplicate chunk_ids are silently
        skipped (upsert semantics).

        Returns the number of new chunks written (0 if all already present).
        """
        from app.pipeline import vector_store_payload
        payload = vector_store_payload(result)
        ids = payload["ids"]
        if not ids:
            return 0
        # Chroma requires metadata values to be str/int/float/bool.
        # vector_store_payload already serialises lists/dicts to JSON strings.
        self._collection.upsert(
            ids=ids,
            embeddings=payload["embeddings"],
            documents=payload["documents"],
            metadatas=payload["metadatas"],
        )
        return len(ids)

    def add_payload(self, payload: dict) -> int:
        """Ingest a payload dict (from ``vector_store_payload``) directly."""
        ids = payload["ids"]
        if not ids:
            return 0
        self._collection.upsert(
            ids=ids,
            embeddings=payload["embeddings"],
            documents=payload["documents"],
            metadatas=payload["metadatas"],
        )
        return len(ids)

    def query(
        self,
        query_vector: list[float],
        *,
        k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Dense (cosine) nearest-neighbour search.

        Parameters
        ----------
        query_vector:
            A 384-d normalised float list (output of ``EmbeddingService.embed_query``).
        k:
            Number of results to return.
        where:
            Optional ChromaDB metadata filter, e.g. ``{"paper_id": {"$eq": "abc123"}}``.

        Returns
        -------
        List of result dicts, each with keys:
            ``chunk_id``, ``text``, ``score`` (cosine similarity 0–1),
            and all stored metadata fields.
        """
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_vector],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        raw = self._collection.query(**kwargs)
        results = []
        for cid, doc, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # Chroma returns L2 distance for cosine space; convert to similarity.
            score = max(0.0, 1.0 - float(dist) / 2.0)
            results.append({"chunk_id": cid, "text": doc, "score": score, **_decode_meta(meta)})
        return results

    def count(self) -> int:
        return self._collection.count()

    def delete_paper(self, paper_id: str) -> None:
        """Remove all chunks for a given paper_id."""
        self._collection.delete(where={"paper_id": {"$eq": paper_id}})
