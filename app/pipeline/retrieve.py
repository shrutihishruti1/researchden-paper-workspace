"""Retrieval orchestration: dense + BM25 → RRF → optional reranking.

Imports from sub-modules and re-exports all public names so existing code
that does ``from app.pipeline.retrieve import PaperStore`` continues to work.
"""

from __future__ import annotations

from app.pipeline.store import PaperStore, _decode_meta
from app.pipeline.bm25 import BM25Index, _tokenize, _TOKEN_RE
from app.pipeline.fusion import hybrid_search
from app.pipeline.reranker import Reranker

__all__ = [
    "PaperStore",
    "BM25Index",
    "hybrid_search",
    "Reranker",
    "retrieve",
    "_decode_meta",
    "_tokenize",
    "_TOKEN_RE",
]


def retrieve(
    query_text: str,
    query_vector: list[float],
    store: PaperStore,
    bm25: BM25Index,
    *,
    k: int = 5,
    where: dict | None = None,
    reranker: "Reranker | None" = None,
    fetch_factor: int = 3,
) -> list[dict]:
    """Full retrieval pipeline: dense + BM25 → RRF → optional reranking.

    Parameters
    ----------
    query_text:
        Raw query string (used for BM25 and reranking).
    query_vector:
        384-d normalised embedding of ``query_text``.
    store:
        A ``PaperStore`` instance with indexed chunks.
    bm25:
        A ``BM25Index`` built from the same chunks.
    k:
        Final number of chunks to return.
    where:
        Optional ChromaDB metadata filter for the dense leg only.
    reranker:
        If provided, apply cross-encoder reranking after RRF fusion.
    fetch_factor:
        Multiplier applied to ``k`` when fetching from each retriever before
        fusion (wider pool → better fusion quality).  Default: 3×.

    Returns
    -------
    Up to ``k`` result dicts, each with ``chunk_id``, ``text``, ``rrf_score``,
    metadata fields, and (if reranking) ``rerank_score``.
    """
    fetch_k = k * fetch_factor
    dense = store.query(query_vector, k=fetch_k, where=where)
    sparse = bm25.query(query_text, k=fetch_k)
    fused = hybrid_search(dense, sparse, k=k if reranker is None else fetch_k, rrf_k=60)
    if reranker is not None:
        fused = reranker.rerank(query_text, fused, k=k)
    return fused[:k]
