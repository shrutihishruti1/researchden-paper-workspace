"""Retrieval evaluation metrics.

All functions operate on abstract ranked lists of IDs; they are decoupled from
any retrieval implementation so they can be unit-tested independently.

Metrics
-------
recall_at_k     – fraction of relevant IDs found in the top-k results
hit_rate_at_k   – 1 if *any* relevant ID is in top-k, else 0  (per-query)
mrr             – Mean Reciprocal Rank over a list of queries
ndcg_at_k       – Normalised Discounted Cumulative Gain (binary relevance)
evaluate        – Compute all four metrics in one call
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Per-query helpers
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of *relevant* IDs that appear in the first ``k`` retrieved IDs.

    Returns 0.0 when ``relevant`` is empty (nothing to recall).
    """
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return len(set(top) & relevant) / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant ID is in the first ``k`` results, else 0.0."""
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the *first* relevant result (1-based).

    Returns 0.0 if no relevant result appears in ``retrieved``.
    """
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary nDCG@k.

    DCG uses ``1 / log2(rank + 1)`` for each hit.  IDCG is the DCG of a
    perfect ranking (all ``|relevant|`` hits at the top, capped at k).
    Returns 0.0 when ``relevant`` is empty.
    """
    if not relevant:
        return 0.0
    top = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(top, start=1)
        if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------

def evaluate(
    queries: list[dict],
    *,
    k: int = 5,
) -> dict:
    """Aggregate retrieval metrics over a list of query dicts.

    Parameters
    ----------
    queries:
        Each entry must have:
        - ``"retrieved"`` : ``list[str]`` – ordered chunk IDs from the retriever
        - ``"relevant"``  : ``set[str] | list[str]`` – ground-truth relevant IDs
    k:
        Cut-off rank for all metrics.

    Returns
    -------
    Dict with keys ``recall@k``, ``hit_rate@k``, ``mrr``, ``ndcg@k``,
    ``num_queries``, and ``k``.
    """
    if not queries:
        return {"recall@k": 0.0, "hit_rate@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0,
                "num_queries": 0, "k": k}

    recalls, hits, rrs, ndcgs = [], [], [], []
    for q in queries:
        retrieved: list[str] = q["retrieved"]
        relevant: set[str] = set(q["relevant"])
        recalls.append(recall_at_k(retrieved, relevant, k))
        hits.append(hit_rate_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))

    n = len(queries)
    return {
        f"recall@{k}": sum(recalls) / n,
        f"hit_rate@{k}": sum(hits) / n,
        "mrr": sum(rrs) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
        "num_queries": n,
        "k": k,
    }
