"""Reciprocal Rank Fusion for combining dense and sparse results."""

from __future__ import annotations


def hybrid_search(
    dense_results: list[dict],
    sparse_results: list[dict],
    *,
    k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """Combine dense and sparse result lists using Reciprocal Rank Fusion (RRF).

    RRF score for a document ``d`` across N ranked lists is::

        sum_i  1 / (rrf_k + rank_i(d))

    where ``rank_i`` is 1-based.  Documents absent from a list receive no
    contribution from that list (they are not penalised).

    Parameters
    ----------
    dense_results:
        Ordered list from ``PaperStore.query`` (highest score first).
    sparse_results:
        Ordered list from ``BM25Index.query`` (highest score first).
    k:
        Number of final results to return.
    rrf_k:
        The RRF constant (default 60, standard in the literature).

    Returns
    -------
    List of dicts; each has ``chunk_id``, ``text``, ``rrf_score``, and all
    metadata fields from the dense result (sparse-only hits carry no metadata).
    """
    by_id: dict[str, dict] = {}
    for r in dense_results:
        by_id[r["chunk_id"]] = dict(r)
    for r in sparse_results:
        if r["chunk_id"] not in by_id:
            by_id[r["chunk_id"]] = dict(r)

    rrf: dict[str, float] = {}
    for rank, r in enumerate(dense_results, start=1):
        cid = r["chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, r in enumerate(sparse_results, start=1):
        cid = r["chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
    out = []
    for cid, score in ranked:
        entry = dict(by_id[cid])
        entry["rrf_score"] = score
        entry.pop("score", None)
        out.append(entry)
    return out
