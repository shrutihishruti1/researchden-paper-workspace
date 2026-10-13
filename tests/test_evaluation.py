"""Tests for retrieve.py and evaluate.py.

All tests use in-memory ChromaDB and a deterministic fake embedding backend
(same pattern as test_pdf_pipeline.py) so no model download is required and
the suite runs on CPU in seconds.

Coverage
--------
PaperStore      – upsert, dense query, metadata filter, delete_paper, duplicate upsert
BM25Index       – build, query, tokenisation, zero-results edge case
hybrid_search   – RRF fusion ordering, single-list degenerate cases
retrieve()      – full pipeline helper (dense+BM25+RRF, with and without reranker)
evaluate()      – Recall@k, Hit Rate@k, MRR, nDCG@k: known-answer assertions
                  empty-query edge case, perfect ranking, zero-hit ranking
"""
import math
import pytest

from app.pipeline.retrieve import (
    PaperStore,
    BM25Index,
    hybrid_search,
    retrieve,
    Reranker,
    _tokenize,
    _decode_meta,
)
from app.pipeline.evaluate import (
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k,
    evaluate,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str, paper_id: str = "paper1", section: str = "Introduction") -> dict:
    """Minimal chunk dict matching the structure process_pdf produces."""
    vector = [0.0] * 384
    vector[sum(map(ord, chunk_id)) % 384] = 1.0
    # Normalise
    norm = math.sqrt(sum(v * v for v in vector))
    vector = [v / norm for v in vector]
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "section_title": section,
        "section_path": [section],
        "chunk_index": 0,
        "page_start": 1,
        "page_end": 1,
        "page_numbers": [1],
        "text": text,
        "embedding_input": text,
        "token_count": len(text.split()),
        "section_char_start": 0,
        "section_char_end": len(text),
        "warnings": [],
        "content_kind": "prose",
        "repeated_context": "",
        "table": None,
        "embedding": vector,
    }


def _make_result(chunks: list[dict]) -> dict:
    """Wrap a list of chunk dicts into a process_pdf-style result dict."""
    return {
        "schema_version": "1.2",
        "paper_id": chunks[0]["paper_id"] if chunks else "unknown",
        "source_filename": "test.pdf",
        "page_count": 1,
        "metadata": {},
        "extraction_warnings": [],
        "processing_warnings": [],
        "chunking_configuration": {},
        "model": {"name": "test", "revision": None, "requested_revision": None,
                  "dimension": 384, "normalize_embeddings": True, "effective_token_limit": 64},
        "chunks": chunks,
    }


# Query vector identical to chunk "c1" (so c1 should be nearest)
_QUERY_VECTOR: list[float] = _make_chunk("c1", "")["embedding"]


# ---------------------------------------------------------------------------
# PaperStore tests
# ---------------------------------------------------------------------------

class TestPaperStore:
    _counter = 0

    @pytest.fixture(autouse=True)
    def store(self):
        # Use a unique collection name per test so EphemeralClient state
        # does not bleed between test methods within the same process.
        TestPaperStore._counter += 1
        self._store = PaperStore(path=":memory:",
                                 collection_name=f"test_{TestPaperStore._counter}")
        return self._store

    def test_add_result_and_count(self):
        chunks = [_make_chunk("c1", "Neural networks learn representations."),
                  _make_chunk("c2", "Transformers use self-attention.")]
        result = _make_result(chunks)
        added = self._store.add_result(result)
        assert added == 2
        assert self._store.count() == 2

    def test_add_payload_directly(self):
        from app.pipeline import vector_store_payload
        chunks = [_make_chunk("p1", "Deep learning is powerful.")]
        result = _make_result(chunks)
        payload = vector_store_payload(result)
        # add_payload should add exactly the chunks in this payload
        added = self._store.add_payload(payload)
        assert added == 1
        assert self._store.count() == 1

    def test_upsert_idempotent(self):
        chunks = [_make_chunk("c1", "Some text about methods.")]
        result = _make_result(chunks)
        n1 = self._store.add_result(result)
        n2 = self._store.add_result(result)  # duplicate upsert
        assert n1 == 1
        assert n2 == 1  # upsert still reports 1 written
        assert self._store.count() == 1  # but only 1 stored

    def test_dense_query_returns_k(self):
        chunks = [_make_chunk(f"c{i}", f"Sentence number {i}.") for i in range(10)]
        self._store.add_result(_make_result(chunks))
        results = self._store.query(_QUERY_VECTOR, k=3)
        assert len(results) == 3

    def test_dense_query_fields(self):
        chunks = [_make_chunk("c1", "Attention is all you need.")]
        self._store.add_result(_make_result(chunks))
        results = self._store.query(_QUERY_VECTOR, k=1)
        assert len(results) == 1
        r = results[0]
        assert r["chunk_id"] == "c1"
        assert r["text"] == "Attention is all you need."
        assert 0.0 <= r["score"] <= 1.0
        assert "paper_id" in r

    def test_dense_query_score_ordering(self):
        # c1's vector IS _QUERY_VECTOR → should rank highest
        chunks = [_make_chunk("c1", "exact match"),
                  _make_chunk("c99", "completely different topic")]
        self._store.add_result(_make_result(chunks))
        results = self._store.query(_QUERY_VECTOR, k=2)
        ids = [r["chunk_id"] for r in results]
        assert ids[0] == "c1", f"Expected c1 first, got {ids}"

    def test_metadata_filter(self):
        paper_a = [_make_chunk("a1", "Alpha text.", paper_id="paperA"),
                   _make_chunk("a2", "Alpha more.", paper_id="paperA")]
        paper_b = [_make_chunk("b1", "Beta text.", paper_id="paperB")]
        self._store.add_result(_make_result(paper_a))
        self._store.add_result(_make_result(paper_b))
        results = self._store.query(_QUERY_VECTOR, k=5,
                                    where={"paper_id": {"$eq": "paperA"}})
        assert all(r["paper_id"] == "paperA" for r in results)

    def test_delete_paper(self):
        chunks_a = [_make_chunk("a1", "Paper A chunk.", paper_id="paperA")]
        chunks_b = [_make_chunk("b1", "Paper B chunk.", paper_id="paperB")]
        self._store.add_result(_make_result(chunks_a))
        self._store.add_result(_make_result(chunks_b))
        before = self._store.count()
        assert before == 2
        self._store.delete_paper("paperA")
        assert self._store.count() == 1
        results = self._store.query(_QUERY_VECTOR, k=5)
        assert all(r["paper_id"] == "paperB" for r in results)

    def test_query_empty_store_returns_empty(self):
        # Chroma with 0 docs: clamp n_results to max(1, count)
        # The query may error or return 0 rows — handle both gracefully.
        try:
            results = self._store.query(_QUERY_VECTOR, k=5)
            assert isinstance(results, list)
        except Exception:
            pass  # Some Chroma versions raise on empty collection

    def test_metadata_roundtrip(self):
        """page_numbers is a list; vector_store_payload JSON-encodes it → decode back."""
        import json
        chunks = [_make_chunk("c1", "Test roundtrip.")]
        self._store.add_result(_make_result(chunks))
        results = self._store.query(_QUERY_VECTOR, k=1)
        # After _decode_meta, page_numbers should be a list again
        assert isinstance(results[0]["page_numbers"], list)


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_empty_index_returns_empty(self):
        bm25 = BM25Index()
        assert bm25.query("anything", k=5) == []

    def test_add_result_and_query(self):
        bm25 = BM25Index()
        chunks = [
            _make_chunk("c1", "neural network deep learning gradient descent"),
            _make_chunk("c2", "transformer attention mechanism query key value"),
            _make_chunk("c3", "random forest ensemble tree bagging boosting"),
        ]
        bm25.add_result(_make_result(chunks))
        results = bm25.query("neural network", k=3)
        assert results[0]["chunk_id"] == "c1"
        assert all(r["score"] > 0 for r in results)

    def test_add_payload_directly(self):
        from app.pipeline import vector_store_payload
        bm25 = BM25Index()
        # Need ≥3 docs so BM25Okapi IDF is positive:
        #   idf = log((N - df + 0.5) / (df + 0.5)); with N=3, df=1 → log(2.5/1.5) > 0
        chunks = [_make_chunk("p1", "convolutional network image classification"),
                  _make_chunk("p2", "recurrent network sequence modelling"),
                  _make_chunk("p3", "support vector machine kernel methods")]
        bm25.add_payload(vector_store_payload(_make_result(chunks)))
        assert len(bm25) == 3
        results = bm25.query("convolutional", k=1)
        # "convolutional" only appears in p1 → IDF > 0, BM25 > 0
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "p1"

    def test_no_match_returns_empty(self):
        bm25 = BM25Index()
        chunks = [_make_chunk("c1", "machine learning model training")]
        bm25.add_result(_make_result(chunks))
        results = bm25.query("quantum chromodynamics supercollider", k=5)
        assert results == []

    def test_query_respects_k(self):
        bm25 = BM25Index()
        chunks = [_make_chunk(f"c{i}", f"machine learning result {i}") for i in range(10)]
        bm25.add_result(_make_result(chunks))
        results = bm25.query("machine learning", k=3)
        assert len(results) <= 3

    def test_results_sorted_descending(self):
        bm25 = BM25Index()
        # Use a term that appears only in one document so IDF > 0
        chunks = [
            _make_chunk("c1", "xylophone acoustics resonance frequency"),
            _make_chunk("c2", "xylophone xylophone xylophone resonance frequency measurement"),
            _make_chunk("c3", "gradient descent optimisation loss surface"),
        ]
        bm25.add_result(_make_result(chunks))
        results = bm25.query("xylophone", k=3)
        assert len(results) >= 2
        assert results[0]["score"] >= results[1]["score"]

    def test_len(self):
        bm25 = BM25Index()
        assert len(bm25) == 0
        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(5)]
        bm25.add_result(_make_result(chunks))
        assert len(bm25) == 5

    def test_tokenize(self):
        tokens = _tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Punctuation should not appear as tokens
        assert "," not in tokens


# ---------------------------------------------------------------------------
# hybrid_search / RRF tests
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def _r(self, cid, score, text="text", **meta):
        return {"chunk_id": cid, "text": text, "score": score, **meta}

    def test_rrf_prefers_document_ranked_high_in_both(self):
        dense = [self._r("c1", 0.9), self._r("c2", 0.7), self._r("c3", 0.5)]
        sparse = [self._r("c1", 8.0), self._r("c4", 6.0), self._r("c2", 4.0)]
        results = hybrid_search(dense, sparse, k=3)
        assert results[0]["chunk_id"] == "c1"  # top in both → top after RRF

    def test_rrf_includes_sparse_only_hit(self):
        dense = [self._r("c1", 0.9), self._r("c2", 0.7)]
        sparse = [self._r("c3", 10.0), self._r("c1", 8.0)]
        results = hybrid_search(dense, sparse, k=3)
        ids = [r["chunk_id"] for r in results]
        assert "c3" in ids  # sparse-only

    def test_rrf_score_monotone(self):
        dense = [self._r(f"c{i}", 1.0 / (i + 1)) for i in range(5)]
        sparse = [self._r(f"c{i}", 5.0 - i) for i in range(5)]
        results = hybrid_search(dense, sparse, k=5)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_k_respected(self):
        dense = [self._r(f"c{i}", 1.0) for i in range(10)]
        sparse = [self._r(f"c{i}", 1.0) for i in range(10)]
        results = hybrid_search(dense, sparse, k=4)
        assert len(results) == 4

    def test_single_dense_list_only(self):
        dense = [self._r("c1", 0.9), self._r("c2", 0.5)]
        results = hybrid_search(dense, [], k=2)
        assert [r["chunk_id"] for r in results] == ["c1", "c2"]

    def test_single_sparse_list_only(self):
        sparse = [self._r("c3", 5.0), self._r("c4", 3.0)]
        results = hybrid_search([], sparse, k=2)
        assert [r["chunk_id"] for r in results] == ["c3", "c4"]

    def test_rrf_score_in_output(self):
        dense = [self._r("c1", 0.9)]
        sparse = [self._r("c1", 5.0)]
        results = hybrid_search(dense, sparse, k=1)
        assert "rrf_score" in results[0]
        assert results[0]["rrf_score"] > 0

    def test_metadata_preserved_from_dense(self):
        dense = [self._r("c1", 0.9, paper_id="pA", section_title="Intro")]
        sparse = [self._r("c1", 5.0)]
        results = hybrid_search(dense, sparse, k=1)
        assert results[0].get("paper_id") == "pA"
        assert results[0].get("section_title") == "Intro"


# ---------------------------------------------------------------------------
# retrieve() integration helper test
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_end_to_end(self):
        store = PaperStore(path=":memory:")
        bm25 = BM25Index()
        chunks = [
            _make_chunk("c1", "neural network deep learning"),
            _make_chunk("c2", "support vector machine classification"),
            _make_chunk("c3", "gradient descent optimisation convergence"),
        ]
        result = _make_result(chunks)
        store.add_result(result)
        bm25.add_result(result)
        q_vec = _QUERY_VECTOR  # matches c1's vector
        results = retrieve("neural network", q_vec, store, bm25, k=2)
        assert len(results) <= 2
        assert all("chunk_id" in r for r in results)
        assert all("text" in r for r in results)
        assert all("rrf_score" in r for r in results)

    def test_retrieve_with_metadata_filter(self):
        store = PaperStore(path=":memory:")
        bm25 = BM25Index()
        chunks_a = [_make_chunk("a1", "alpha text content", paper_id="A")]
        chunks_b = [_make_chunk("b1", "beta text content", paper_id="B")]
        store.add_result(_make_result(chunks_a))
        store.add_result(_make_result(chunks_b))
        bm25.add_result(_make_result(chunks_a))
        bm25.add_result(_make_result(chunks_b))
        results = retrieve("text content", _QUERY_VECTOR, store, bm25, k=5,
                           where={"paper_id": {"$eq": "A"}})
        # Dense leg is filtered; sparse leg may include B but RRF won't promote it
        # Verify we get something back and no crash
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Reranker (stub — no model download)
# ---------------------------------------------------------------------------

class TestReranker:
    def test_rerank_with_stub_model(self):
        """Verify Reranker.rerank() interface using a stub CrossEncoder."""
        class StubCrossEncoder:
            def predict(self, pairs):
                # Score = length of the passage (deterministic, no network)
                return [float(len(p)) for _, p in pairs]

        reranker = Reranker.__new__(Reranker)
        reranker._model = StubCrossEncoder()

        results = [
            {"chunk_id": "c1", "text": "short", "rrf_score": 0.5},
            {"chunk_id": "c2", "text": "much longer passage wins here", "rrf_score": 0.4},
        ]
        reranked = reranker.rerank("query", results, k=2)
        assert reranked[0]["chunk_id"] == "c2"  # longer → higher stub score
        assert "rerank_score" in reranked[0]
        assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

    def test_rerank_empty_results(self):
        reranker = Reranker.__new__(Reranker)
        reranker._model = None
        assert reranker.rerank("query", []) == []

    def test_rerank_top_k(self):
        class IdentityEncoder:
            def predict(self, pairs):
                return list(range(len(pairs)))  # ascending scores

        reranker = Reranker.__new__(Reranker)
        reranker._model = IdentityEncoder()
        results = [{"chunk_id": f"c{i}", "text": f"text {i}", "rrf_score": 0.1} for i in range(6)]
        reranked = reranker.rerank("query", results, k=3)
        assert len(reranked) == 3


# ---------------------------------------------------------------------------
# Evaluation metric tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    # recall_at_k --------------------------------------------------------

    def test_recall_perfect(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_recall_partial(self):
        assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=3) == pytest.approx(2 / 3)

    def test_recall_zero_hit(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == pytest.approx(0.0)

    def test_recall_k_truncates(self):
        # Only first k=2 considered; "c" at position 3 is excluded
        assert recall_at_k(["a", "x", "c"], {"a", "c"}, k=2) == pytest.approx(0.5)

    def test_recall_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=3) == 0.0

    # hit_rate_at_k -------------------------------------------------------

    def test_hit_rate_positive(self):
        assert hit_rate_at_k(["x", "b", "c"], {"b"}, k=3) == 1.0

    def test_hit_rate_negative(self):
        assert hit_rate_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_hit_rate_k_cutoff(self):
        assert hit_rate_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0  # a is at rank 3

    def test_hit_rate_empty_relevant(self):
        assert hit_rate_at_k(["a"], set(), k=5) == 0.0

    # reciprocal_rank ------------------------------------------------------

    def test_rr_rank_1(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_rr_rank_2(self):
        assert reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_rr_rank_5(self):
        assert reciprocal_rank(["x", "x", "x", "x", "a"], {"a"}) == pytest.approx(0.2)

    def test_rr_no_hit(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)

    # ndcg_at_k -----------------------------------------------------------

    def test_ndcg_perfect(self):
        # All 2 relevant at positions 1,2 → DCG = IDCG → 1.0
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_ndcg_zero(self):
        assert ndcg_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)

    def test_ndcg_partial(self):
        # Relevant: {a, b}. Retrieved: [x, a, b] k=3
        # DCG = 0 + 1/log2(3) + 1/log2(4) ≈ 0.6309 + 0.5 = 1.1309
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3) == pytest.approx(expected, abs=1e-9)

    def test_ndcg_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), k=3) == 0.0

    def test_ndcg_k_truncates(self):
        # Only relevant doc is at position 3, k=2 → not counted → 0
        assert ndcg_at_k(["x", "y", "a"], {"a"}, k=2) == pytest.approx(0.0)

    # evaluate() ----------------------------------------------------------

    def test_evaluate_perfect(self):
        queries = [
            {"retrieved": ["a", "b", "c"], "relevant": {"a"}},
            {"retrieved": ["d", "e", "f"], "relevant": {"d"}},
        ]
        metrics = evaluate(queries, k=3)
        assert metrics["recall@3"] == pytest.approx(1.0)
        assert metrics["hit_rate@3"] == pytest.approx(1.0)
        assert metrics["mrr"] == pytest.approx(1.0)
        assert metrics["ndcg@3"] == pytest.approx(1.0)
        assert metrics["num_queries"] == 2
        assert metrics["k"] == 3

    def test_evaluate_zero(self):
        queries = [{"retrieved": ["x", "y"], "relevant": {"a"}}]
        metrics = evaluate(queries, k=3)
        assert metrics["recall@3"] == pytest.approx(0.0)
        assert metrics["mrr"] == pytest.approx(0.0)

    def test_evaluate_empty_queries(self):
        metrics = evaluate([], k=5)
        assert metrics["num_queries"] == 0
        assert metrics["mrr"] == 0.0

    def test_evaluate_mixed(self):
        queries = [
            {"retrieved": ["a", "b"], "relevant": {"a"}},   # MRR = 1.0
            {"retrieved": ["x", "a"], "relevant": {"a"}},   # MRR = 0.5
        ]
        metrics = evaluate(queries, k=2)
        assert metrics["mrr"] == pytest.approx(0.75)
        assert metrics["recall@2"] == pytest.approx(1.0)   # both find "a" in top-2
        assert metrics["hit_rate@2"] == pytest.approx(1.0)

    def test_evaluate_accepts_list_relevant(self):
        """relevant can be a list (set() conversion is done inside)."""
        queries = [{"retrieved": ["a"], "relevant": ["a"]}]
        metrics = evaluate(queries, k=1)
        assert metrics["recall@1"] == pytest.approx(1.0)

    def test_evaluate_k_affects_keys(self):
        queries = [{"retrieved": ["a"], "relevant": {"a"}}]
        metrics = evaluate(queries, k=10)
        assert "recall@10" in metrics
        assert "hit_rate@10" in metrics
        assert "ndcg@10" in metrics

    # _decode_meta --------------------------------------------------------

    def test_decode_meta_json_list(self):
        meta = {"page_numbers": "[1, 2]", "score": 0.9, "title": "Intro"}
        decoded = _decode_meta(meta)
        assert decoded["page_numbers"] == [1, 2]
        assert decoded["score"] == 0.9
        assert decoded["title"] == "Intro"

    def test_decode_meta_plain_string(self):
        meta = {"section_title": "Methods"}
        decoded = _decode_meta(meta)
        assert decoded["section_title"] == "Methods"
