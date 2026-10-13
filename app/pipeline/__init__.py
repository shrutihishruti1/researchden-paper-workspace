"""Jamal's public PDF processing and pure vector-store handoff."""
from dataclasses import asdict
from pathlib import Path
import json
from .ingest import ingest_pdf
from .chunk import ChunkConfig, detect_sections, chunk_sections
from .embed import EmbeddingService


def process_pdf(source: str | Path | bytes, *, source_filename: str | None = None,
                config: ChunkConfig = ChunkConfig(), service: EmbeddingService | None = None) -> dict:
    paper = ingest_pdf(source, source_filename)
    service = service if service is not None else EmbeddingService()
    sections = detect_sections(paper)
    chunks = chunk_sections(paper, sections, service.tokenizer, service.token_limit, config)
    vectors = service.embed([c.embedding_input for c in chunks])
    return {'schema_version': '1.2', 'paper_id': paper.paper_id,
            'source_filename': paper.source_filename, 'page_count': paper.page_count,
            'metadata': paper.metadata, 'extraction_warnings': paper.warnings,
            'processing_warnings': paper.processing_warnings,
            'chunking_configuration': asdict(config), 'model': service.info(),
            'chunks': [dict(asdict(c), embedding=v) for c, v in zip(chunks, vectors, strict=True)]}


def vector_store_payload(result: dict) -> dict:
    """Return aligned arrays without initializing any storage client."""
    chunks = result['chunks']
    return {'ids': [c['chunk_id'] for c in chunks],
            'documents': [c['text'] for c in chunks],
            'embeddings': [c['embedding'] for c in chunks],
            'metadatas': [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                           for k, v in c.items() if v is not None and k not in ('embedding', 'text', 'embedding_input')}
                          for c in chunks]}

__all__ = ['process_pdf', 'vector_store_payload', 'ChunkConfig', 'EmbeddingService']
