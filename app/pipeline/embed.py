from __future__ import annotations
import math
import re
from collections.abc import Callable
from .chunk import token_count

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

class EmbeddingError(RuntimeError):
    """Loading or encoding failed, or the backend violated the output contract."""

class EmbeddingService:
    def __init__(self, *, batch_size: int = 32, device: str = 'cpu', model=None, loader: Callable | None = None, revision: str | None = None):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.batch_size, self.device = batch_size, device
        self._model, self._loader, self.revision = model, loader, revision

    @property
    def model(self):
        if self._model is None:
            try:
                if self._loader:
                    self._model = self._loader()
                else:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(MODEL_NAME, device=self.device, revision=self.revision)
            except Exception as exc:
                raise EmbeddingError(f'Unable to load {MODEL_NAME}; check model cache/network: {exc}') from exc
        return self._model

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def token_limit(self) -> int:
        limit = min(int(self.model.max_seq_length), int(self.tokenizer.model_max_length), 256)
        if limit <= 0:
            raise EmbeddingError('Model has no usable token limit')
        return limit

    def info(self) -> dict:
        try:
            resolved = getattr(self.model[0].auto_model.config, '_commit_hash', None)
        except (TypeError, AttributeError, KeyError):
            resolved = None
        if not resolved:
            # Some Transformers releases omit _commit_hash while retaining
            # the actual snapshot path in tokenizer initialization metadata.
            for value in getattr(self.tokenizer, 'init_kwargs', {}).values():
                if isinstance(value, str):
                    match = re.search(r'snapshots[/\\]+([0-9a-f]{40})(?:[/\\]|$)', value)
                    if match:
                        resolved = match[1]
                        break
        return {'name': MODEL_NAME, 'revision': resolved, 'requested_revision': self.revision,
                'dimension': 384, 'normalize_embeddings': True, 'effective_token_limit': self.token_limit}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        for text in texts:
            if token_count(self.tokenizer, text) > self.token_limit:
                raise EmbeddingError('Input exceeds model token limit; truncation is forbidden.')
        try:
            raw = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False, prompt='')
            if len(raw) != len(texts):
                raise EmbeddingError('Backend returned an incorrect number of vectors')
            result = []
            for vector in raw:
                values = [float(v) for v in vector]
                if len(values) != 384 or not all(math.isfinite(v) for v in values):
                    raise EmbeddingError('Expected finite 384-dimensional embeddings')
                norm = math.sqrt(sum(v*v for v in values))
                if not norm or abs(norm - 1) > .001:
                    raise EmbeddingError('Backend returned a non-normalized embedding')
                result.append(values)
            return result
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f'Embedding failed: {exc}') from exc

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError('Query must contain text')
        return self.embed([text])[0]
