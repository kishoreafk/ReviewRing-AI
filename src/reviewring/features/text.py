"""Frozen text embeddings with an on-disk cache and a TF-IDF fallback.

The default encoder is ``sentence-transformers/all-MiniLM-L6-v2`` (384-d).
Its HuggingFace revision is resolved once, recorded, and pinned in the cache
key so that re-encoding happens automatically when the model, preprocessing
version or text content changes. When the encoder cannot be downloaded (e.g.
offline machine), a deterministic TF-IDF + SVD fallback keeps the pipeline
runnable and sets ``encoder_id`` accordingly, which propagates into run
manifests so no result is ever ambiguous about which text representation
produced it.

Cache key = sha256(encoder_id | revision | preprocessing_version | text_hash).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from reviewring.utils.runtime import sha256_text

logger = logging.getLogger(__name__)

PREPROCESSING_VERSION = "1.0.0"  # change when title/body join rules change
MINILM_DIM = 384
FALLBACK_DIM = 128


def build_encoder_text(title: str | None, body: str | None) -> str:
    """Join title and body with a separator; negation/punctuation preserved.

    No stemming, no stop-word removal: transformer embeddings expect raw text.
    """
    title_part = (title or "").strip()
    body_part = (body or "").strip()
    if title_part and body_part:
        return f"{title_part}. {body_part}"
    return title_part or body_part or ""


class TextEncoder:
    """Common interface wrapper so downstream code is encoder-agnostic."""

    def __init__(self, encoder_id: str, dim: int, revision: str | None, kind: str):
        self.encoder_id = encoder_id
        self.dim = dim
        self.revision = revision
        self.kind = kind  # "minilm" | "tfidf"

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError

    def encode_many(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), 4096):
            chunk = texts[start : start + 4096]
            out[start : start + len(chunk)] = self.encode(chunk, batch_size=batch_size)
        return out


class MiniLMEncoder(TextEncoder):
    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        # pin the concrete HF revision for reproducibility (spec 10.1/17.1)
        rev = None
        try:
            rev = model[0].auto_model.config._commit_hash
        except Exception:  # noqa: BLE001
            rev = None
        if rev is None:
            rev = getattr(getattr(model, "model_card_data", None), "model_revision", None)
        super().__init__(encoder_id=model_name, dim=MINILM_DIM, revision=rev, kind="minilm")
        self._model = model
        self.batch_size = batch_size

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        filled = [t if t.strip() else " " for t in texts]
        vectors = self._model.encode(
            filled,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # empty texts stay all-zero so the missingness flag carries meaning
        out = np.asarray(vectors, dtype=np.float32)
        for i, t in enumerate(texts):
            if not t.strip():
                out[i] = 0.0
        return out


class TfidfHashingEncoder(TextEncoder):
    """Deterministic offline fallback: hashed char+word TF-IDF then SVD."""

    def __init__(self, dim: int = FALLBACK_DIM):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            dtype=np.float32,
        )
        self._svd = TruncatedSVD(n_components=dim, random_state=0)
        self._fitted = False
        super().__init__(encoder_id="tfidf-hash-unigram-bigram-svd", dim=dim, revision=None, kind="tfidf")

    def fit(self, texts: list[str]) -> TfidfHashingEncoder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        min_df = 2 if len(texts) >= 50 else 1  # tiny corpora need min_df=1
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=min_df,
            dtype=np.float32,
        )
        matrix = self._vectorizer.fit_transform(texts)
        n_comp = max(2, min(self.dim, matrix.shape[1] - 1, matrix.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self._svd.fit(matrix)
        self.dim = n_comp
        self._fitted = True
        return self

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfHashingEncoder must be fit before encoding")
        filled = [t if t.strip() else " " for t in texts]
        matrix = self._vectorizer.transform(filled)
        vectors = self._svd.transform(matrix).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        for i, t in enumerate(texts):
            if not t.strip():
                vectors[i] = 0.0
        return vectors


def get_encoder(config: dict, texts_for_fit: list[str] | None = None) -> tuple[TextEncoder, dict]:
    """Create the configured encoder; fall back to TF-IDF when unavailable.

    Returns ``(encoder, info)`` where ``info`` records what was actually used
    so manifests never lie about the text representation.
    """
    text_cfg = config.get("text", {})
    fallback_allowed = text_cfg.get("fallback", "tfidf") is not None
    try:
        encoder = MiniLMEncoder(text_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"))
        info = {
            "encoder_id": encoder.encoder_id,
            "revision": encoder.revision,
            "dim": encoder.dim,
            "kind": encoder.kind,
            "fallback_used": False,
        }
        return encoder, info
    except Exception as exc:  # network or dependency failure
        if not fallback_allowed:
            raise
        logger.warning("MiniLM unavailable (%s); using TF-IDF fallback", exc)
        if texts_for_fit is None:
            raise RuntimeError(
                "TF-IDF fallback requires texts_for_fit to fit the vectorizer"
            )
        encoder = TfidfHashingEncoder()
        filled = [t if t.strip() else " " for t in texts_for_fit]
        encoder.fit(filled)
        info = {
            "encoder_id": encoder.encoder_id,
            "revision": None,
            "dim": encoder.dim,
            "kind": encoder.kind,
            "fallback_used": True,
        }
        return encoder, info


class EmbeddingCache:
    """Disk cache keyed by (encoder, revision, preprocessing, text hash)."""

    def __init__(self, directory: str | Path, encoder_info: dict):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.info = encoder_info

    def _path_for(self, text: str) -> Path:
        key = sha256_text(
            f"{self.info['encoder_id']}|{self.info['revision']}|{PREPROCESSING_VERSION}|{sha256_text(text)}"
        )
        return self.dir / f"{key}.npy"

    def get_or_encode(self, texts: list[str], encoder: TextEncoder, batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), encoder.dim), dtype=np.float32)
        missing: list[int] = []
        for i, text in enumerate(texts):
            path = self._path_for(text)
            if path.exists():
                out[i] = np.load(path)
            else:
                missing.append(i)
        if missing:
            logger.info("encoding %d new texts (cache hit %d)", len(missing), len(texts) - len(missing))
            encoded = encoder.encode_many([texts[i] for i in missing], batch_size=batch_size)
            for j, i in enumerate(missing):
                out[i] = encoded[j]
                np.save(self._path_for(texts[i]), encoded[j])
        return out


def encode_reviews(
    reviews: pd.DataFrame,
    encoder: TextEncoder,
    cache: EmbeddingCache | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Return a float32 [N, d] embedding matrix aligned with ``reviews`` rows.

    Reviews with missing text produce zero vectors; ``text_missing`` already
    records that state for the gate and the missingness indicators.
    """
    texts = [
        "" if bool(m) else build_encoder_text(t, b)
        for m, t, b in zip(reviews["text_missing"], reviews["title"], reviews["text"])
    ]
    if cache is not None:
        return cache.get_or_encode(texts, encoder, batch_size=batch_size)
    return encoder.encode_many(texts, batch_size=batch_size)
