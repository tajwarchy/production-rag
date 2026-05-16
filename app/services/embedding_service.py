"""
Embedding service — all-MiniLM-L6-v2 on MPS (M1).
Singleton model load: loaded once, reused across all calls.
num_workers=0 is required on MPS to avoid multiprocessing issues.
"""

from functools import lru_cache
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    cfg = get_settings().embedding
    device = cfg.device if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading embedding model '{}' on device '{}'", cfg.model_name, device)
    model = SentenceTransformer(cfg.model_name, device=device)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings. Returns a list of float vectors.
    batch_size and num_workers come from config — never hardcoded.
    """
    cfg = get_settings().embedding
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,   # 0 on MPS — required
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,     # unit vectors → cosine = dot product
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Single query embedding — used at retrieval time."""
    return embed_texts([text])[0]