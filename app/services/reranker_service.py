"""
Cross-encoder reranker — runs locally on MPS.

Why rerank?
  Bi-encoder embeddings (all-MiniLM-L6-v2) retrieve quickly but score
  query-document similarity independently. A cross-encoder sees the
  query AND the document together, producing a much more accurate
  relevance score — at the cost of being too slow to run over the
  full collection. Standard pattern: retrieve top_k with bi-encoder,
  rerank with cross-encoder, pass top_k_rerank to the LLM.
"""

from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _load_reranker():
    cfg = get_settings().reranker
    device = cfg.device if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading reranker '{}' on device '{}'", cfg.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def rerank(query: str, chunks: list[str]) -> list[tuple[float, str]]:
    """
    Score each (query, chunk) pair with the cross-encoder.
    Returns chunks sorted by descending relevance score.

    Args:
        query:  the (possibly rewritten) user query
        chunks: list of retrieved text chunks

    Returns:
        List of (score, chunk_text) tuples, best first.
    """
    if not chunks:
        return []

    cfg = get_settings().reranker
    tokenizer, model, device = _load_reranker()

    pairs = [[query, chunk] for chunk in chunks]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        scores = model(**inputs).logits.squeeze(-1).tolist()

    # Handle single-item case where squeeze returns a scalar
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    logger.debug("Reranked {} chunks — top score: {:.4f}", len(chunks), ranked[0][0])
    return ranked