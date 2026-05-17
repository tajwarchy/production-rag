"""
Retrieval service — three strategies + reranking + graceful fallback.

Strategies:
  1. similarity — standard cosine nearest-neighbour search
  2. mmr        — Maximal Marginal Relevance: balances relevance vs diversity
  3. hyde       — HyDE: embed a *hypothetical* answer, search with that vector

All strategies feed into the same reranking step before hitting the LLM.

CAP theorem / graceful fallback:
  Qdrant is CP — under a partition it refuses writes/reads rather than
  serving stale data. If Qdrant is unreachable, is_qdrant_healthy()
  returns False and we return a safe fallback response immediately,
  rather than letting the request hang or crash.
"""

import numpy as np
from loguru import logger

from app.core.config import get_settings
from app.db.qdrant import (
    is_qdrant_healthy,
    similarity_search,
    scroll_all,
)
from app.services.embedding_service import embed_query, embed_texts
from app.services.reranker_service import rerank
from app.services.query_rewriter import rewrite_query
from app.services.llm_service import get_llm, generate_answer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ------------------------------------------------------------------ #
#  Fallback response                                                   #
# ------------------------------------------------------------------ #

FALLBACK_RESPONSE = (
    "The retrieval service is temporarily unavailable. "
    "Please try again in a moment."
)


# ------------------------------------------------------------------ #
#  Strategy implementations                                            #
# ------------------------------------------------------------------ #

def _retrieve_similarity(
    collection: str,
    query_vector: list[float],
    top_k: int,
    score_threshold: float,
) -> list[str]:
    """Standard cosine similarity — fastest, simplest."""
    results = similarity_search(
        collection_name=collection,
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return [r.payload["text"] for r in results]


def _retrieve_mmr(
    collection: str,
    query_vector: list[float],
    top_k: int,
    mmr_lambda: float,
) -> list[str]:
    """
    Maximal Marginal Relevance — reduces redundancy in retrieved chunks.
    Iteratively picks the chunk that maximises:
        score = λ * relevance_to_query - (1-λ) * max_similarity_to_selected

    mmr_lambda=1.0 → pure relevance (same as similarity search)
    mmr_lambda=0.0 → pure diversity
    mmr_lambda=0.5 → balanced (default in config.yaml)
    """
    # Get all candidates from the collection
    records = scroll_all(collection)
    if not records:
        return []

    candidate_vecs = np.array([r.vector for r in records], dtype=np.float32)
    candidate_texts = [r.payload["text"] for r in records]
    q = np.array(query_vector, dtype=np.float32)

    # Relevance scores: cosine similarity to query (vectors are unit-normalised)
    relevance = candidate_vecs @ q

    selected_indices = []
    remaining = list(range(len(records)))

    for _ in range(min(top_k, len(records))):
        if not remaining:
            break

        if not selected_indices:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: relevance[i])
        else:
            selected_vecs = candidate_vecs[selected_indices]
            scores = []
            for i in remaining:
                rel = relevance[i]
                # Max similarity to any already-selected chunk
                redundancy = float(np.max(candidate_vecs[i] @ selected_vecs.T))
                mmr_score = mmr_lambda * rel - (1 - mmr_lambda) * redundancy
                scores.append((mmr_score, i))
            best = max(scores, key=lambda x: x[0])[1]

        selected_indices.append(best)
        remaining.remove(best)

    return [candidate_texts[i] for i in selected_indices]


HYDE_PROMPT = PromptTemplate.from_template("""
Write a short, factual passage (2-3 sentences) that would directly answer
the following question. Write as if it were an excerpt from a document.

Question: {question}

Passage:""")


def _retrieve_hyde(
    collection: str,
    query: str,
    top_k: int,
    score_threshold: float,
) -> list[str]:
    """
    HyDE — Hypothetical Document Embeddings.
    Instead of embedding the raw query, we ask the LLM to generate a
    hypothetical answer, then embed THAT. The intuition: a hypothetical
    answer lives in the same vector space as real document chunks,
    so retrieval is more accurate than embedding the question directly.
    """
    chain = HYDE_PROMPT | get_llm() | StrOutputParser()
    hypothetical_doc = chain.invoke({"question": query}).strip()
    logger.debug("HyDE hypothetical doc: '{}'", hypothetical_doc[:120])

    hyde_vector = embed_query(hypothetical_doc)
    results = similarity_search(
        collection_name=collection,
        query_vector=hyde_vector,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return [r.payload["text"] for r in results]


# ------------------------------------------------------------------ #
#  Main entry point                                                    #
# ------------------------------------------------------------------ #

def retrieve_and_answer(
    question: str,
    collection: str,
    strategy: str | None = None,
) -> dict:
    """
    Full RAG pipeline:
      1. Check Qdrant health (graceful fallback if down)
      2. Rewrite query
      3. Retrieve chunks with selected strategy
      4. Rerank with cross-encoder
      5. Generate answer with LLM

    Args:
        question:   raw user question
        collection: user's Qdrant collection name
        strategy:   "similarity" | "mmr" | "hyde" — defaults to config value

    Returns:
        dict with answer, rewritten_query, strategy, chunks_used
    """
    # ── 1. Graceful fallback ──────────────────────────────────────
    if not is_qdrant_healthy():
        logger.warning("Qdrant unreachable — returning fallback response")
        return {
            "answer": FALLBACK_RESPONSE,
            "rewritten_query": question,
            "strategy": strategy or "unknown",
            "chunks_used": [],
            "fallback": True,
        }

    cfg = get_settings()
    strategy = strategy or cfg.retrieval.strategy
    top_k = cfg.retrieval.top_k
    top_k_rerank = cfg.retrieval.top_k_rerank
    score_threshold = cfg.retrieval.score_threshold
    mmr_lambda = cfg.retrieval.mmr_lambda

    # ── 2. Query rewriting ────────────────────────────────────────
    rewritten = rewrite_query(question)

    # ── 3. Retrieval ──────────────────────────────────────────────
    if strategy == "similarity":
        query_vec = embed_query(rewritten)
        chunks = _retrieve_similarity(collection, query_vec, top_k, score_threshold)

    elif strategy == "mmr":
        query_vec = embed_query(rewritten)
        chunks = _retrieve_mmr(collection, query_vec, top_k, mmr_lambda)

    elif strategy == "hyde":
        chunks = _retrieve_hyde(collection, rewritten, top_k, score_threshold)

    else:
        raise ValueError(f"Unknown retrieval strategy '{strategy}'. "
                         "Choose from: similarity, mmr, hyde")

    if not chunks:
        return {
            "answer": "No relevant documents found for your query.",
            "rewritten_query": rewritten,
            "strategy": strategy,
            "chunks_used": [],
            "fallback": False,
        }

    # ── 4. Rerank ─────────────────────────────────────────────────
    ranked = rerank(rewritten, chunks)
    top_chunks = [text for _, text in ranked[:top_k_rerank]]

    # ── 5. Generate answer ────────────────────────────────────────
    answer = generate_answer(rewritten, top_chunks)

    logger.info(
        "RAG complete: strategy={} chunks_retrieved={} chunks_to_llm={}",
        strategy, len(chunks), len(top_chunks),
    )

    return {
        "answer": answer,
        "rewritten_query": rewritten,
        "strategy": strategy,
        "chunks_used": top_chunks,
        "fallback": False,
    }