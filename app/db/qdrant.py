"""
Qdrant client wrapper.
All Qdrant interactions go through this module — nothing else imports
qdrant_client directly. Keeps the blast radius small if the client API changes.

CAP theorem note (for README):
  Qdrant is CP under partition — it prioritises consistency over availability.
  During a network split it will refuse writes rather than risk divergent state.
  Our graceful fallback in retrieval_service.py handles the availability gap.
"""

from functools import lru_cache
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings


# ------------------------------------------------------------------ #
#  Singleton client                                                    #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    cfg = get_settings().qdrant
    client = QdrantClient(
        host=cfg.host,
        port=cfg.port,
        grpc_port=cfg.grpc_port,
        prefer_grpc=False,            # REST is fine for local dev
        timeout=cfg.timeout_seconds,
    )
    logger.info("Qdrant client initialised → {}:{}", cfg.host, cfg.port)
    return client


# ------------------------------------------------------------------ #
#  Health check                                                        #
# ------------------------------------------------------------------ #

def is_qdrant_healthy() -> bool:
    """
    Used by the graceful fallback in retrieval_service.py.
    Returns False if Qdrant is unreachable — never raises.
    """
    try:
        get_qdrant_client().get_collections()
        return True
    except Exception as exc:
        logger.warning("Qdrant health check failed: {}", exc)
        return False


# ------------------------------------------------------------------ #
#  Collection management                                               #
# ------------------------------------------------------------------ #

def create_collection(collection_name: str) -> None:
    """
    Creates a Qdrant collection for a user.
    Idempotent — safe to call if the collection already exists.
    """
    cfg = get_settings().qdrant
    client = get_qdrant_client()

    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        logger.info("Collection '{}' already exists — skipping create", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=cfg.vector_size,
            distance=qmodels.Distance[cfg.distance.upper()],
        ),
    )
    logger.info("Created Qdrant collection '{}'", collection_name)


def delete_collection(collection_name: str) -> None:
    get_qdrant_client().delete_collection(collection_name)
    logger.info("Deleted Qdrant collection '{}'", collection_name)


def list_collections() -> list[str]:
    return [c.name for c in get_qdrant_client().get_collections().collections]


# ------------------------------------------------------------------ #
#  Upsert                                                              #
# ------------------------------------------------------------------ #

def upsert_vectors(
    collection_name: str,
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict],
) -> None:
    """
    Upsert a batch of vectors with payload metadata.
    Payload example:
      {
        "document_id": "abc-123",
        "user_id":     "usr-456",
        "filename":    "paper.pdf",
        "chunk_index": 3,
        "text":        "...chunk text...",
      }
    """
    client = get_qdrant_client()
    points = [
        qmodels.PointStruct(id=pid, vector=vec, payload=pl)
        for pid, vec, pl in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)
    logger.debug("Upserted {} vectors into '{}'", len(points), collection_name)


# ------------------------------------------------------------------ #
#  Search                                                              #
# ------------------------------------------------------------------ #

def similarity_search(
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    score_threshold: float = 0.0,
    payload_filter: Optional[qmodels.Filter] = None,
) -> list[qmodels.ScoredPoint]:
    """Standard cosine similarity search with optional payload filter."""
    return get_qdrant_client().search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=payload_filter,
        with_payload=True,
    )


def scroll_all(collection_name: str, batch_size: int = 100) -> list[qmodels.Record]:
    """Return all points in a collection (used for MMR candidate pool)."""
    client = get_qdrant_client()
    records, next_offset = client.scroll(
        collection_name=collection_name,
        limit=batch_size,
        with_payload=True,
        with_vectors=True,
    )
    all_records = list(records)
    while next_offset is not None:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        all_records.extend(records)
    return all_records