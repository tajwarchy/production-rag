"""
Phase 2 verification script.
Run from project root: python scripts/verify_qdrant.py
Confirms: client connects, collection CRUD works, vector search returns results.
"""

import sys
import uuid
import random
sys.path.insert(0, ".")

from app.core.config import get_settings
from app.db.qdrant import (
    create_collection, delete_collection,
    list_collections, upsert_vectors, similarity_search,
    is_qdrant_healthy,
)

cfg = get_settings()
TEST_COLLECTION = "phase2_verify_test"
DIM = cfg.qdrant.vector_size  # 384


def rand_vec() -> list[float]:
    return [random.uniform(-1, 1) for _ in range(DIM)]


def main():
    print("\n=== Phase 2 — Qdrant verification ===\n")

    # 1. Health
    healthy = is_qdrant_healthy()
    print(f"[1] Qdrant healthy:        {healthy}")
    assert healthy, "Qdrant is not reachable. Is docker-compose up?"

    # 2. Create collection
    create_collection(TEST_COLLECTION)
    cols = list_collections()
    print(f"[2] Collections after create: {cols}")
    assert TEST_COLLECTION in cols

    # 3. Upsert 5 vectors with payload
    ids      = [str(uuid.uuid4()) for _ in range(5)]
    vectors  = [rand_vec() for _ in range(5)]
    payloads = [
        {"document_id": "doc-1", "user_id": "usr-1",
         "filename": "test.pdf", "chunk_index": i, "text": f"chunk {i}"}
        for i in range(5)
    ]
    upsert_vectors(TEST_COLLECTION, ids, vectors, payloads)
    print("[3] Upserted 5 vectors      ✓")

    # 4. Similarity search
    results = similarity_search(
        collection_name=TEST_COLLECTION,
        query_vector=vectors[0],     # searching with the exact first vector
        top_k=3,
    )
    print(f"[4] Search returned {len(results)} results — top id: {results[0].id}, score: {results[0].score:.4f}")
    assert len(results) > 0

    # 5. Cleanup
    delete_collection(TEST_COLLECTION)
    assert TEST_COLLECTION not in list_collections()
    print("[5] Collection deleted      ✓")

    print("\n✅ All Phase 2 checks passed. Ready for Phase 3.\n")


if __name__ == "__main__":
    main()