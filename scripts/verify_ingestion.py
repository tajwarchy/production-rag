"""
Phase 3 verification — tests the full ingestion pipeline WITHOUT Celery.
Calls ingest_pdf() directly so you can confirm PDF → chunks → Qdrant works
before wiring up the worker.

Run: python scripts/verify_ingestion.py
Requires: a small PDF at scripts/sample.pdf  (any PDF will do)
"""

import sys
import asyncio
import uuid
sys.path.insert(0, ".")

from pathlib import Path
from app.core.config import get_settings
from app.db.sqlite import init_db, create_user, create_document, create_job, get_job
from app.db.qdrant import list_collections, delete_collection, similarity_search
from app.services.ingestion_service import ingest_pdf
from app.services.embedding_service import embed_query

SAMPLE_PDF = Path("scripts/sample.pdf")
TEST_QUERY = "What is this document about?"


async def setup_user() -> tuple[str, str, str]:
    await init_db()
    cfg = get_settings().qdrant
    user_id = str(uuid.uuid4())
    email = f"{user_id[:8]}@test.com"
    collection = f"{cfg.collection_prefix}{user_id}"

    # Insert user directly for test (bypass full create_user to avoid email unique clash)
    from app.db.sqlite import get_db
    async with get_db() as conn:
        await conn.execute(
            "INSERT INTO users (id, email, collection) VALUES (?, ?, ?)",
            (user_id, email, collection),
        )
        await conn.commit()

    doc = await create_document(user_id, SAMPLE_PDF.name, SAMPLE_PDF.stat().st_size)
    return user_id, doc["id"], collection


async def main():
    print("\n=== Phase 3 — Ingestion pipeline verification ===\n")

    if not SAMPLE_PDF.exists():
        print(f"[!] Place any PDF at {SAMPLE_PDF} and rerun.")
        sys.exit(1)

    # 1. Setup
    user_id, doc_id, collection = await setup_user()
    print(f"[1] Test user={user_id[:8]}...  collection={collection}")

    # 2. Run ingestion pipeline directly (no Celery)
    pdf_bytes = SAMPLE_PDF.read_bytes()
    chunks_upserted = ingest_pdf(
        pdf_bytes=pdf_bytes,
        user_id=user_id,
        document_id=doc_id,
        filename=SAMPLE_PDF.name,
        collection_name=collection,
    )
    print(f"[2] Ingested {chunks_upserted} chunks into Qdrant ✓")
    assert chunks_upserted > 0

    # 3. Confirm collection exists in Qdrant
    cols = list_collections()
    assert collection in cols
    print(f"[3] Collection '{collection}' visible in Qdrant ✓")

    # 4. Run a similarity search against the ingested vectors
    query_vec = embed_query(TEST_QUERY)
    results = similarity_search(
        collection_name=collection,
        query_vector=query_vec,
        top_k=3,
    )
    print(f"[4] Similarity search returned {len(results)} results")
    print(f"    Top chunk (score={results[0].score:.4f}):")
    print(f"    \"{results[0].payload['text'][:120]}...\"")
    assert len(results) > 0

    # 5. Cleanup
    delete_collection(collection)
    print("[5] Test collection cleaned up ✓")

    print("\n✅ Phase 3 ingestion pipeline verified. Ready for Phase 4.\n")


if __name__ == "__main__":
    asyncio.run(main())