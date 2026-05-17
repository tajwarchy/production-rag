"""
Phase 4 verification — runs all 3 retrieval strategies against an ingested PDF.

Run: python scripts/verify_retrieval.py
Requires:
  - Docker running (Qdrant + Redis)
  - Ollama running with Mistral pulled: ollama pull mistral
  - A PDF at scripts/sample.pdf (same one used in Phase 3)
"""

import sys
import asyncio
import uuid
sys.path.insert(0, ".")

from pathlib import Path
from app.db.sqlite import init_db, get_db
from app.db.qdrant import delete_collection
from app.services.ingestion_service import ingest_pdf
from app.services.retrieval_service import retrieve_and_answer
from app.core.config import get_settings

SAMPLE_PDF = Path("scripts/sample.pdf")
QUESTION = "What are the main topics covered in this document?"
STRATEGIES = ["similarity", "mmr", "hyde"]


async def setup_test_user():
    await init_db()
    cfg = get_settings().qdrant
    user_id = str(uuid.uuid4())
    collection = f"{cfg.collection_prefix}{user_id}"
    async with get_db() as conn:
        await conn.execute(
            "INSERT INTO users (id, email, collection) VALUES (?, ?, ?)",
            (user_id, f"{user_id[:8]}@verify.com", collection),
        )
        await conn.commit()
    return user_id, collection


async def main():
    print("\n=== Phase 4 — Retrieval strategies verification ===\n")

    if not SAMPLE_PDF.exists():
        print(f"[!] PDF not found at {SAMPLE_PDF}")
        sys.exit(1)

    # Setup
    user_id, collection = await setup_test_user()
    doc_id = str(uuid.uuid4())

    # Ingest
    print(f"[*] Ingesting '{SAMPLE_PDF.name}' into collection '{collection}'...")
    pdf_bytes = SAMPLE_PDF.read_bytes()
    n = ingest_pdf(pdf_bytes, user_id, doc_id, SAMPLE_PDF.name, collection)
    print(f"[*] Ingested {n} chunks\n")

    # Run all 3 strategies
    print(f"Question: \"{QUESTION}\"\n")
    print("=" * 70)

    for strategy in STRATEGIES:
        print(f"\n▶ Strategy: {strategy.upper()}")
        print("-" * 70)
        result = retrieve_and_answer(
            question=QUESTION,
            collection=collection,
            strategy=strategy,
        )
        print(f"  Rewritten query : {result['rewritten_query']}")
        print(f"  Chunks used     : {len(result['chunks_used'])}")
        print(f"  Answer          :\n\n  {result['answer']}\n")
        print("-" * 70)

    # Cleanup
    delete_collection(collection)
    print("\n[*] Test collection cleaned up")
    print("\n✅ Phase 4 verified. All 3 strategies working. Ready for Phase 5.\n")


if __name__ == "__main__":
    asyncio.run(main())