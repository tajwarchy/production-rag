"""
Phase 2 verification script — SQLite.
Run from project root: python scripts/verify_sqlite.py
"""

import asyncio
import sys
sys.path.insert(0, ".")

from app.db.sqlite import (
    init_db, create_user, get_user,
    get_user_collection, create_document,
    create_job, update_job_status, get_job,
)


async def main():
    print("\n=== Phase 2 — SQLite verification ===\n")

    # 1. Init schema
    await init_db()
    print("[1] Schema initialised      ✓")

    # 2. Create user
    user = await create_user("test@example.com")
    print(f"[2] Created user: {user}")
    assert user["collection"].startswith("user_")

    # 3. Fetch user
    fetched = await get_user(user["id"])
    assert fetched["email"] == "test@example.com"
    print("[3] Fetched user            ✓")

    # 4. Get collection name
    col = await get_user_collection(user["id"])
    print(f"[4] Collection name: {col}")
    assert col == user["collection"]

    # 5. Create document
    doc = await create_document(user["id"], "paper.pdf", 204800)
    print(f"[5] Created document: {doc['id']}")

    # 6. Create job and walk through status transitions
    job_id = await create_job(doc["id"], user["id"])
    await update_job_status(job_id, "running")
    await update_job_status(job_id, "completed", chunks_upserted=42)
    job = await get_job(job_id)
    print(f"[6] Job status: {job['status']}, chunks: {job['chunks_upserted']}")
    assert job["status"] == "completed"
    assert job["chunks_upserted"] == 42

    print("\n✅ All SQLite checks passed. Ready for Phase 3.\n")


if __name__ == "__main__":
    asyncio.run(main())