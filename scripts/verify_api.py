"""
End-to-end API smoke test.
Tests the full flow: register user → upload PDF → poll job → query.

Requires:
  - FastAPI running:  uvicorn app.main:app --reload
  - Celery running:   celery -A app.worker.celery_app.celery_app worker --loglevel=info --concurrency=1
  - Qdrant + Redis:   docker-compose up -d
  - Ollama running with mistral pulled

Run: python scripts/verify_api.py
"""

import sys
import time
import httpx
from pathlib import Path

BASE_URL   = "http://localhost:8001/api/v1"
SAMPLE_PDF = Path("scripts/sample.pdf")
QUESTION   = "What are the main topics covered in this document?"


def step(n: int, msg: str):
    print(f"\n[{n}] {msg}")


def main():
    print("\n=== Full API smoke test ===\n")
    client = httpx.Client(timeout=120)

    # ── 1. Health check ───────────────────────────────────────────
    step(1, "Health check")
    r = client.get("http://localhost:8001/health")
    assert r.status_code == 200, r.text
    print(f"    {r.json()}")

    # ── 2. Register user ──────────────────────────────────────────
    step(2, "Register user")
    r = client.post(f"{BASE_URL}/users", json={"email": "smoketest@example.com"})
    if r.status_code == 409:
        print("    User already exists — fetching existing user not supported by this script.")
        print("    Delete data/metadata.db and rerun, or use a different email.")
        sys.exit(1)
    assert r.status_code == 201, r.text
    user = r.json()
    user_id = user["id"]
    print(f"    user_id:    {user_id}")
    print(f"    collection: {user['collection']}")

    headers = {"X-User-Id": user_id}

    # ── 3. Upload PDF ─────────────────────────────────────────────
    step(3, f"Upload PDF: {SAMPLE_PDF.name}")
    assert SAMPLE_PDF.exists(), f"PDF not found at {SAMPLE_PDF}"
    with open(SAMPLE_PDF, "rb") as f:
        r = client.post(
            f"{BASE_URL}/ingest",
            headers=headers,
            files={"file": (SAMPLE_PDF.name, f, "application/pdf")},
        )
    assert r.status_code == 202, r.text
    ingest_resp = r.json()
    job_id = ingest_resp["job_id"]
    print(f"    job_id: {job_id}")
    print(f"    status: {ingest_resp['status']}")

    # ── 4. Poll job until complete ────────────────────────────────
    step(4, "Polling ingestion job...")
    for attempt in range(30):
        time.sleep(5)
        r = client.get(f"{BASE_URL}/jobs/{job_id}", headers=headers)
        assert r.status_code == 200, r.text
        job = r.json()
        print(f"    attempt {attempt+1:02d}: status={job['status']} chunks={job['chunks_upserted']}")
        if job["status"] in ("completed", "failed"):
            break

    assert job["status"] == "completed", f"Job failed: {job.get('error_message')}"
    print(f"    ✓ Ingestion complete — {job['chunks_upserted']} chunks")

    # ── 5. Query — all 3 strategies ───────────────────────────────
    for strategy in ["similarity", "mmr", "hyde"]:
        step(5, f"Query — strategy={strategy}")
        r = client.post(
            f"{BASE_URL}/query",
            headers=headers,
            json={"question": QUESTION, "strategy": strategy},
        )
        assert r.status_code == 200, r.text
        resp = r.json()
        print(f"    latency_ms:      {resp['latency_ms']}")
        print(f"    rewritten_query: {resp['rewritten_query']}")
        print(f"    chunks_used:     {len(resp['chunks_used'])}")
        print(f"    answer:          {resp['answer'][:120]}...")

    # ── 6. List documents ─────────────────────────────────────────
    step(6, "List documents")
    r = client.get(f"{BASE_URL}/documents", headers=headers)
    assert r.status_code == 200, r.text
    docs = r.json()
    print(f"    {len(docs['documents'])} document(s) for user")

    print("\n✅ Full API smoke test passed.\n")


if __name__ == "__main__":
    main()