"""
Celery tasks — ingestion worker.

Design note (for README):
  Ingestion is a background task and NOT part of the API handler because:
  - PDF processing + embedding can take 30–120 seconds for large documents.
  - Blocking the API handler would tie up a server thread/process for that
    entire duration, killing throughput for all other users.
  - The worker runs in a separate process, so a crash during ingestion
    never affects the query API.
  - The SQLite job table is the coordination layer: the API returns a job_id
    immediately (HTTP 202), and the client polls /jobs/{job_id} for status.
"""

import asyncio
from loguru import logger

from app.worker.celery_app import celery_app
from app.services.ingestion_service import ingest_pdf

# SQLite helpers are async — we run them in a tiny event loop
# inside the synchronous Celery task.
from app.db.sqlite import update_job_status, update_document_chunks


def _run(coro):
    """Run an async function from sync Celery task context."""
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(
    bind=True,
    name="tasks.ingest_document",
    max_retries=2,
    default_retry_delay=30,
)
def ingest_document(
    self,
    job_id: str,
    document_id: str,
    user_id: str,
    filename: str,
    collection_name: str,
    pdf_bytes_hex: str,         # bytes serialised as hex for JSON transport
) -> dict:
    """
    Background task: PDF → chunks → embeddings → Qdrant.

    Args:
        job_id:          SQLite ingestion_jobs primary key
        document_id:     SQLite documents primary key
        user_id:         owner of this document
        filename:        original filename (stored in Qdrant payload)
        collection_name: user's Qdrant collection
        pdf_bytes_hex:   PDF content as a hex string (JSON-serialisable)

    Returns:
        dict with job_id and chunks_upserted
    """
    logger.info("Task started: job={} doc={} user={}", job_id, document_id, user_id)

    # Mark job as running
    _run(update_job_status(job_id, "running"))

    try:
        pdf_bytes = bytes.fromhex(pdf_bytes_hex)

        chunks_upserted = ingest_pdf(
            pdf_bytes=pdf_bytes,
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            collection_name=collection_name,
        )

        # Update SQLite: job completed, document chunk count known
        _run(update_job_status(job_id, "completed", chunks_upserted=chunks_upserted))
        _run(update_document_chunks(document_id, chunks_upserted))

        logger.info("Task complete: job={} chunks={}", job_id, chunks_upserted)
        return {"job_id": job_id, "chunks_upserted": chunks_upserted}

    except Exception as exc:
        logger.error("Task failed: job={} error={}", job_id, exc)
        _run(update_job_status(job_id, "failed", error_message=str(exc)))

        # Retry up to max_retries times before giving up
        raise self.retry(exc=exc)