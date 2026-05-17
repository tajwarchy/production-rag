"""
Ingest routes.

POST /api/v1/ingest  — upload a PDF, enqueue background ingestion task
GET  /api/v1/jobs/{job_id} — poll ingestion job status
GET  /api/v1/documents     — list all documents for the current user

Design note:
  The handler returns HTTP 202 Accepted immediately after enqueuing the
  Celery task. The PDF bytes are passed to the worker as a hex string
  (JSON-serialisable). The client polls /jobs/{job_id} for completion.
  This is why ingestion NEVER blocks the query path.
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.deps import get_current_user_id, get_current_user_collection
from app.db.sqlite import (
    create_document, create_job, get_job, list_documents,
)
from app.worker.tasks import ingest_document

router = APIRouter()


# ------------------------------------------------------------------ #
#  Response schemas                                                    #
# ------------------------------------------------------------------ #

class IngestResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: str = "pending"
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    chunks_upserted: int
    error_message: str | None


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF for background ingestion",
)
async def ingest_pdf_route(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    collection: str = Depends(get_current_user_collection),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Create SQLite records
    doc = await create_document(user_id, file.filename, len(pdf_bytes))
    job_id = await create_job(doc["id"], user_id)

    # Enqueue Celery task — returns immediately (HTTP 202)
    ingest_document.delay(
        job_id=job_id,
        document_id=doc["id"],
        user_id=user_id,
        filename=file.filename,
        collection_name=collection,
        pdf_bytes_hex=pdf_bytes.hex(),
    )

    return IngestResponse(
        job_id=job_id,
        document_id=doc["id"],
        filename=file.filename,
        message="Ingestion queued. Poll /api/v1/jobs/{job_id} for status.",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll ingestion job status",
)
async def get_job_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    if job["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this job.",
        )
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        chunks_upserted=job["chunks_upserted"] or 0,
        error_message=job["error_message"],
    )


@router.get(
    "/documents",
    summary="List documents for the current user",
)
async def list_user_documents(user_id: str = Depends(get_current_user_id)):
    docs = await list_documents(user_id)
    return {"user_id": user_id, "documents": docs}