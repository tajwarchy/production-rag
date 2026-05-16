"""
Ingestion pipeline — pure functions.
Flow: PDF bytes → extract text → chunk → embed → upsert into Qdrant.
Called by the Celery worker task, not by any API handler directly.
"""

import uuid
from pathlib import Path

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from app.core.config import get_settings
from app.db.qdrant import create_collection, upsert_vectors
from app.services.embedding_service import embed_texts


# ------------------------------------------------------------------ #
#  PDF extraction                                                      #
# ------------------------------------------------------------------ #

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    doc.close()
    full_text = "\n\n".join(pages)
    logger.debug("Extracted {} chars from PDF ({} pages)", len(full_text), len(pages))
    return full_text


# ------------------------------------------------------------------ #
#  Chunking                                                            #
# ------------------------------------------------------------------ #

def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter. All params from config.yaml.
    """
    cfg = get_settings().chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.debug("Split into {} chunks (size={}, overlap={})",
                 len(chunks), cfg.chunk_size, cfg.chunk_overlap)
    return chunks


# ------------------------------------------------------------------ #
#  Full pipeline                                                       #
# ------------------------------------------------------------------ #

def ingest_pdf(
    pdf_bytes: bytes,
    user_id: str,
    document_id: str,
    filename: str,
    collection_name: str,
) -> int:
    """
    End-to-end ingestion for one PDF.
    Returns the number of chunks upserted.

    Steps:
      1. Ensure the user's Qdrant collection exists
      2. Extract text from PDF
      3. Chunk the text
      4. Embed all chunks (batched, on MPS)
      5. Upsert vectors with payload metadata into Qdrant

    This function is synchronous — it runs inside the Celery worker process,
    not inside an async event loop.
    """
    logger.info("Starting ingestion: doc={} user={} file={}", document_id, user_id, filename)

    # 1. Ensure collection exists (idempotent)
    create_collection(collection_name)

    # 2. Extract
    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise ValueError(f"No extractable text found in '{filename}'")

    # 3. Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"Chunking produced zero chunks for '{filename}'")

    # 4. Embed
    logger.info("Embedding {} chunks...", len(chunks))
    vectors = embed_texts(chunks)

    # 5. Build payloads and upsert
    ids = [str(uuid.uuid4()) for _ in chunks]
    payloads = [
        {
            "document_id": document_id,
            "user_id":     user_id,
            "filename":    filename,
            "chunk_index": i,
            "text":        chunk,        # stored in payload for retrieval
        }
        for i, chunk in enumerate(chunks)
    ]

    upsert_vectors(collection_name, ids, vectors, payloads)
    logger.info("Ingestion complete: {} chunks upserted for doc={}", len(chunks), document_id)

    return len(chunks)