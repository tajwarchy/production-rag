"""
SQLite async interface (aiosqlite).
Handles connection, schema init, and all CRUD used across the app.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite
from loguru import logger

from app.core.config import get_settings

# ------------------------------------------------------------------ #
#  Connection                                                          #
# ------------------------------------------------------------------ #

def _db_path() -> Path:
    p = Path(get_settings().database.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(_db_path()) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
        await conn.execute("PRAGMA foreign_keys=ON")
        yield conn


# ------------------------------------------------------------------ #
#  Schema init — called once at startup                               #
# ------------------------------------------------------------------ #

async def init_db() -> None:
    schema = (Path(__file__).parent / "schema.sql").read_text()
    async with get_db() as conn:
        await conn.executescript(schema)
        await conn.commit()
    logger.info("SQLite schema initialised at {}", _db_path())


# ------------------------------------------------------------------ #
#  Users                                                               #
# ------------------------------------------------------------------ #

async def create_user(email: str) -> dict:
    user_id = str(uuid.uuid4())
    cfg = get_settings()
    collection = f"{cfg.qdrant.collection_prefix}{user_id}"
    async with get_db() as conn:
        await conn.execute(
            "INSERT INTO users (id, email, collection) VALUES (?, ?, ?)",
            (user_id, email, collection),
        )
        await conn.commit()
    return {"id": user_id, "email": email, "collection": collection}


async def get_user(user_id: str) -> dict | None:
    async with get_db() as conn:
        async with conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def get_user_collection(user_id: str) -> str | None:
    """Returns the Qdrant collection name for this user, or None."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT collection FROM users WHERE id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row["collection"] if row else None


# ------------------------------------------------------------------ #
#  Documents                                                           #
# ------------------------------------------------------------------ #

async def create_document(user_id: str, filename: str, file_size: int) -> dict:
    doc_id = str(uuid.uuid4())
    async with get_db() as conn:
        await conn.execute(
            """INSERT INTO documents (id, user_id, filename, file_size_bytes)
               VALUES (?, ?, ?, ?)""",
            (doc_id, user_id, filename, file_size),
        )
        await conn.commit()
    return {"id": doc_id, "user_id": user_id, "filename": filename}


async def update_document_chunks(doc_id: str, num_chunks: int) -> None:
    async with get_db() as conn:
        await conn.execute(
            "UPDATE documents SET num_chunks = ? WHERE id = ?",
            (num_chunks, doc_id),
        )
        await conn.commit()


async def list_documents(user_id: str) -> list[dict]:
    async with get_db() as conn:
        async with conn.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ #
#  Ingestion jobs                                                      #
# ------------------------------------------------------------------ #

async def create_job(document_id: str, user_id: str) -> str:
    job_id = str(uuid.uuid4())
    async with get_db() as conn:
        await conn.execute(
            """INSERT INTO ingestion_jobs (id, document_id, user_id)
               VALUES (?, ?, ?)""",
            (job_id, document_id, user_id),
        )
        await conn.commit()
    return job_id


async def update_job_status(
    job_id: str,
    status: str,
    chunks_upserted: int = 0,
    error_message: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as conn:
        if status == "running":
            await conn.execute(
                "UPDATE ingestion_jobs SET status=?, started_at=? WHERE id=?",
                (status, now, job_id),
            )
        elif status in ("completed", "failed"):
            await conn.execute(
                """UPDATE ingestion_jobs
                   SET status=?, completed_at=?, chunks_upserted=?, error_message=?
                   WHERE id=?""",
                (status, now, chunks_upserted, error_message, job_id),
            )
        await conn.commit()


async def get_job(job_id: str) -> dict | None:
    async with get_db() as conn:
        async with conn.execute(
            "SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------------ #
#  Retrieval logs                                                      #
# ------------------------------------------------------------------ #

async def log_retrieval(
    user_id: str,
    query_original: str,
    strategy: str,
    query_rewritten: str | None = None,
    num_chunks: int = 0,
    latency_ms: int = 0,
) -> None:
    log_id = str(uuid.uuid4())
    async with get_db() as conn:
        await conn.execute(
            """INSERT INTO retrieval_logs
               (id, user_id, query_original, query_rewritten,
                retrieval_strategy, num_chunks_returned, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (log_id, user_id, query_original, query_rewritten,
             strategy, num_chunks, latency_ms),
        )
        await conn.commit()