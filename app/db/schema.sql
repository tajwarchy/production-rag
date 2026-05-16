-- ============================================================
-- RAG Production — SQLite metadata schema
-- Designed before any Qdrant interaction (Phase 1)
-- ============================================================

-- ------------------------------------------------------------
-- users
-- Maps each user to their isolated Qdrant collection.
-- At scale: replace with PostgreSQL users table.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,                 -- UUID string
    email       TEXT NOT NULL UNIQUE,
    collection  TEXT NOT NULL UNIQUE,             -- Qdrant collection name: "user_{id}"
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ------------------------------------------------------------
-- documents
-- Tracks every uploaded file per user.
-- One row per file. Chunks live in Qdrant — not here.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,             -- UUID string
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    file_size_bytes INTEGER,
    num_chunks      INTEGER,                      -- filled in after ingestion completes
    uploaded_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, filename)
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);

-- ------------------------------------------------------------
-- ingestion_jobs
-- Tracks the lifecycle of each ingestion task (background worker).
-- Status flow: pending → running → completed | failed
-- The query endpoint NEVER touches this table.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              TEXT PRIMARY KEY,             -- UUID string (also the Celery task ID)
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'running', 'completed', 'failed')),
    error_message   TEXT,                         -- NULL unless status = 'failed'
    chunks_upserted INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    started_at      TEXT,
    completed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_document  ON ingestion_jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user      ON ingestion_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status    ON ingestion_jobs(status);

-- ------------------------------------------------------------
-- retrieval_logs  (optional but useful for RAGAS evaluation)
-- Records every query, which strategy was used, latency, 
-- and the retrieved chunk IDs so RAGAS can pull them back.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id                  TEXT PRIMARY KEY,         -- UUID string
    user_id             TEXT NOT NULL,
    query_original      TEXT NOT NULL,
    query_rewritten     TEXT,
    retrieval_strategy  TEXT NOT NULL             -- 'similarity' | 'mmr' | 'hyde'
                            CHECK(retrieval_strategy IN ('similarity', 'mmr', 'hyde')),
    num_chunks_returned INTEGER,
    latency_ms          INTEGER,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_logs_user     ON retrieval_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_logs_strategy ON retrieval_logs(retrieval_strategy);