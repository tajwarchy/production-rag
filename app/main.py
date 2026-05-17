"""
FastAPI application entrypoint.
Routers for query and ingest are registered here.
DB schema is initialised on startup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.sqlite import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────
    setup_logging()
    await init_db()
    logger.info("RAG API started — version {}", get_settings().app.version)
    yield
    # ── shutdown ─────────────────────────────────────────────
    logger.info("RAG API shutting down")


def create_app() -> FastAPI:
    cfg = get_settings().app
    app = FastAPI(
        title=cfg.title,
        version=cfg.version,
        debug=cfg.debug,
        lifespan=lifespan,
    )

    from app.api.routes.ingest import router as ingest_router
    from app.api.routes.query import router as query_router
    from app.api.routes.users import router as users_router
    app.include_router(users_router, prefix="/api/v1", tags=["users"])
    app.include_router(ingest_router, prefix="/api/v1", tags=["ingest"])
    app.include_router(query_router,  prefix="/api/v1", tags=["query"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": cfg.version}

    return app


app = create_app()