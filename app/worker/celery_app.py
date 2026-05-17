"""
Celery application instance.
Import this in tasks.py and in FastAPI to enqueue tasks.
All config comes from config.yaml — never hardcoded here.
"""

from celery import Celery
from app.core.config import get_settings


def create_celery_app() -> Celery:
    cfg = get_settings().worker

    app = Celery(
        "rag_worker",
        broker=cfg.broker_url,
        backend=cfg.result_backend,
        include=["app.worker.tasks"],
    )

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_time_limit=cfg.task_time_limit,
        task_soft_time_limit=cfg.task_soft_time_limit,
        worker_prefetch_multiplier=1,   # one task at a time — embedding is heavy
        worker_pool="solo",             # M1 MPS fix: no forking, runs in-process
        task_acks_late=True,            # ack only after task completes (safer)
        timezone="UTC",
        broker_connection_retry_on_startup=True,
    )

    return app


celery_app = create_celery_app()