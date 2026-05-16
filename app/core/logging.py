import sys
from loguru import logger
from app.core.config import get_settings

def setup_logging() -> None:
    cfg = get_settings()
    logger.remove()   # remove default handler

    level = "DEBUG" if cfg.app.debug else "INFO"

    # human-readable console output
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )

    # structured file log — rotates at 10 MB, keeps 7 days
    logger.add(
        "logs/rag.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        serialize=True,   # NDJSON — easy to ship to any log aggregator
    )