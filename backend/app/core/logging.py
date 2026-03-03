import sys
import os
from loguru import logger


def setup_logging(log_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )
    return logger
