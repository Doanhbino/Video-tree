import logging
from logging.handlers import RotatingFileHandler
import os
from config import settings


def setup_logging():
    """Configure logging for the application"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                os.path.join(log_dir, "app.log"),
                maxBytes=1024 * 1024,
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )