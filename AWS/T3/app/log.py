import os
import logging
from datetime import datetime, timedelta
from rich.logging import RichHandler

RICH_FORMAT = "[%(filename)s:%(lineno)s] >> %(message)s"
FILE_HANDLER_FORMAT = "[%(asctime)s] %(levelname)s [%(filename)s:%(funcName)s:%(lineno)s] >> %(message)s"

LOG_PATH = os.path.join(os.getcwd(), "app", "logs")

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=RICH_FORMAT,
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger = logging.getLogger(os.path.join(LOG_PATH , f"{datetime.now().year}{datetime.now().month}{datetime.now().day}_log.txt"))

    if len(logger.handlers) > 0:
        return logger

    file_handler = logging.FileHandler(os.path.join(LOG_PATH , f"{datetime.now().year}{datetime.now().month}{datetime.now().day}_log.txt"), mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    logger.addHandler(file_handler)

    return logger
              
def handle_exception(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger()
    logger.error("Unexpected exception", exc_info=(exc_type, exc_value, exc_traceback))


