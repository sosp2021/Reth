import logging
import sys
from .interval import Interval
from .schedule import Schedule


def getLogger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
