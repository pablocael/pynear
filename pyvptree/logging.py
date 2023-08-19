import logging
import os


def create_and_configure_log(name):
    logger = logging.getLogger(name)
    logger.setLevel("INFO")
    level_envar = "PYVPTREE_LOG_LEVEL"
    if level_envar in os.environ:
        logger.setLevel(os.environ[level_envar])

    # TODO: proper logging configuration
    formatter = logging.Formatter("%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
