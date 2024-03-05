import logging

NO_FILE_LOGGER = "__NO__"
BIBIS_LOGGER_NAME = "bibis"
BIBIS_LOGGING_PATH = "bibis.log"

def get_logger(name: str, path: str | None = None) -> logging.Logger:
    if path is None:
        path = f"{name}.log"

    logger = logging.getLogger(name)
    
    stream_handler = logging.StreamHandler()
    stream_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)

    if path != NO_FILE_LOGGER:
        file_handler = logging.FileHandler(path)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

def get_bibis_logger() -> logging.Logger:
    return get_logger(name=BIBIS_LOGGER_NAME, path=BIBIS_LOGGER_NAME)