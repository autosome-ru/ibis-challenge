import logging
from pyclbr import Class
from typing import ClassVar
from dataclasses import dataclass

NO_FILE_LOGGER = "__NO__"
BIBIS_DEFAULT_LOGGING_PATH = NO_FILE_LOGGER

def get_logger(name: str, path: str | None = None, level: int = logging.INFO) -> logging.Logger:
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
    logger.setLevel(level)
    return logger

@dataclass
class BIBIS_LOGGER_CFG:
    NAME: ClassVar[str] = "bibis"
    PATH: ClassVar[str] = NO_FILE_LOGGER
    LEVEL: ClassVar[int] = logging.INFO

    @classmethod
    def set_name(cls, name: str):
        cls.NAME = name
    
    @classmethod
    def set_path(cls, path: str):
        cls.NAME = path

def get_bibis_logger():
    return get_logger(name=BIBIS_LOGGER_CFG.NAME, 
                      path=BIBIS_LOGGER_CFG.PATH,
                      level=BIBIS_LOGGER_CFG.LEVEL)