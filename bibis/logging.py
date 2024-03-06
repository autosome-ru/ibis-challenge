import logging
from pyclbr import Class
from typing import ClassVar
from dataclasses import dataclass

NO_FILE_LOGGER = "__NO__"
BIBIS_DEFAULT_LOGGING_PATH = NO_FILE_LOGGER

def add_stream_handler(logger: logging.Logger) -> logging.StreamHandler:
    stream_handler = logging.StreamHandler()
    stream_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)
    return stream_handler

def add_file_handler(logger: logging.Logger, path: str) -> logging.FileHandler | None:
    if path != NO_FILE_LOGGER:
        file_handler = logging.FileHandler(path)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        return file_handler
    return None

def get_logger(name: str, path: str | None = None, level: int = logging.INFO) -> logging.Logger:
    if path is None:
        path = f"{name}.log"

    logger = logging.getLogger(name)
    add_stream_handler(logger)
    add_file_handler(logger, path=path)
    
    logger.setLevel(level)
    return logger

@dataclass
class BIBIS_LOGGER_CFG:
    NAME: ClassVar[str] = "bibis"
    PATH: ClassVar[str] = NO_FILE_LOGGER
    LEVEL: ClassVar[int] = logging.INFO
    FILE_HANDLER: ClassVar[logging.FileHandler | None] = None
    STREAM_HANDLER: ClassVar[logging.FileHandler | None] = None
    INITIALIZED: ClassVar[bool] = False

    @classmethod
    def set_path(cls, path: str):
        if path != cls.PATH:
            logger = cls.get_logger()
            if cls.FILE_HANDLER is not None:
                logger.removeHandler(cls.FILE_HANDLER)
            
            cls.PATH = path
            cls._add_file_handler(logger)

    @classmethod
    def set_level(cls, level: int):
        cls.LEVEL = level
        logger = cls.get_logger()
        logger.setLevel(level)

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if not cls.INITIALIZED:
            cls.INITIALIZED = True
            cls._init_bibis_logger()
        return logging.getLogger(cls.NAME)

    @classmethod
    def _add_stream_handler(cls, logger):
        stream_handler = add_stream_handler(logger)
        cls.STREAM_HANDLER = stream_handler
    
    @classmethod
    def _add_file_handler(cls, logger):
        file_handler = add_file_handler(logger, cls.PATH)
        cls.FILE_HANDLER = file_handler
        logger.setLevel(cls.LEVEL)
    
    @classmethod
    def _init_bibis_logger(cls):
        logger = logging.getLogger(cls.NAME)
        cls._add_stream_handler(logger)
        cls._add_file_handler(logger)   

def get_bibis_logger() -> logging.Logger:
    return BIBIS_LOGGER_CFG.get_logger()