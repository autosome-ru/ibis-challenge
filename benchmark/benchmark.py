import json
from dataset import DatasetInfo
from pathlib import Path
from attrs import define
from enum import Enum 
from typing import ClassVar, List
from scorer import ScorerInfo
from exceptions import BenchmarkConfigException
from utils import register_enum

@register_enum
class BenchmarkMode(Enum):
    USER = 1
    ADMIN = 2

@define
class BenchmarkConfig:
    name: str
    datasets: List[DatasetInfo]
    scorers: List[ScorerInfo]
    metainfo: dict 
    
    NAME_FIELD: ClassVar[str] = 'name'
    DATASETS_FIELD: ClassVar[str] = 'datasets'
    SCORERS_FIELD: ClassVar[str] = 'scorers'

    @classmethod
    def validate_benchmark_dict(cls, dt: dict):
        if not cls.NAME_FIELD in dt:
            raise BenchmarkConfigException(
                    f"Benchmark config must has field '{cls.NAME_FIELD}'")
        if not cls.DATASETS_FIELD in dt:
            raise BenchmarkConfigException("No information about datasets found")
        if not cls.SCORERS_FIELD in dt:
            raise BenchmarkConfigException("No information about scorers found")

    @classmethod
    def from_dt(cls, dt: dict):
        cls.validate_benchmark_dict(dt)
        name = dt[cls.NAME_FIELD]
        datasets = [DatasetInfo.from_dict(rec)\
                        for rec in dt[cls.DATASETS_FIELD]]
        scorers = [ScorerInfo.from_dict(rec)\
                        for rec in dt[cls.SCORERS_FIELD]]
        metainfo = dt.get('metainfo', {})
        for key, value in dt.items():
            if key not in (cls.NAME_FIELD, cls.DATASETS_FIELD, cls.SCORERS_FIELD):
                metainfo[key] = value
        return cls(name, datasets, scorers, metainfo)

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r") as inp:
            dt = json.load(inp)
        return cls.from_dt(dt)
