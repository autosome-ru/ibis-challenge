from dataclasses import dataclass, field
import json

from pathlib import Path
from typing import ClassVar

from ..scoring.scorer import ScorerInfo

from pathlib import Path 

from .dataset import DatasetInfo

@dataclass
class BenchmarkConfig:
    name: str
    kind: str
    datasets: list[DatasetInfo]
    scorers: list[ScorerInfo]
    pwmeval_path: Path
    metainfo: dict = field(default_factory=dict)
    
    NAME_FIELD: ClassVar[str] = 'name'
    KIND_FIELD: ClassVar[str] = "kind"
    DATASETS_FIELD: ClassVar[str] = 'datasets'
    SCORERS_FIELD: ClassVar[str] = 'scorers'
    PWMEVAL_PATH_FIELD: ClassVar[str] = "pwmeval"

    @classmethod
    def validate_benchmark_dict(cls, dt: dict):
        if not cls.NAME_FIELD in dt:
            raise Exception(
                    f"Benchmark config must has field '{cls.NAME_FIELD}'")
        if not cls.DATASETS_FIELD in dt:
            raise Exception("No information about datasets found")
        if not cls.SCORERS_FIELD in dt:
            raise Exception("No information about scorers found")

    @classmethod
    def from_dt(cls, dt: dict) -> 'BenchmarkConfig':
        cls.validate_benchmark_dict(dt)
        name = dt[cls.NAME_FIELD]
        datasets = [DatasetInfo.from_dict(rec)\
                        for rec in dt[cls.DATASETS_FIELD]]
        scorers = [ScorerInfo.from_dict(rec)\
                        for rec in dt[cls.SCORERS_FIELD]]
        
        kind = dt[cls.KIND_FIELD]
        
        pwmeval_path = dt.get(cls.PWMEVAL_PATH_FIELD)
        if pwmeval_path is None:
            raise Exception("PWMEval path must be provided")
        
        metainfo = dt.get('metainfo', {})

        for key, value in dt.items():
            if key not in (cls.NAME_FIELD, cls.DATASETS_FIELD, cls.SCORERS_FIELD, cls.PWMEVAL_PATH_FIELD):
                metainfo[key] = value

        return cls(name=name, 
                   kind=kind,
                   datasets=datasets,
                   scorers=scorers, 
                   pwmeval_path=pwmeval_path,
                   metainfo=metainfo)

    @classmethod
    def from_json(cls, path: Path | str) -> 'BenchmarkConfig':
        with open(path, "r") as inp:
            dt = json.load(inp)
        return cls.from_dt(dt)
    
    def to_dict(self) -> dict:
        dt = {}
        dt[self.NAME_FIELD] = self.name
        dt[self.KIND_FIELD] = self.kind
        dt['metainfo'] = self.metainfo
        dt[self.PWMEVAL_PATH_FIELD] = str(self.pwmeval_path)
        
        datasets = []
        for ds in self.datasets:
            datasets.append(ds.to_dict())
        dt[self.DATASETS_FIELD] = datasets
        
        scorers = []
        for sc in self.scorers:
            scorers.append(sc.to_dict())
        dt[self.SCORERS_FIELD] = scorers
        
        return dt
    
    def save(self, path: Path | str):
        dt = self.to_dict()
        with open(path, 'w') as inp:
            json.dump(dt, inp, indent=4)