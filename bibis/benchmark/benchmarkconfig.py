from dataclasses import dataclass, field, asdict
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
    tfs: list[str]
    tags: list[str]
    pwmeval_path: Path
    metainfo: dict = field(default_factory=dict)


    @classmethod
    def from_dt(cls, dt: dict) -> 'BenchmarkConfig':
        
        dt["datasets"] =  [DatasetInfo.from_dict(rec)\
                        for rec in dt["datasets"]]
        dt["scorers"] = [ScorerInfo.from_dict(rec)\
                        for rec in dt["scorers"]]
        dt["metainfo"] = dt.get('metainfo', {})

        return cls(**dt)

    @classmethod
    def from_json(cls, path: Path | str) -> 'BenchmarkConfig':
        with open(path, "r") as inp:
            dt = json.load(inp)
        return cls.from_dt(dt)
    
    def to_dict(self) -> dict:
        dt = asdict(self)
        
        dt["datasets"] = [ds.to_dict() for ds in self.datasets]
        dt["scorers"] = [sc.to_dict() for sc in self.scorers]
        
        return dt
    
    def save(self, path: Path | str):
        dt = self.to_dict()
        with open(path, 'w') as inp:
            json.dump(dt, inp, indent=4)