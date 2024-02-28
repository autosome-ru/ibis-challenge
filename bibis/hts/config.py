import json 

from dataclasses import dataclass, asdict
from pathlib import Path

from .dataset import  HTSRawDataset

@dataclass
class HTSRawConfig:
    tf_name: str
    tf_id: int
    stage: str
    stage_id: int
    split: str
    datasets: list[HTSRawDataset] # path to tsvs with train chips

    def save(self, path: str | Path):
        dt = asdict(self)
        with open(path, "w") as out:
            json.dump(obj=dt,
                      fp=out,
                      indent=4)
            
    @classmethod
    def load(cls, path: str | Path):
        with open(path, "r") as inp:
            dt = json.load(inp)
        dt['datasets'] = [HTSRawDataset(**d) for d in  dt['datasets'] ]
        return cls(**dt)