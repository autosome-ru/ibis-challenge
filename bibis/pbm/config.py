import json

from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class PBMConfig:
    tf_name: str
    train_paths: list[str] # path to tsvs with train chips
    test_paths: list[str] # path to tsvs with test chips
    protocol: str # name of the way to get positives
    neg2pos_ratio: int # how much negative should be sampled

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
        return cls(**dt)
