from typing import Dict

from attr import define
from pathlib import Path

@define
class Prediction:
    dt: Dict[str, float]

    def __getitem__(self, tag: str) -> float:
        return self.dt[tag]

    def get(self, tag: str, default=None):
        return self.dt.get(tag, default)

    def __len__(self) -> int:
        return len(self.dt)

    @classmethod
    def load(cls, path: Path, sep="\t"):
        with path.open(mode="r") as infile:
            dt = {}
            for line in infile:
                tag, score = line.split(sep)
                score = float(score)
                dt[tag] = score
        return cls(dt)
