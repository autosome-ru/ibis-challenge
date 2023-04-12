from __future__ import annotations

import numpy as np 

from typing import Dict
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List
from ..utils import END_LINE_CHARS


@dataclass
class Prediction:
    _pred: Dict[str, float] = field(default_factory=dict)

    REPR_SKIPVALUE: ClassVar[str] = "nodata"
    SKIPVALUE: ClassVar[float] = np.nan
    MAX_PRECISION: ClassVar[int] = 5

    def __getitem__(self, tag: str) -> float:
        return self._pred[tag]

    def get(self, tag: str):
        return self._pred.get(tag, None)

    def __len__(self) -> int:
        return len(self._pred)

    def __setitem__(self, tag:str, score: float):
        self._pred[tag] = score

    @property
    def tags(self):
        return list(self._pred.keys())

    @classmethod
    def load(cls, path: Path, sep="\t"):
        with path.open(mode="r") as infile:
            dt = {}
            for line in infile:
                tag, score = line.split(sep)
                score = float(score)
                dt[tag] = score
        return cls(dt)

    @classmethod
    def template(cls, tags: List[str]):
        dt = {t: cls.SKIPVALUE for t in tags}
        return Prediction(dt)

    @classmethod
    def is_skipvalue(cls, value):
        return value is cls.SKIPVALUE

    @classmethod
    def str2val(cls, value: str) -> float:
        if value == cls.REPR_SKIPVALUE:
            return cls.SKIPVALUE
        return float(value)

    @classmethod
    def val2str(cls, value: float, precision: int = 5) -> str:
        if cls.is_skipvalue(value):
            return cls.REPR_SKIPVALUE
        return f"{value:.0{cls.MAX_PRECISION}f}"