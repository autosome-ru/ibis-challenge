from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetInfo:
    name: str
    tf: str
    background: str
    path: Path
    
    @classmethod
    def from_dict(cls, path: Path):
        raise NotImplementedError()
