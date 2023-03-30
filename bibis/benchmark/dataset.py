from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetInfo:
    name: str
    tf: str
    path: Path
    
    @classmethod
    def from_dict(cls, path: Path):
        return cls("", "", path)
