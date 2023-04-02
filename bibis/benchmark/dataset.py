from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetInfo:
    name: str
    tf: str
    background: str
    path: Path
    
    @classmethod
    def from_dict(cls, dt: dict[str, str | Path]):
        for field in ('name', 'tf', 'background', 'path'):
            if not field in dt:
                raise Exception(f"DataInfo must contain {field}")
        dt['path'] = Path(dt['path'])
        return cls(**dt) # type: ignore

    def to_dict(self) -> dict[str, str]:
        dt = {}
        dt['name'] = self.name
        dt['tf'] = self.tf
        dt['background'] = self.background
        dt['path'] = str(self.path)
        return dt