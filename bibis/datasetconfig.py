from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

@dataclass
class DatasetConfig:
    name: str
    tf: str
    path: Path
    metainfo: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, dt: dict):
        names = {f.name for f in fields(cls)}
        
        args = {}
        metainfo = {}
        for key, value in dt.items():
            if key in names:
                args[key] = value
            else:
                metainfo[key] = value
    
        if "metainfo" in args:
            metainfo['metainfo'] = args['metainfo']
        args['metainfo'] = metainfo

        return cls(**args)
    
    def to_dict(self):
        dt = asdict(self)
        meta = dt.pop("metainfo")
        for key, value in meta.items():
            dt[key] = value
        return dt