import json

from dataclasses import dataclass, asdict
from pathlib import Path

from ..seq.seqentry import read_fasta

@dataclass
class DatasetInfo:
    name: str
    tf: str
    background: str
    path: str
    
    @classmethod
    def from_dict(cls, dt: dict[str, str | Path]):
        for field in ('name', 'tf', 'background', 'path'):
            if not field in dt:
                raise Exception(f"DataInfo must contain {field}")
        return cls(**dt) # type: ignore
    
    def save(self, path: str | Path):
        with open(path, "w") as out:
            json.dump(obj=asdict(self),
                      fp=out,
                      indent=4)
    
    @classmethod
    def load(cls, path: str | Path) -> 'DatasetInfo':
        with open(path)  as inp:
            dt = json.load(inp)
        return cls(**dt)

    def to_dict(self) -> dict[str, str]:
        dt = {}
        dt['name'] = self.name
        dt['tf'] = self.tf
        dt['background'] = self.background
        dt['path'] = str(self.path)
        return dt
    
    def write_tsv(self, path: str | Path):
        entries = read_fasta(self.path)
        with open(path, "w") as out:
            for e in entries:
                chr = e.metainfo['chr'] # type: ignore
                start = e.metainfo['start'] # type: ignore
                end = e.metainfo['end'] # type: ignore
                tag = e.tag
                print(chr, start, end, tag, sep="\t", file=out)
            