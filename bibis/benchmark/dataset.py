import json

from dataclasses import dataclass, asdict
from pathlib import Path

from ..seq.seqentry import SeqEntry, read_fasta
from ..scoring.label import Label

@dataclass
class DatasetInfo:
    name: str
    tf: str
    background: str
    fasta_path: str
    answer_path: str | None
    
    @classmethod
    def from_dict(cls, dt: dict[str, str | Path]):
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
        return asdict(self)
    
    def write_tsv(self, path: str | Path):
        entries = read_fasta(self.fasta_path)
        entries2tsv(entries=entries,
                    path=path)
        
                
    def answer(self) -> dict[str, Label]:
        if self.answer_path is None:
            raise Exception("dataset used doesn't have information about correct answer")
        with open(self.answer_path) as inp:
            ans = json.load(inp)
        return ans
    
    
def seqentry2interval_key(s: SeqEntry):
        m = s.metainfo
        try:
            return m['chr'], m['start'], m['end'] # type: ignore
        except KeyError:
            raise Exception("seqentry2interval_key works only for datasets with information about chr, start and end provided")
    
def entries2tsv(entries: list[SeqEntry], path: str | Path):
    with open(path, "w") as out:
        for e in entries:
            try:
                chr = e.metainfo['chr'] # type: ignore
                start = e.metainfo['start'] # type: ignore
                end = e.metainfo['end'] # type: ignore
            except KeyError:
                raise Exception("write tsv works only for entries with information about chr, start and end provided")
            tag = e.tag
            print(chr, start, end, tag, sep="\t", file=out)
                