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
    left_flank: str | None = None # valid for SMS and HT-SELEX
    right_flank: str | None = None # valid for SMS and HT-SELEX
    
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
    

def get_seqentrykey(s: SeqEntry, kind: str):
    if kind in ("GHTS", "CHS"):
        return seqentry2interval_key(s)
    elif kind == "PBM":
        return seqentry2pbm_key(s)
    else:
        raise Exception(f"entries2tsv is not implemented for benchmark {kind}")

def seqentry2interval_key(s: SeqEntry):
        m = s.metainfo
        try:
            return m['chr'], m['start'], m['end'] # type: ignore
        except KeyError:
            raise Exception("seqentry2interval_key works only for datasets with information about chr, start and end provided")

def seqentry2pbm_key(s: SeqEntry):
    return s.tag

def entries2tsv(entries: list[SeqEntry], path: str | Path, kind: str):
    if kind in ("GHTS", "CHS"):
        return peaks_entries2tsv(entries, path)
    elif kind == "PBM":
        return pbm_entries2tsv(entries, path)
    elif kind == "SMS":
        return sms_entries2tsv(entries, path)
    else:
        raise Exception(f"entries2tsv is not implemented for benchmark {kind}")

def peaks_entries2tsv(entries: list[SeqEntry], path: str | Path):
    with open(path, "w") as out:
        for e in entries:
            try:
                chr = e.metainfo['chr'] # type: ignore
                start = e.metainfo['start'] # type: ignore
                end = e.metainfo['end'] # type: ignore
            except KeyError:
                raise Exception("peaks_entries2tsv works only for entries with information about chr, start and end provided")
            tag = e.tag
            print(chr, start, end, tag, sep="\t", file=out)

def pbm_entries2tsv(entries: list[SeqEntry], path: str | Path):
    with open(path, "w") as out:
        for e in entries:
            try:
                id_spot = e.metainfo['id_spot'] # type: ignore
                row = e.metainfo['row'] # type: ignore
                col = e.metainfo['col'] # type: ignore
                linker = e.metainfo['linker'] 
            except KeyError:
                raise Exception("pbm_entries2tsv works only for entries with information about id_spot, row, col and linker provided")
            tag = e.tag
            print(id_spot, row, col, tag, sep="\t", file=out)

def sms_entries2tsv(entries: list[SeqEntry], path: str | Path):
    with open(path, "w") as out:
        for e in entries:
            tag = e.tag
            print(tag, file=out)