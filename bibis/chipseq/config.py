import json
from pyclbr import Class
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from turtle import back
from typing import ClassVar

from Bio import Data

from ..benchmark.dataset import DatasetInfo
from ..seq.seqentry import SeqEntry, read as seq_read
from ..seq.seqentry import write as seq_write
from ..scoring.label import POSITIVE_LABEL, NEGATIVE_LABEL

@dataclass
class ShadesConfig:
    balance: int
    max_dist: int
    
@dataclass
class ForeignConfig:
    balance: int
    foreigns_path: list[str]
    
@dataclass
class GenomeSampleConfig:
    balance: int
    max_overlap: int | None 
    n_procs: int
    exact: bool
    precalc_profile: bool
    
@dataclass
class ChipSeqSplit:
    chroms: list[str]
    hide_regions: str | None 

@dataclass
class ChipSeqConfig:
    tf_name: str
    tf_path: list[str]
    black_list_path: str | None
    friends_path: list[str]
    window_size: int
    genome_path: str
    seed: int 
    shades_cfg: ShadesConfig
    foreign_cfg: ForeignConfig
    genome_sample_cfg: GenomeSampleConfig
    splits: dict[str, ChipSeqSplit]
    
    SPLIT_TYPES: ClassVar[tuple[str, str, str]] = ('train', 'test', 'train/test')
    
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
        for name, split_dt in dt['splits'].items():
            dt['splits'][name] = ChipSeqSplit(**split_dt)
        dt["shades_cfg"] = ShadesConfig(**dt["shades_cfg"])
        dt["foreign_cfg"] = ForeignConfig(**dt["foreign_cfg"])
        dt["genome_sample_cfg"] = GenomeSampleConfig(**dt["genome_sample_cfg"])
        return cls(**dt)
    
@dataclass
class ChipSeqDatasetConfig:
    tf_name: str
    tf_path: str 
    
    POSITIVE_NAME: ClassVar[str] = "positives"
    FULL_BCK_NAME: ClassVar[str] = "full"
    BACKGROUNDS: ClassVar[list[str]] = ['shades', 'foreigns', 'genome']
    
    def part_path(self, part: str, fmt: str) -> Path:
        return Path(self.tf_path) / 'parts' / f"{part}.{fmt}"
    
    def save(self, path: str | Path):
        with open(path, "w") as out:
            json.dump(obj=asdict(self),
                      fp=out,
                      indent=4)
    
    @classmethod
    def load(cls, path: str | Path):
        with open(path) as inp:
            dt = json.load(inp)
        return cls(**dt)
    
    def get_negatives(self, background: str) -> list[SeqEntry]:
        if background in self.BACKGROUNDS:
            return seq_read(self.part_path(part=background,
                                        fmt="fasta"))
        elif background == self.FULL_BCK_NAME:
            seqs = []
            for bck in self.BACKGROUNDS:
                ss = seq_read(self.part_path(part=bck,
                                        fmt="fasta"))
                seqs.extend(ss)
            return seqs                 
        else:
            raise Exception(f"Wrong background: {background}")
        
    def get_positives(self) -> list[SeqEntry]:
        return seq_read(self.part_path(part=self.POSITIVE_NAME, 
                                        fmt="fasta"))
    
    @staticmethod
    def entry2key(s: SeqEntry):
        m = s.metainfo
        return m['chr'], m['start'], m['end'] # type: ignore
    
    def make_ds(self, 
                path: str | Path, 
                background: str,
                hide_labels: bool = True) -> DatasetInfo:
        positives = self.get_positives()
        negatives = self.get_negatives(background)

        if not hide_labels:
            for e in positives:
                e.label = POSITIVE_LABEL
            for e in negatives:
                e.label = NEGATIVE_LABEL
        total = positives + negatives
        
        total.sort(key=self.entry2key)
        
        seq_write(total, 
                  path)    
        
        name = f"{self.tf_name}_{background}"
        return DatasetInfo(name=name,
                           tf=self.tf_name,
                           background=background, 
                           path=str(path))
        
    def make_full_ds(self, 
                     path: str | Path, 
                     hide_labels: bool = True):
        return self.make_ds(path=path,
                            background=self.FULL_BCK_NAME,
                            hide_labels=hide_labels)