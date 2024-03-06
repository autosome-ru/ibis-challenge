import json

from pathlib import Path

from dataclasses import dataclass, asdict

from typing import ClassVar
from venv import logger

from ..benchmark.dataset import DatasetInfo, seqentry2interval_key
from ..seq.seqentry import SeqEntry, read as seq_read
from ..seq.seqentry import write as seq_write
from ..scoring.label import POSITIVE_LABEL, NEGATIVE_LABEL
from ..logging import get_bibis_logger

logger = get_bibis_logger()

@dataclass
class ShadesConfig:
    balance: int
    min_dist: int
    max_dist: int
    
@dataclass
class ForeignConfig:
    balance: int
    min_dist: int
    foreigns_path: list[str]
    
@dataclass
class GenomeSampleConfig:
    balance: int
    min_dist: int
    max_overlap: int | None 
    n_procs: int
    exact: bool
    precalc_profile: bool
    
@dataclass
class PeakSeqSplit:
    reps: dict[str, str]
    chroms: list[str]
    hide_regions: str | None 

@dataclass
class PeakSeqConfig:
    tf_name: str
    black_list_path: str | None
    friends_path: list[str]
    window_size: int
    genome_path: str
    seed: int 
    shades_cfg: ShadesConfig
    foreign_cfg: ForeignConfig
    genome_sample_cfg: GenomeSampleConfig
    splits: dict[str, PeakSeqSplit]
    
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
            dt['splits'][name] = PeakSeqSplit(**split_dt)
        dt["shades_cfg"] = ShadesConfig(**dt["shades_cfg"])
        dt["foreign_cfg"] = ForeignConfig(**dt["foreign_cfg"])
        dt["genome_sample_cfg"] = GenomeSampleConfig(**dt["genome_sample_cfg"])
        return cls(**dt)
    
@dataclass
class PeakSeqDatasetConfig:
    tf_name: str
    tf_path: str 
    
    POSITIVE_NAME: ClassVar[str] = "positives"
    FULL_BCK_NAME: ClassVar[str] = "full"
    BACKGROUNDS: ClassVar[list[str]] = ['shades', 'aliens', 'random']
    
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
            part_path = self.part_path(part=background,
                                        fmt="fasta")
            if not part_path.exists():
                logger.warning(f"Skipping {background}: doesn't exists")
                return []
            return seq_read(part_path)
        elif background == self.FULL_BCK_NAME:
            seqs = []
            for bck in self.BACKGROUNDS:
                part_path = self.part_path(part=bck,
                                        fmt="fasta")
                if not part_path.exists():
                    logger.warning(f"Skipping {background}: doesn't exists")
                    continue
                ss = seq_read(part_path)
                seqs.extend(ss)
            return seqs                 
        else:
            raise Exception(f"Wrong background: {background}")
        
    def get_positives(self) -> list[SeqEntry]:
        return seq_read(self.part_path(part=self.POSITIVE_NAME, 
                                        fmt="fasta"))
    
    def fasta_path(self, path_pref: str) -> str:
        return f"{path_pref}.fasta"
    
    def answer_path(self, path_pref: str) -> str:
        return f"{path_pref}_answer.json"
    
    def make_ds(self, 
                path_pref: str, 
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
        
        total.sort(key=seqentry2interval_key)
        
        fasta_path = self.fasta_path(path_pref=path_pref)
        seq_write(total, 
                  fasta_path) 

        if not hide_labels:
            answer_path = self.answer_path(path_pref=path_pref)
            answers = {'labels': {e.tag: e.label for e in total}}
            with open(answer_path, "w") as out:
                json.dump(obj=answers, 
                          fp=out,
                          indent=4)
        else:
            answer_path = None
          
        
        name = f"{self.tf_name}_{background}"
        return DatasetInfo(name=name,
                           tf=self.tf_name,
                           background=background, 
                           fasta_path=str(fasta_path),
                           answer_path=answer_path)
        
    def make_full_ds(self, 
                     path_pref: str, 
                     hide_labels: bool = True):
        return self.make_ds(path_pref=path_pref,
                            background=self.FULL_BCK_NAME,
                            hide_labels=hide_labels)