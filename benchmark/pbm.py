from __future__ import annotations

import numpy as np

from statistics import mean
from sys import path_importer_cache
from typing import Dict, List, Optional
from attrs import define, field
from sequence import Sequence
from math import nan
from utils import END_LINE_CHARS, auto_convert

@define(field_transformer=auto_convert)
class PBMRecord:
    id_spot: int 
    row: int 
    col: int
    control: bool
    id_probe: str 
    pbm_sequence: Sequence
    linker_sequence: Sequence
    mean_signal_intensity: float
    mean_background_intensity: Optional[float] = field(
        converter=lambda x: float(x) if x else None)
    flag: bool
    metainfo: Dict = field(factory=dict, repr=False)

    @classmethod
    def from_dict(cls, dt):
        return cls(**dt)

@define
class PBMExperiment:
    records: List[PBMRecord] = field(repr=False)
    mean: float = field(init=False, repr=True)
    std: float = field(init=False, repr=True)

    def __attrs_post_init__(self):
        vals = [r.mean_signal_intensity for r in self.records]
        self.mean = np.mean(vals)
        self.std = np.std(vals)

    @staticmethod
    def parse_header(header):
        names = header\
                .lstrip("#")\
                .split('\t')
        return names

    @staticmethod
    def parse_record(names, line, sep):
        values = line.split(sep)
        record = PBMRecord.from_dict(
                    dict(zip(names,values))
                )
        return record

    @classmethod
    def read(cls, path, sep="\t"):
        records = []
        with open(path, "r") as inp:
            header = inp.readline().rstrip(END_LINE_CHARS)
            names = cls.parse_header(header)
            for line in inp:
                line = line.rstrip(END_LINE_CHARS)
                record = cls.parse_record(names, line, sep)
                records.append(record)
        return cls(records)

@define
class PBMDataset:
    positive: List[PBMRecord]
    negative: List[PBMRecord]
    
    def get_records(self, hide_labels=True):
        pos_seq = [r.pbm_sequence for r in self.positive]
        pos_labels = [0] * len(pos_seq)
        neg_seq = [r.pbm_sequence for r in self.negative]
        neg_labels = [0] * len(neg_seq)

        seqs = pos_seq + neg_seq
        labels = pos_labels + neg_labels
        return seqs, labels

    @staticmethod
    def weirauch_threshold(experiment: PBMExperiment,
                           min_probs=50,
                           max_probs=1300):
        vals = [r.mean_signal_intensity for r in experiment.records]
        vals.sort()
        mean = np.mean(vals)
        std = np.std(vals)
        th1 = mean + 4 * std
        th2 = vals[-min_probs]
        th = min(th1, th2)
        th3 = vals[-max_probs]
        th = max(th, th3)
        return th

    @classmethod
    def weirauch_protocol(cls, 
                        experiment: PBMExperiment):
        threshold = cls.weirauch_threshold(experiment)
        return cls.one_threshold_protocol(experiment, 
                                          threshold)

    @classmethod
    def one_threshold_protocol(cls,
                               experiment: PBMExperiment,
                               threshold: float):
        pos = []
        neg = []
        for r in experiment.records:
            if r.mean_signal_intensity >= threshold:
                pos.append(r)
            else:
                neg.append(r)
        
        return cls(pos, neg)

    def to_tsv(self, path):
        raise NotImplementedError

    def to_json(self, path):
        raise NotImplementedError

    def to_classic_format(self, path):
        return 

    