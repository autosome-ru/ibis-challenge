from __future__ import annotations
from typing import List
from dataclasses import dataclass, field 


from .pbmrecord import PBMRecord

from pathlib import Path
from enum import Enum
from ..utils import END_LINE_CHARS

class PBMType(Enum):
    ME = "me"
    HK = "hk"

class PBMPreprocessing(Enum):
    RAW = "raw"
    SD = "sd"
    QNZS = "qnzs"
    SDQN = "sdqn"

@dataclass
class PBMExperiment:
    records: List[PBMRecord] = field(repr=False)

    def __iter__(self):
        return self.records.__iter__()
    
    def __getitem__(self, ind):
        return self.records[ind]

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
    def read(cls, 
             path: Path,
             sep="\t"):
        records = []
        with open(path, "r") as inp:
            header = inp.readline().rstrip(END_LINE_CHARS)
            names = cls.parse_header(header)
            for line in inp:
                line = line.rstrip(END_LINE_CHARS)
                record = cls.parse_record(names, line, sep)
                records.append(record)
       
        return cls(records=records)