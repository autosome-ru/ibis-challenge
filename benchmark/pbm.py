from __future__ import annotations
from typing import List
from attrs import define, field
from sklearn.preprocessing import normalize
from utils import END_LINE_CHARS, auto_convert, register_enum
from pbmrecord import PBMRecord
from experiment import Experiment
from pathlib import Path
from enum import Enum

@register_enum
class PBMType(Enum):
    ME = "me"
    HK = "hk"

@register_enum
class PBMPreprocessing(Enum):
    RAW = "raw"
    SD = "sd"
    QNZS = "qnzs"
    SDQN = "sdqn"


@define(field_transformer=auto_convert)
class PBMExperiment(Experiment):
    name: str
    records: List[PBMRecord] = field(repr=False)
    motif: str 
    pbm_type: PBMType
    preprocessing: PBMPreprocessing
    metainfo: dict = {}

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
             name: str,
             path: Path,
             motif: str = "uknown", 
             metainfo: dict = {},
             sep="\t"):
        records = []
        with open(path, "r") as inp:
            header = inp.readline().rstrip(END_LINE_CHARS)
            names = cls.parse_header(header)
            for line in inp:
                line = line.rstrip(END_LINE_CHARS)
                record = cls.parse_record(names, line, sep)
                records.append(record)
        metainfo['path'] = path
        
        pbm_type = metainfo.pop("pbm_type")
        preprocessing = metainfo.pop("preprocessing")
        return cls(name=name,
                   records=records, 
                   motif=motif,
                   pbm_type=pbm_type,
                   preprocessing=preprocessing, 
                   metainfo=metainfo)