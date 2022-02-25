from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import ClassVar, List, Optional
from exceptions import WrongDatasetModeException
from labels import BinaryLabel
from utils import register_enum, undict, auto_convert
from seqentry import SeqEntry
from attrs import define, fields
from pbmrecord import PBMRecord
from pathlib import Path

@register_enum
class DatasetMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    FULL = 4

@register_enum
class ExperimentType(Enum):
    PBM = 1
    ChIPSeq = 2

@register_enum
class CurationStatus(Enum):
    NOT_CURATED = 1
    ACCEPTED = 2
    REJECTED = 3
    QUESTIONABLE = 4

@define(field_transformer=auto_convert)
class DatasetInfo:
    name: str
    type: ExperimentType
    motif: str
    mode: DatasetMode
    path: Path
    curation_status: CurationStatus
    metainfo: dict

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
        if "metainfo" not in args:
            args['metainfo'] = metainfo
        else:
            args['metainfo'].update(metainfo)
        return cls(**args)

@define
class Dataset(metaclass=ABCMeta):
    entries: List[SeqEntry]
    SEQUENCE_FIELD: ClassVar[str] = "sequence"
    LABEL_FIELD: ClassVar[str] = "label"
    NO_INFO_VALUE: ClassVar[str] = "NoInfo"

    def infer_fields(self):
        fields = set()
        for en in self.entries:
            dt = undict(en.metainfo)
            fields.update(dt.keys())
        fields = list(fields)
        return fields
    
    def get_fields(self, mode: DatasetMode):
        if mode is DatasetMode.TRAIN:
            return self.get_train_fields()
        if mode is DatasetMode.TEST:
            return self.get_test_fields()
        if mode is DatasetMode.VALIDATION:
            return self.get_valid_fields()
        if mode is DatasetMode.FULL:
            return self.get_full_fields()
        raise WrongDatasetModeException(f"{mode}")

    @abstractmethod
    def get_train_fields(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_test_fields(self):
        raise NotImplementedError()

    def get_valid_fields(self):
        return self.get_train_fields()
    
    def get_full_fields(self):
        return self.get_train_fields()
    
    def to_tsv(self, path, mode: Optional[DatasetMode] = DatasetMode.TEST):
        if mode is None:
            fields = self.infer_fields()
        else:
            fields = self.get_fields(mode)
        
        with open(path, "w") as out:
            header = "\t".join(fields)
            print(header, file=out)
            for en in self.entries:
                values = []
                for fld in fields:
                    if fld == self.SEQUENCE_FIELD:
                        seq = getattr(en, fld)
                        val = seq.seq
                    elif fld == self.LABEL_FIELD:
                        label = getattr(en, fld)
                        if isinstance(label, BinaryLabel):
                           val = label.name 
                        else:
                            raise NotImplementedError()
                    else:
                        val = en.metainfo.get(fld, self.NO_INFO_VALUE)
                    values.append(str(val))
                print("\t".join(values), file=out)
                    
    def to_json(self, path, mode: Optional[DatasetMode] = DatasetMode.TEST):
        raise NotImplementedError()

    @abstractmethod
    def to_canonical_format(self, path, mode: Optional[DatasetMode] = DatasetMode.TEST):
        raise NotImplementedError()

    def split(self):
        raise NotImplementedError()

class PBMDataset(Dataset):
    TRAIN_ONLY_META_FIELDS = [
        "mean_signal_intensity",
        "mean_background_intensity"
    ]

    META_FIELDS = [f.name for f in fields(PBMRecord)\
                          if f.name not in (Dataset.SEQUENCE_FIELD, 
                                            Dataset.LABEL_FIELD, 
                                            'pbm_sequence')]\
                  + ['protocol']

    def get_test_fields(self):
        fields = [self.SEQUENCE_FIELD]
        for f in self.META_FIELDS:
            if f not in self.TRAIN_ONLY_META_FIELDS:
                fields.append(f)
        return fields

    def get_train_fields(self):
        fields = [self.SEQUENCE_FIELD, self.LABEL_FIELD]
        fields.extend(self.META_FIELDS)
        return fields

    def to_canonical_format(self, path, mode: Optional[DatasetMode] = DatasetMode.TEST):
        return self.to_tsv(path, mode)
