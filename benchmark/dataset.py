from abc import abstractmethod, ABCMeta
from importlib.resources import path
from typing import ClassVar, List, Optional
from exceptions import (WrongDatasetTypeException, WrongExperimentTypeException,
                        WrongProtocolException, WrongProtocolException)
from labels import BinaryLabel
from utils import undict, register_enum, register_enum, auto_convert
from seqentry import SeqEntry
from attrs import define, fields, field
from pbmrecord import PBMRecord
from enum import Enum
from pathlib import Path
from pbm import PBMExperiment


@register_enum
class DatasetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    FULL = 4

@define
class Dataset(metaclass=ABCMeta):
    name: str
    motif: str
    entries: List[SeqEntry] = field(repr=False)
    metainfo: dict

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
    
    def get_fields(self, mode: DatasetType):
        if mode is DatasetType.TRAIN:
            return self.get_train_fields()
        if mode is DatasetType.TEST:
            return self.get_test_fields()
        if mode is DatasetType.VALIDATION:
            return self.get_valid_fields()
        if mode is DatasetType.FULL:
            return self.get_full_fields()
        raise WrongDatasetTypeException(f"{mode}")

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
    
    def to_tsv(self, path, mode: Optional[DatasetType] = DatasetType.TEST):
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
                    
    def to_json(self, path, mode: Optional[DatasetType] = DatasetType.TEST):
        raise NotImplementedError()

    @abstractmethod
    def to_canonical_format(self, path, mode: Optional[DatasetType] = DatasetType.TEST):
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

    def to_canonical_format(self, path, mode: Optional[DatasetType] = DatasetType.TEST):
        return self.to_tsv(path, mode)

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
class DatasetConfig:
    name: str
    exp_type: ExperimentType
    motif: str
    ds_type: DatasetType
    path: Path
    curation_status: CurationStatus
    protocol: str
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

    def get_entries(self):
        if self.exp_type is ExperimentType.PBM:
            experiment = PBMExperiment.read(self.path)
            if self.protocol == "weirauch":
                entries = experiment.weirauch_protocol()
            else:
                raise WrongProtocolException(f"No protocal {self.protocol} for {self.exp_type}")
        elif self.exp_type is ExperimentType.ChIPSeq:
            raise NotImplementedError()
        else:
            raise WrongExperimentTypeException(f"Wrong experiment type: {self.exp_type}")
        return entries 

    def get_ds_metainfo(self):
        metainfo = self.metainfo.copy()
        metainfo['protocol'] = self.protocol
        metainfo['exp_type'] = self.exp_type
        metainfo['path'] = self.path
        metainfo['curation_status'] = self.curation_status
        return metainfo

    def infer_ds_cls(self):
        if self.exp_type is ExperimentType.PBM:
            cls = PBMDataset
        elif self.exp_type is ExperimentType.ChIPSeq:
            raise NotImplementedError()
        else:
            raise WrongExperimentTypeException(f"Wrong experiment type: {self.exp_type}")
        return cls


    def make_dataset(self):
        entries = self.get_entries()
        metainfo = self.get_ds_metainfo()
        cls = self.infer_ds_cls()
        return cls(self.name, self.motif, entries, metainfo)
    

