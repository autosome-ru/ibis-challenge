from abc import abstractmethod, ABCMeta
from importlib.resources import path
from typing import ClassVar, Iterator, List, Optional
from exceptions import WrongDatasetTypeException
from utils import undict, register_enum, register_enum, auto_convert
from seqentry import SeqEntry
from attrs import define, fields, field
from pbmrecord import PBMRecord
from enum import Enum
from pathlib import Path


@register_enum
class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    FULL = "full"

@define
class Dataset(metaclass=ABCMeta):
    name: str
    type: DatasetType
    tf_name: str
    entries: List[SeqEntry] = field(repr=False)
    metainfo: dict

    SEQUENCE_FIELD: ClassVar[str] = "sequence"
    LABEL_FIELD: ClassVar[str] = "label"
    TAG_FIELD: ClassVar[str] = "tag"
    NO_INFO_VALUE: ClassVar[str] = "NoInfo"

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[SeqEntry]:
        return iter(self.entries)
    
    def infer_fields(self):
        fields = set()
        for en in self.entries:
            dt = undict(en.metainfo)
            fields.update(dt.keys())
        fields = list(fields)
        return fields
    
    def get_fields(self):
        if self.type is DatasetType.TRAIN:
            return self.get_train_fields()
        if self.type is DatasetType.TEST:
            return self.get_test_fields()
        if self.type is DatasetType.VALIDATION:
            return self.get_valid_fields()
        if self.type is DatasetType.FULL:
            return self.get_full_fields()
        raise WrongDatasetTypeException(f"{self.type}")

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

    def get(self, key, default):
        try:
            val = getattr(self, key)
        except AttributeError:
            val = self.metainfo.get(key, default)
        return val
    
    def to_tsv(self,
               path: Path,
               ds_fields: Optional[List[str]] = None):
        fields = self.get_fields()
        
        if ds_fields:
           ds_values = [self.get(fl, self.NO_INFO_VALUE) for fl in ds_fields]

        with open(path, "w") as out:
            if ds_fields:
                header = "\t".join(fields + ds_fields)
            else:
                header = "\t".join(fields)
            print(header, file=out)
            for en in self.entries:
                values = [en.get(fld, self.NO_INFO_VALUE) for fld in fields]  
                if ds_fields:
                    values.extend(ds_values) # type: ignore
                print("\t".join(values), file=out)
                    
    def to_json(self, path, mode: Optional[DatasetType] = DatasetType.TEST):
        raise NotImplementedError()

    @abstractmethod
    def to_canonical_format(self, path, mode: Optional[DatasetType] = DatasetType.TEST):
        raise NotImplementedError()

    def split(self):
        raise NotImplementedError()

class PBMDataset(Dataset):
    TRAIN_ONLY_META_FIELDS: ClassVar[List[str]] = [
        "mean_signal_intensity",
        "mean_background_intensity"
    ]

    META_FIELDS: ClassVar[List[str]] = [f.name for f in fields(PBMRecord)\
                          if f.name not in (Dataset.TAG_FIELD,
                                            Dataset.SEQUENCE_FIELD, 
                                            Dataset.LABEL_FIELD,
                                            'pbm_sequence')]\
                  + ['protocol']

    def get_test_fields(self):
        fields = [self.TAG_FIELD, self.SEQUENCE_FIELD]
        for f in self.META_FIELDS:
            if f not in self.TRAIN_ONLY_META_FIELDS:
                fields.append(f)
        return fields

    def get_train_fields(self):
        fields = [self.TAG_FIELD, self.SEQUENCE_FIELD, self.LABEL_FIELD]
        fields.extend(self.META_FIELDS)
        return fields

    def to_canonical_format(self, path):
        return self.to_tsv(path)