from utils import singledispatchmethod
from attrs import define, asdict
from pbmrecord import PBMRecord
from utils import auto_convert
from sequence import Sequence
from labels import Label
from record import Record
from typing import Optional, ClassVar


@define(field_transformer=auto_convert)
class SeqEntry:
    sequence: Sequence
    label: Label
    tag: str
    metainfo: dict

    @singledispatchmethod
    @classmethod
    def from_record(cls, record: Record, label: Optional[Label]):
        raise NotImplementedError
    
    @from_record.register
    @classmethod
    def from_pbm_record(cls, record: PBMRecord, label: Label):
        sequence = record.pbm_sequence
        tag = record.id_probe
        metainfo = asdict(record, recurse=True)
        metainfo.pop('pbm_sequence')
        metainfo['linker_sequence'] = metainfo['linker_sequence']['seq']
        metainfo['source'] = "PBM"
        return cls(sequence, label, tag, metainfo)

    def get(self, key, default=None):
        try:
            val = getattr(self, key)
            if key == "sequence":
                val = val.seq
            elif key == "label":
                val = val.name 
        except AttributeError:
            val = self.metainfo.get(key, default)
        return val