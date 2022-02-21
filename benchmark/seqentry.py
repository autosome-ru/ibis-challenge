from utils import singledispatchmethod
from attrs import define, asdict
from pbmrecord import PBMRecord
from utils import auto_convert
from sequence import Sequence
from metainfo import SeqMetaInfo
from labels import Label
from record import Record
from typing import Optional


@define(field_transformer=auto_convert)
class SeqEntry:
    seq: Sequence
    label: Label
    metainfo: SeqMetaInfo

    @singledispatchmethod
    @classmethod
    def from_record(cls, record: Record, label: Optional[Label]):
        raise NotImplementedError
    
    @from_record.register
    @classmethod
    def from_pbm_record(cls, record: PBMRecord, label: Label):
        seq = record.pbm_sequence
        metainfo = SeqMetaInfo(asdict(record, recurse=True))
        metainfo.pop('pbm_sequence')
        metainfo['linker_sequence'] = metainfo['linker_sequence']['seq']
        return cls(seq, label, metainfo)