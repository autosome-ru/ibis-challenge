from attr import define
from pbm import PBMRecord
from utils import auto_convert
from sequence import Sequence
from metainfo import SeqMetaInfo
from labels import Label

@define(field_transformer=auto_convert)
class SeqEntry:
    seq: Sequence
    label: Label
    metainfo: SeqMetaInfo

    @classmethod
    def from_pbm_record(cls, record: PBMRecord, label: Label):
        metainfo = SeqMetaInfo()
        metainfo['linker'] = record.linker_sequence
        return cls(PBMRecord.pbm_sequence, label, metainfo)