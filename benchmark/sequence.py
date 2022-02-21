from typing import Dict, Optional
from attrs import define, field
from alphabets import DNAAlphabet
from utils import auto_convert, register_global_converter

@define(field_transformer=auto_convert)
class Sequence:
    seq: str
    id: Optional[str] = None
    metainfo: Dict = field(factory=dict, repr=False)

register_global_converter('Sequence', Sequence)