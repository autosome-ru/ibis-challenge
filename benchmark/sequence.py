from typing import Dict, Optional
from attrs import define, field
from alphabets import DNAAlphabet
from utils import auto_convert, register_type

@register_type
@define(field_transformer=auto_convert)
class Sequence:
    seq: str