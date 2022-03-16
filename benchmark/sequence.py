from typing import Dict, Optional
from attrs import define, field
from utils import auto_convert, register_type

@register_type
@define(field_transformer=auto_convert)
class Sequence:
    seq: str

    def __repr__(self) -> str:
        return self.seq