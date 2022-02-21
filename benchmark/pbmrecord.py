from __future__ import annotations

from typing import Dict, Optional
from attrs import define, field
from sequence import Sequence
from utils import  auto_convert
from record import Record 

@define(field_transformer=auto_convert)
class PBMRecord(Record):
    id_spot: int 
    row: int 
    col: int
    control: bool
    id_probe: str 
    pbm_sequence: Sequence
    linker_sequence: Sequence
    mean_signal_intensity: float
    mean_background_intensity: Optional[float] = field(
        converter=lambda x: float(x) if x else None)
    flag: bool

    @classmethod
    def from_dict(cls, dt):
        return cls(**dt)