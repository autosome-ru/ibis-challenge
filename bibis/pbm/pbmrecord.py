from __future__ import annotations

from typing import Optional
from dataclasses import dataclass 
from Bio.Seq import Seq 
from ..seq.seqentry import SeqEntry
from ..scoring.label import NO_LABEL

@dataclass 
class PBMRecord:
    id_spot: int 
    row: int 
    col: int
    control: bool
    id_probe: str 
    pbm_sequence: Seq
    linker_sequence: Seq
    mean_signal_intensity: float
    mean_background_intensity: Optional[float]
    flag: bool

    @classmethod
    def from_dict(cls, dt):
        dt['id_spot'] = int(dt['id_spot'])
        dt['row'] = int(dt['row'])
        dt['col'] = int(dt['col'])
        dt['control'] = bool(dt['control'])
        dt['pbm_sequence'] = Seq(dt['pbm_sequence'])
        dt['linker_sequence'] = Seq(dt['linker_sequence'])
        dt['mean_signal_intensity'] = float(dt["mean_signal_intensity"])
        
        if dt["mean_background_intensity"] == "":
            dt["mean_background_intensity"] = None
        elif dt["mean_background_intensity"] is not None:
            dt["mean_background_intensity"] = float(dt["mean_background_intensity"])
        dt['flag'] = bool(dt['flag'])
        return cls(**dt)

    def to_seqentry(self, label=NO_LABEL) -> SeqEntry:
        return SeqEntry(sequence=self.linker_sequence + self.pbm_sequence,
                        tag=self.id_probe,
                        label=label,
                        metainfo={"id_spot": self.id_spot,
                                  "row": self.row,
                                  "col": self.col,
                                  "linker": self.linker_sequence,
                                  })