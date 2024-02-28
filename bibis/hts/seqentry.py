from dataclasses import dataclass, astuple
from typing import ClassVar

@dataclass 
class SeqAssignEntry:
    seq: str
    cycle: int
    rep_ind: int
    tf_ind: int 
    stage_ind: int 
    gc_content: float

    SEP: ClassVar[str] = "\t"

    def to_line(self):
        return self.SEP.join(map(str, astuple(self)))

    @classmethod
    def from_line(cls, line):
        fields = line.split(cls.SEP)
        self = cls(*fields)
        self.cycle = int(self.cycle)
        self.rep_ind = int(self.rep_ind)
        self.tf_ind = int(self.tf_ind)
        self.stage_ind = int(self.stage_ind)
        self.gc_content = float(self.gc_content)
        return self