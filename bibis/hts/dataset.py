from dataclasses import dataclass

@dataclass
class HTSRawDataset:
    rep_id: int
    cycle: int
    size: int 
    path: str
    rep: str
    exp_tp: str
    left_flank: str
    right_flank: str
    raw_paths: list[str]