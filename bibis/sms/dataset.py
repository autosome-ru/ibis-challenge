from dataclasses import dataclass

@dataclass
class SMSRawDataset:
    path: str
    size: int
    left_flank: str
    right_flank: str