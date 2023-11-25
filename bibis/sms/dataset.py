from dataclasses import dataclass

@dataclass
class SMSRawDataset:
    path: str
    left_flank: str
    right_flank: str