from enum import Enum
from utils import register_enum

class Experiment:
    pass

@register_enum
class ExperimentType(Enum):
    PBM = 1
    ChIPSeq = 2

@register_enum
class CurationStatus(Enum):
    NOT_CURATED = 1
    ACCEPTED = 2
    REJECTED = 3
    QUESTIONABLE = 4