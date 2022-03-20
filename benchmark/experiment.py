from enum import Enum
from utils import register_enum

class Experiment:
    pass

@register_enum
class ExperimentType(Enum):
    PBM = "pbm"
    ChIPSeq = "chipseq"

@register_enum
class CurationStatus(Enum):
    NOT_CURATED = "not_curated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUESTIONABLE = "questionable"