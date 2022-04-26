from enum import Enum
from utils import register_enum

class Label:
    pass

@register_enum
class BinaryLabel(Label, Enum):
    NEGATIVE = 0
    POSITIVE = 1

    def to_json(self):
        return self.name