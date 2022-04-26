from enum import Enum
from utils import register_enum

from exceptions import WrongLabelException

class Label:
    pass

@register_enum
class BinaryLabel(Label, Enum):
    NEGATIVE = 0
    POSITIVE = 1

    def to_json(self):
        return self.name

    @classmethod
    def from_str(cls, s: str) -> 'BinaryLabel':
        for tp in cls:
            if tp.name.lower() == s.lower():
                return tp
            try:
                if tp.value == int(s):
                    return tp
            except ValueError:
                pass
        raise WrongLabelException(f"wrong value for {type(cls)}: {repr(s)}")
            