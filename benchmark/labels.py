from enum import Enum
class Label:
    pass

class BinaryLabel(Label, Enum):
    NEGATIVE = 0
    POSITIVE = 1

    def to_json(self):
        return self.name