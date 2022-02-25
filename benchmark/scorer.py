from attrs import define
from utils import auto_convert

@define(field_transformer=auto_convert)
class ScorerInfo:
    name: str
    alias: str = ""
    params: dict = {}

    @classmethod
    def from_dict(cls, dt: dict):
        return cls(**dt)

    def __attrs_post_init__(self):
        if not self.alias:
            self.alias = self.name

class Scorer:
    pass