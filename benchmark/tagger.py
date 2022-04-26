import json 
import random 
import typing

from attr import define, field 
from pathlib import Path
from typing import ClassVar, Union 

from utils import END_LINE_CHARS

@define
class UniqueTagger:
    parts: typing.Sequence[str]
    vocabularies: dict[str, list[str]] = field(repr=False)
    used_tags: set[str] = field(repr=False, factory=set) 
    max_occupancy: float = 0.5
    seed: int = 777
    max_size: int = field(init=False)
    random_generator: random.Random = field(init=False, repr=False)


    DEFAULT_PARTS: ClassVar[typing.Sequence[str]] = ("adj", "adj", "nat", "ani")
    DEFAULT_PARTS_PATH: ClassVar[dict[str, Path]] = {'adj': Path("adjectives.txt"), 'nat': Path('nations.txt'), 'ani': Path('animals.txt')}

    PARTS_FIELD: ClassVar[str]="PARTS"
    VOC_FIELD: ClassVar[str] = "VOC"
    TAGS_FIELD: ClassVar[str] = "USED_TAGS"
    PARTS_SEP: ClassVar[str] = '-'

    def __attrs_post_init__(self):
        self.max_size = self._calc_maxsize()
        self.random_generator = random.Random(self.seed)

    @classmethod
    def make(cls, parts: typing.Sequence[str], parts_path: dict[str, Path]):
        dt = {}
        for part in parts:
            with parts_path[part].open() as inp:
                voc =  [line.rstrip(END_LINE_CHARS) for line in inp]
                dt[part] = voc
        return cls(parts, dt)

    @classmethod
    def default(cls):
        return cls.make(cls.DEFAULT_PARTS, cls.DEFAULT_PARTS_PATH)        

    def write(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        with path.open('w') as store:
            dt = {self.PARTS_FIELD: self.parts,
                  self.VOC_FIELD: self.vocabularies,
                  self.TAGS_FIELD: list(self.used_tags)}
            json.dump(dt, store, indent=4)

    @classmethod
    def load(cls, path: Union[Path, str]) -> 'UniqueTagger':
        if isinstance(path, str):
            path = Path(path)
        with path.open() as inp:
            dt = json.load(inp)
        parts = dt[cls.PARTS_FIELD]
        if not isinstance(parts, list):
            raise Exception('Wrong format')
        vocs = dt[cls.VOC_FIELD]
        if not isinstance(vocs, dict):
            raise Exception('Wrong format')
        tags = dt[cls.TAGS_FIELD]
        if not isinstance(tags, list):
            raise Exception('Wrong format')

        return cls(tuple(parts), vocs, set(tags)) 
    
    def _non_unique_tag(self) -> str:
        tag_prts = [self.random_generator.choice(self.vocabularies[p]) for p in self.parts]
        tag = self.PARTS_SEP.join(tag_prts)
        return tag
    
    def tag(self) -> str:
        if len(self.used_tags) >= self.max_size:
            raise Exception("Max size for fast random generation reached")
        while (tag := self._non_unique_tag()) in self.used_tags:
            pass
        self.used_tags.add(tag)
        return tag
    
    def _calc_maxsize(self) -> int:
        cnt = 1
        for p in self.parts:
            cnt *= len(self.vocabularies[p])
        return int(cnt * self.max_occupancy) 