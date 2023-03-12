import random 
from typing import ClassVar
from dataclasses import dataclass, field


@dataclass
class UniqueTagger:
    parts: list[str]
    vocabularies: dict[str, list[str]] = field(repr=False)
    max_occupancy: float 
    max_size: int
    rng: random.Random
    used_tags: set[str] = field(repr=False, default_factory=set) 

    PARTS_SEP: ClassVar[str] = '-'
    
    @classmethod
    def make(cls, 
             parts: list[str], 
             vocabularies: dict[str, list[str]], 
             max_occupancy: float=0.5,
             seed: int=777) -> 'UniqueTagger':
        cnt = 1
        for p in parts:
            cnt *= len(vocabularies[p])
        max_size = int(cnt * max_occupancy)
        generator = random.Random(seed)
        
        return cls(parts=parts,
                   vocabularies=vocabularies, 
                   used_tags=set(), 
                   max_occupancy=max_occupancy,
                   max_size=max_size,
                   rng=generator)
    
    def _non_unique_tag(self) -> str:
        tag_prts = [self.rng.choice(self.vocabularies[p]) for p in self.parts]
        tag = self.PARTS_SEP.join(tag_prts)
        return tag
    
    def tag(self) -> str:
        if len(self.used_tags) >= self.max_size:
            raise Exception("Max size for fast random generation reached")
        while (tag := self._non_unique_tag()) in self.used_tags:
            pass
        self.used_tags.add(tag)
        return tag