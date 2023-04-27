import random 
import string 

from typing import ClassVar, Iterable, Protocol
from dataclasses import dataclass, field


class UniqueTagger(Protocol):
    def tag(self) -> str:
        ...
        
    def update_used(self, tags: Iterable[str]):
        ...


@dataclass
class NameTagger:
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
             seed: int | None =None) -> 'NameTagger':
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
            continue
        self.used_tags.add(tag)
        return tag
    
    def update_used(self, tags: Iterable[str]):
        self.used_tags.update(tags)
      
@dataclass  
class AlphaTagger:
    tag_length: int
    max_occupancy: float 
    max_size: int
    rng: random.Random
    used_tags: set[str] = field(repr=False, default_factory=set) 
    
    ALPHABET: ClassVar[str] = string.ascii_lowercase
    def _non_unique_tag(self) -> str:
        tag_prts = self.rng.choices(self.ALPHABET, k=self.tag_length)
        tag = "".join(tag_prts)
        return tag
    
    def tag(self) -> str:
        if len(self.used_tags) >= self.max_size:
            raise Exception("Max size for fast random generation reached")
        while (tag := self._non_unique_tag()) in self.used_tags:
            continue
        self.used_tags.add(tag)
        return tag
    
    def update_used(self, tags: Iterable[str]):
        self.used_tags.update(tags)
        
    @classmethod
    def make(cls, 
             tag_length: int,
             max_occupancy: float=0.5,
             seed: int | None =None) -> 'AlphaTagger':
       
        possible_size = len(cls.ALPHABET) ** tag_length
        max_size = int(possible_size * max_occupancy)
        generator = random.Random(seed)
        
        return cls(tag_length=tag_length,
                   used_tags=set(), 
                   max_occupancy=max_occupancy,
                   max_size=max_size,
                   rng=generator)
    
    