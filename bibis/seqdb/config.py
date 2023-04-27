import json

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import ClassVar

from ..utils import END_LINE_CHARS
from .tagger import AlphaTagger, NameTagger, UniqueTagger
from .seqdb import TagDatabase 

@dataclass
class DBConfig:
    db_path: Path 
    tagger_type: str
    max_occupancy: float
    wait_time: float
    # name tagger params 
    parts: list[str] | None
    parts_cont: dict[str, list[str]] | None
    # alpha tagget params 
    tag_length: int | None
    
    NAME_TAGGER_NAME: ClassVar[str] = "name"
    ALPHA_TAGGER_NAME: ClassVar[str] = "alpha"
    
    @classmethod
    def make(cls, 
             db_path: str| Path,
             tagger_type: str, 
             max_occupancy: float = 0.5,
             wait_time: float = 0.1,
             tag_length: int | None = None,
             parts: list[str] | None = None, 
             parts_path: dict[str, str] | None = None):
        """
        Make config using names from some initial files 
        In general it's better to make config ones and 
        then use save and load to work with it
        """
        if isinstance(db_path, str):
            db_path = Path(db_path)
            
        if tagger_type == cls.NAME_TAGGER_NAME:
            if parts is None or parts_path is None:
                raise Exception("parts and parts_path must be specified for name tagger")
            parts_cont = cls._prepare_parts(parts_path)
        else:
            parts_cont = None
            
        return cls(db_path=db_path,
                   parts=parts,
                   parts_cont=parts_cont,
                   max_occupancy=max_occupancy,
                   wait_time=wait_time,
                   tagger_type=tagger_type,
                   tag_length=tag_length)
        
    @classmethod
    def _prepare_parts(cls, 
                        parts_path: dict[str, str]) -> dict[str, list[str]]:
        parts_cont: dict[str, list[str]] = {}
        for name, path in parts_path.items():
            with open(path, "r") as inp:
                lst = [line.strip(END_LINE_CHARS) for line in inp]
            parts_cont[name] = lst
        return parts_cont
    
    def save(self, path: str | Path):
        dt = asdict(self)
        dt["db_path"] = str(dt["db_path"])
        with open(path, "w") as out:
            json.dump(obj=dt, 
                      fp=out, 
                      indent=4)
            
    @classmethod
    def load(cls, path: str | Path) -> 'DBConfig':
        with open(path, "r") as inp:
            dt = json.load(inp)
        return cls(**dt)
    
    def build_tagger(self) -> UniqueTagger:
        if self.tagger_type == self.NAME_TAGGER_NAME:
            if self.parts is None or self.parts_cont is None:
                raise Exception("parts and parts_cont must be specified for name tagger")
            return NameTagger.make(parts=self.parts,
                                   vocabularies=self.parts_cont, 
                                   max_occupancy=self.max_occupancy)
        elif self.tagger_type == self.ALPHA_TAGGER_NAME:
            if self.tag_length is None:
                raise Exception("tag length must be provided for alpha tagger")
            return AlphaTagger.make(tag_length=self.tag_length,
                                    max_occupancy=self.max_occupancy)
        else:
            raise Exception("Wrong tagger type: {self.tagger_type}")
    
    def build(self) -> 'TagDatabase':
        tagger = self.build_tagger()
        return TagDatabase.make(db_path=self.db_path, tagger=tagger, wait_time=self.wait_time)
        
            
    

