import json

from dataclasses import dataclass, asdict
from pathlib import Path

from ..utils import END_LINE_CHARS
from .tagger import UniqueTagger
from .seqdb import TagDatabase 

@dataclass
class DBConfig:
    db_path: Path 
    parts: list[str]
    parts_cont: dict[str, list[str]]
    max_occupancy: float
    wait_time: float
    
    @classmethod
    def make(cls, 
             db_path: str| Path,
             parts: list[str], 
             parts_path: dict[str, str],
             max_occupancy: float = 0.5,
             wait_time: float = 0.1):
        """
        Make config using names from some initial files 
        In general it's better to make config ones and 
        then use save and load to work with it
        """
        if isinstance(db_path, str):
            db_path = Path(db_path)
        parts_cont = {}
        for name, path in parts_path.items():
            with open(path, "r") as inp:
                lst = [line.strip(END_LINE_CHARS) for line in inp]
            parts_cont[name] = lst
            
        return cls(db_path=db_path,
                   parts=parts,
                   parts_cont=parts_cont,
                   max_occupancy=max_occupancy,
                   wait_time=wait_time)
    
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
    
    def build(self) -> 'TagDatabase':
        tagger = UniqueTagger.make(parts=self.parts,
                              vocabularies=self.parts_cont, 
                              max_occupancy=self.max_occupancy,
                              seed=777)
        return TagDatabase.make(db_path=self.db_path, tagger=tagger, wait_time=self.wait_time)
        
            
    

