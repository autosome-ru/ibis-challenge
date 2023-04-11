from ast import Name
import string 
from pathlib import Path
from typing import ClassVar

from dataclasses import dataclass
from bibis.utils import END_LINE_CHARS

class PWMSubmissionFormatException(Exception):
    pass

class PWMSubmissionException(Exception):
    pass

@dataclass
class PFMInfo:
    tag: str
    tf: str
    path: Path
    

@dataclass
class PWMSubmission:
    name: str 
    path: Path | str 
    available_tfs: set[str]
    
    MAX_PWM_PER_TF: ClassVar[int] = 4   
    MAX_TAG_LENGTH: ClassVar[int] = 30 
    POSSIBLE_CHARS: ClassVar[set[str]] = set(string.ascii_lowercase + string.digits + "_")
    MAX_PRECISION: ClassVar[int] = 5
    MAX_1_DIFF: ClassVar[float] = 0.0015
    
    
    def split_into_pfms(self, dir_path: Path | str) -> list[PFMInfo]:
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True, parents=True)
        
        pfms: list[PFMInfo] = []
        for tag, (tf, lines) in  self.split_into_chunks().items():
            pfm_path = dir_path / f"{tag}.pfm"
            if pfm_path.exists():
                raise PWMSubmissionException(f"Pfm path {pfm_path} already exists")
            with pfm_path.open("w") as out:
                print(f">{tf} {tag}", file=out)
                for line in lines:
                    print(line, file=out)
            pfm_info = PFMInfo(tag=tag, 
                               tf=tf,
                               path=pfm_path)
            pfms.append(pfm_info)
        return pfms
            
    def check_tag(self, tag: str, ind: int):
        if len(tag) > self.MAX_TAG_LENGTH:
            raise PWMSubmissionFormatException(f"Tag must be no longer than {self.MAX_TAG_LENGTH}: {tag}, line {ind}")
        for c in tag:
            if c not in self.POSSIBLE_CHARS:
                raise PWMSubmissionFormatException(f"Only alphanumeric and underscore chars are allowed: {c}, line {ind}")
    
    def parse_header(self, header: str, ind: int):
        if not header.startswith(">"):
            raise PWMSubmissionFormatException(f"Header should start with > symbol: {header}, line {ind}")
        header = header[1:]
        fields = header.split(" ")
        if len(fields) != 2:
            raise PWMSubmissionFormatException(f"Header should contain only TF name and unique tag separated by space: {header}, line {ind}")
        tf, tag = fields 
        if not tf in self.available_tfs:
            raise PWMSubmissionFormatException(f"TF provided doesn't exists: {tf}, line {ind}")
        self.check_tag(tag, ind)
        return tf, tag
        
    def check_matrix_line(self, line: str, ind: int):
        fields = line.split(" ")
        if len(fields) != 4:
            raise PWMSubmissionFormatException(f"Each line of matrix should contain 4 numbers separated by spaces: {line}, line {ind}")
        
        s = 0
        for n in fields:
            point_pos = n.find(".")
            if point_pos != -1:
                after_point = n[point_pos+1:]
                if len(after_point) > self.MAX_PRECISION:
                    raise PWMSubmissionFormatException(f"Only up to 5 digits after the decimal point are allowed, {n}, line {ind}")
            try:
                s += float(n)
            except ValueError:
                raise PWMSubmissionFormatException(f"Each frequency must be a real number: {n}, line {ind}")
        if abs(s - 1) > self.MAX_1_DIFF:
            raise PWMSubmissionFormatException(f"Frequences should sum up to 1±0.001: {s}, line {ind}")
    
    def split_into_chunks(self) -> dict[str, tuple[str, list[str]]]:
        chunks: dict[str, tuple[str, list[str]]] = {}
        with open(self.path) as inp:
            waiting_for_header = True
            tf, tag = "", ""
            lines = []
            for ind, line in enumerate(inp, 1):
                line = line.rstrip(END_LINE_CHARS)
                if waiting_for_header:
                    tf, tag = self.parse_header(line, ind)
                    if tag in chunks:
                        raise PWMSubmissionFormatException(f"Tags must be unique: {tag}, line {ind}")
                    waiting_for_header = False
                    lines = []
                elif len(line.strip()) == 0:
                    chunks[tag] = (tf, lines)
                    waiting_for_header = True
                else:
                    if line.startswith(">"):
                        raise PWMSubmissionFormatException(f"Each new matrix should be preceded by extra newline: {line}, line {ind}")
                    self.check_matrix_line(line, ind)
                    lines.append(line)
            if tf == "":
                raise PWMSubmissionFormatException("File is empty")
            if not waiting_for_header:
                chunks[tag] = (tf, lines)
        for tag, (_, lines) in chunks.items():
            if len(lines) == 0:
                raise PWMSubmissionFormatException(f"Each matrix should contain at least one row with nucleotide frequencies: {tag}")
        return chunks
                                 
    def validate(self):
        self.split_into_chunks()