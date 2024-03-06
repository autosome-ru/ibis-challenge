from logging import warning
from pyclbr import Class
import sys
import string 
from pathlib import Path
from typing import ClassVar
from collections import Counter
from dataclasses import dataclass

from .benchmarkconfig import BenchmarkConfig
from .val import ValidationResult

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
    POSSIBLE_CHARS: ClassVar[set[str]] = set(string.ascii_lowercase + string.ascii_uppercase + string.digits + "_")
    MAX_PRECISION: ClassVar[int] = 5
    MAX_1_DIFF: ClassVar[float] = 0.0015
    MIN_PWM_LENGTH: ClassVar[int] = 5
    MAX_PWM_LENGTH: ClassVar[int] = 30
    
    
    def split_into_pfms(self, dir_path: Path | str) -> list[PFMInfo]:
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True, parents=True)
        
        pfms: list[PFMInfo] = []
        
        chunks, _ = self.split_into_chunks()
        for tag, (tf, lines) in  chunks.items():
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
    
    def parse_header(self, header: str, ind: int) -> tuple[str, str, str | None] :
        if not header.startswith(">"):
            raise PWMSubmissionFormatException(f"Header should start with > symbol: {header}, line {ind}")
        header = header[1:]
        fields = header.split(" ")
        if len(fields) != 2:
            raise PWMSubmissionFormatException(f"Header should contain only TF name and unique tag separated by space: {header}, line {ind}")
        tf, tag = fields 
        
        if not tf in self.available_tfs:
            warn =  f"TF provided doesn't exists: {tf}, line {ind}"
        else:
            warn = None

        self.check_tag(tag, ind)
        return tf, tag, warn
        
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
    
    def split_into_chunks(self) -> tuple[dict[str, tuple[str, list[str]]], list[str]]:
        chunks: dict[str, tuple[str, list[str]]] = {}
        warnings = []
        with open(self.path) as inp:
            waiting_for_header = True
            tf, tag = "", ""
            lines = []
            for ind, line in enumerate(inp, 1):
                line = line.rstrip(END_LINE_CHARS)
                if waiting_for_header:
                    tf, tag, warn = self.parse_header(line, ind)
                    if warn is not None:
                        warnings.append(warn)
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
            pwm_len = len(lines)
            if pwm_len > self.MAX_PWM_LENGTH:
                raise PWMSubmissionFormatException(f"Each matrix should contain no more than {self.MAX_PWM_LENGTH} rows with nucleotide frequencies: {tag}")
            if pwm_len < self.MIN_PWM_LENGTH:
                raise PWMSubmissionFormatException(f"Each matrix should contain at least {self.MIN_PWM_LENGTH} rows with nucleotide frequencies: {tag}")
        return chunks, warnings
                                 
    def validate(self, cfg: BenchmarkConfig) -> ValidationResult:
        chunks, warnings = self.split_into_chunks()
            
        submmitted_tfs =  Counter(tf for tf, _ in chunks.values())
        
        top_sub_tf, sub_cnt = submmitted_tfs.most_common(1)[0]
        if  sub_cnt > self.MAX_PWM_PER_TF:
            raise PWMSubmissionFormatException(f"{sub_cnt} pwms for {top_sub_tf} provided, only {self.MAX_PWM_PER_TF} is allowed")
        
        for tf in self.available_tfs:
            if not tf in submmitted_tfs:
                msg = f"no PWM submitted for {tf}"
                warnings.append(msg)
        return ValidationResult(warnings=warnings, errors=[])
        
        