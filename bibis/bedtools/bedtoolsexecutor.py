import shlex
import subprocess
from typing import ClassVar, Optional, Union

from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from ..plogging import get_bibis_logger
logger = get_bibis_logger()


class BedClosestMode(Enum):
    UPSTREAM = "-fu"
    DOWNSTREAM = "-fd"
    ALL = ""


#TODO write tests for methods
@dataclass
class BedtoolsExecutor:
    bedtools_root: Path

    DEFAULT_EXECUTOR: ClassVar[Optional['BedtoolsExecutor']] = None

    @property
    def merge_path(self) -> Path:
        return self.bedtools_root / 'mergeBed'

    @property
    def sort_path(self) -> Path:
        return self.bedtools_root / 'sortBed'

    @property
    def subtract_path(self) -> Path:
        return self.bedtools_root / 'subtractBed'

    @property
    def closest_path(self) -> Path:
        return self.bedtools_root / 'closestBed'

    @property
    def flank_path(self) -> Path:
        return self.bedtools_root / 'flankBed'
    
    @property
    def complement_path(self) -> Path:
        return self.bedtools_root / 'complementBed'
    
    @property
    def slop_path(self) -> Path:
        return self.bedtools_root / 'slopBed'

    @staticmethod
    def _run_bedtools_cmd(cmd: str, out_path: Path | str, name: str=""):
        args = shlex.split(cmd)
        with open(out_path, "w") as outp:
            r = subprocess.run(args, 
                               stdout=outp, 
                               stderr=subprocess.PIPE,
                               text=True)
        if r.stderr:
            raise Exception(f"Bedtools {name} returned error: {r.stderr}")

    def merge(self, inpath: str | Path, out_path: Path | str):
        cmd = f"{self.merge_path} -i {inpath}" # remove peak columns as ambiguous
        self._run_bedtools_cmd(cmd, 
                               out_path=out_path,
                               name='merge')


    def subtract(self, a: str | Path, b: str | Path, out_path: str | Path, full=False):
        cmd = f"{self.subtract_path} {'-A' if full else ''} -a {a} -b {b}"
        self._run_bedtools_cmd(cmd, 
                               out_path=out_path, 
                               name='subtract')

    def closest(self, a: str | Path, b: str | Path, how: BedClosestMode, out_path: str | Path):
        cmd = f"{self.closest_path} -D ref {how.value} -a {a} -b {b}"
        self._run_bedtools_cmd(cmd,
                               out_path=out_path,
                               name='closest')

    def flank(self, path: str | Path, genomesizes: str | Path, size: int, out_path: str | Path):
        cmd = f"{self.flank_path} -i {path} -g {genomesizes} -b {size}"
        self._run_bedtools_cmd(cmd, 
                               out_path=out_path,
                               name="flank")
        
    def complement(self, path: str | Path, genomesizes: str | Path, out_path: str | Path):
        cmd = f"{self.complement_path} -i {path} -g {genomesizes}"
        self._run_bedtools_cmd(cmd, 
                               out_path=out_path,
                               name="complement")
        
    def slop(self, path: str | Path, genomesizes: str | Path, shift: int, out_path: str | Path):
        cmd = f"{self.slop_path} -i {path} -g {genomesizes} -b {shift}"
        self._run_bedtools_cmd(cmd, 
                               out_path=out_path,
                               name="slop")

    @classmethod
    def set_defaull_executor(cls, executor: Union[str, Path, 'BedtoolsExecutor']):
        if isinstance(executor, str):
            executor = Path(executor)
        if isinstance(executor, Path):
            executor = BedtoolsExecutor(executor)
        cls.DEFAULT_EXECUTOR = executor

    def merge_keeppeak(self, 
                       path: str | Path, 
                       out_path: str | Path):
        cmd = f"{self.merge_path} -i {path} -c 4 -o mean"
        self._run_bedtools_cmd(cmd, 
                               out_path=out_path,
                               name="merge")
        logger.info(f"Using {executor.bedtools_root} as bedtools executor")
        cls.DEFAULT_EXECUTOR = executor

    @property
    def intersect_path(self) -> Path:
        return self.bedtools_root / 'intersectBed'
    

    def full_intersect(self, a: str | Path, b: str | Path, out_path: str | Path):
        cmd = f"{self.intersect_path} -a {a} -b {b} -f 1.0"
        self._run_bedtools_cmd(cmd, 
                               out_path=out_path, 
                               name='full_intersect')
   
