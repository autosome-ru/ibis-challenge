import shlex
import subprocess
from typing import ClassVar, Optional, Union

from utils import temporary_file
from attr import define 
from pathlib import Path
from enum import Enum

class BedClosestMode(Enum):
    UPSTREAM = "-fu"
    DOWNSTREAM = "-fd"
    ALL = ""

@define
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

    @staticmethod
    def run_bedtools_cmd(cmd: str, name: str=""):
        out_path = temporary_file()
        args = shlex.split(cmd)
        with open(out_path, "w") as outp:
            r = subprocess.run(args, stdout=outp, stderr=subprocess.PIPE, text=True)
        if r.stderr:
            raise Exception(f"Bedtools {name} returned error: {r.stderr}")
        return out_path

    def merge(self, path: Path) -> 'Path':
        cmd = f"{self.merge_path} -i {path}" # remove peak columns as ambiguous
        out_path = self.run_bedtools_cmd(cmd, name='merge')
        return out_path

    def subtract(self, a: Path, b: Path, full=False) -> Path:
        cmd = f"{self.subtract_path} {'-A' if full else ''} -a {a} -b {b}"
        out_path = self.run_bedtools_cmd(cmd, name='subtract')
        return out_path

    def closest(self, a: Path, b: Path, how: BedClosestMode) -> Path:
        cmd = f"{self.closest_path} -D ref {how.value} -a {a} -b {b}"
        out_path = self.run_bedtools_cmd(cmd, name='closest')
        return out_path

    def flank(self, path: Path, genomesizes: Path, size: int) -> Path:
        cmd = f"{self.flank_path} -i {path} -g {genomesizes} -b {size}"
        out_path = self.run_bedtools_cmd(cmd, name="flank")
        return out_path

    @classmethod
    def set_defaull_executor(cls, executor: Union[str, Path, 'BedtoolsExecutor']):
        if isinstance(executor, str):
            executor = Path(executor)
        if isinstance(executor, Path):
            executor = BedtoolsExecutor(executor)
        cls.DEFAULT_EXECUTOR = executor