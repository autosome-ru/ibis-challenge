from attrs import define
from pathlib import Path
from typing import ClassVar, Union
import numpy as np
import numpy.typing as npt

from utils import END_LINE_CHARS

@define
class PFM:
    name: str
    description: str
    matrix: npt.NDArray[np.float32]

    FLOAT_FMT: ClassVar[str] = '%.5f'
    EPS: ClassVar[float] = 1e-5
    SIGNIGICANT_DIGITS=5

    @classmethod
    def load(cls, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        with path.open() as inp:
            header = inp.readline().strip(END_LINE_CHARS)
            fields = header.split(maxsplit=1)
            if len(fields) == 1:
                name, description = fields[0], ""
            else:
                name, description = fields
            name = name.lstrip(">")
            matrix = np.loadtxt(inp, dtype=np.float32)
        return cls(name, description, matrix)

    def header(self):
        if not self.description:
            return f">{self.name}"
        else:
            return f">{self.name} {self.description}"
    
    def write(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        with path.open("w") as out:
            header = self.header()
            print(header, file=out)
            np.savetxt(out, self.matrix, fmt=self.FLOAT_FMT)

    @classmethod
    def pfm2pwm(cls, matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        matrix = np.log2((matrix + cls.EPS)/ 0.25)
        return matrix 

    @classmethod
    def pwm2intpwm(cls, pwm: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
        intpwm = pwm * (10 ** cls.SIGNIGICANT_DIGITS)
        intpwm = np.asarray(np.round(intpwm), dtype=np.int32)
        return intpwm

    def intpwm(self) -> 'IntPWM':
        pwm = self.pfm2pwm(self.matrix)
        pwm = self.pwm2intpwm(pwm)
        return IntPWM(self.name, self.description, pwm) 

@define
class IntPWM:
    name: str
    description: str
    matrix: npt.NDArray[np.int32]

    @classmethod
    def load(cls, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        with path.open() as inp:
            header = inp.readline().strip(END_LINE_CHARS)
            fields = header.split(maxsplit=1)
            if len(fields) == 1:
                name, description = fields[0], ""
            else:
                name, description = fields
            name = name.lstrip(">")
            matrix = np.loadtxt(inp, dtype=np.int32)
        return cls(name, description, matrix)

    def header(self):
        if not self.description:
            return f">{self.name}"
        else:
            return f">{self.name} {self.description}"
    
    def write(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        with path.open("w") as out:
            header = self.header()
            print(header, file=out)
            np.savetxt(out, self.matrix, fmt="%d")

if __name__ == "__main__":
    pfm = PFM.load("/home_local/dpenzar/ibis-challenge/benchmark/example.pwm")
    pfm.intpwm().write("a.txt")
    pfm = PFM.load('/home_local/vorontsovie/greco-motifs/release_7d_motifs_2021-12-21/LEUTX.FL@CHS@paltry-buff-snail@HughesLab.Streme@Motif2.ppm')
    pfm.intpwm().write("b.txt")
    pwm = IntPWM.load("b.txt")
    pwm.write("c.txt")