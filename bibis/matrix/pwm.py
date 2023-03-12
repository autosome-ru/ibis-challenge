from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Union
import numpy as np
import numpy.typing as npt

from ..utils import END_LINE_CHARS

@dataclass
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
            matrix = np.round(matrix, cls.SIGNIGICANT_DIGITS)
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

    def pwm(self) -> 'PWM':
        pwm = self.pfm2pwm(self.matrix)
        return PWM(self.name, self.description, pwm) 
    
@dataclass
class PWM:
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
            matrix = np.round(matrix, cls.SIGNIGICANT_DIGITS)
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
    def pwm2intpwm(cls, pwm: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
        intpwm = pwm * (10 ** cls.SIGNIGICANT_DIGITS)
        intpwm = np.asarray(np.round(intpwm), dtype=np.int32)
        return intpwm

    def intpwm(self) -> 'IntPWM':
        pwm = self.pwm2intpwm(self.matrix)
        return IntPWM(self.name, self.description, pwm) 


@dataclass
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


@dataclass
class PCM:
    name: str
    description: str
    matrix: npt.NDArray[np.float32]
    PSEUDO_COUNT = 1
    FLOAT_FMT: ClassVar[str] = '%.5f'

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

    def pfm(self) -> 'PFM':
        m = self.matrix  + self.PSEUDO_COUNT
        tot = m.sum(axis=1, keepdims=True)
        m = m / tot
        return PFM(self.name, self.description, m)
    
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