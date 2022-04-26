from typing import Union
from attr import define
from utils import register_type
from pathlib import Path
from utils import END_LINE_CHARS

@register_type
@define
class Sequence:
    seq: str 
    name: str=""
    desc: str=""

    def __repr__(self) -> str:
        if len(self.seq) > 10000:
            return f"Sequence<{self.seq[0:100]}...{self.seq[-100:]}>"
        return f"Sequence<{self.seq}>"

    def __str__(self) -> str:
        return self.seq

    def __getitem__(self, item)-> 'Sequence':
        return Sequence(self.seq.__getitem__(item))

    def __len__(self) -> int:
        return len(self.seq)

    @classmethod
    def from_file(cls, path: Union[Path, str], upper=True):
        '''
        path contains only ONE sequence
        '''
        if isinstance(path, str):
            path = Path(path)
        seq = []
        with path.open() as inp:
            header = inp.readline().lstrip(">").rstrip(END_LINE_CHARS)
            header = header.split(maxsplit=1)
            if len(header) == 1:
                name, desc = header[0], ""
            else:
                name, desc = header
            seq = "".join(line.strip().upper() for line in inp)
        return cls(seq, name, desc)