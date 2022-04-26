import glob 

from attr import define 
from sequence import Sequence
from pathlib import Path
from typing import Union, Protocol, Iterable

class GenomeInterval(Protocol):
    chr: str
    start: int 
    end: int

@define
class Genome:
    chroms : dict[str, Sequence]

    def __len__(self)->int:
        return len(self.chroms)

    def __getitem__(self, interval: GenomeInterval) -> Sequence:
        return self.chroms[interval.chr][interval.start:interval.end]

    def cut(self, it: Iterable[GenomeInterval]) -> list[Sequence]:
        return [self[e] for e in it]

    def chrom_sizes(self) -> dict[str, int]:
        return {ch: len(seq.seq) for ch, seq in self.chroms.items()}

    def write_bed_genome_file(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        ch_sizes = self.chrom_sizes()
        with path.open('w') as outp:
            for ch, size in ch_sizes.items():
                print(ch, size, file=outp, sep="\t")

    @classmethod
    def from_dir(cls, dirpath:Path, ext='.fa'):
        dt = {}
        for chrom in glob.glob(str(dirpath/f"*{ext}")):

            ch_name = Path(chrom).name.replace(ext, "")
            seq = Sequence.from_file(chrom)
            dt[ch_name] = seq

        return cls(dt)