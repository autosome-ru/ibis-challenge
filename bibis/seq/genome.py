import glob 

from dataclasses import dataclass 
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqIO import SeqRecord
from pathlib import Path
from typing import Union, Protocol, Iterable
from .seqentry import SeqEntry
from ..bedtools.constants import CHROM_ORDER

class GenomeInterval(Protocol):
    chr: str
    start: int 
    end: int

@dataclass
class Genome:
    chroms : dict[str, Seq]

    def __len__(self)->int:
        return len(self.chroms)

    def __getitem__(self, interval: GenomeInterval) -> Seq:
        # always use interval so indexing always returns Seq 
        return self.chroms[interval.chr][interval.start:interval.end] # type: ignore  

    def cut(self, it: Iterable[GenomeInterval]) -> list[SeqEntry]:
        return [SeqEntry(self[e], metainfo={"chr": e.chr,
                                            "start": e.start, 
                                            "end": e.end}) for e in it]

    def chrom_sizes(self) -> dict[str, int]:
        return {ch: len(seq) for ch, seq in self.chroms.items()}

    def write_bed_genome_file(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        ch_sizes = self.chrom_sizes()
        with path.open('w') as outp:
            for ch in CHROM_ORDER:
                if ch in ch_sizes:
                    size = ch_sizes[ch]
                    print(ch, size, file=outp, sep="\t")

    @classmethod
    def from_dir(cls, dirpath: Path | str, ext='.fa'):
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        dt = {}
        for chrom in glob.glob(str(dirpath/f"*{ext}")):
            ch_name = Path(chrom).name.replace(ext, "")
            seq = SeqIO.read(chrom, format="fasta")
            dt[ch_name] = Seq(str(seq.seq).upper())
        return cls(dt)
    
    def to_dir(self, dirpath: Path | str, ext=".fa"):
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        for ch_name, seq in self.chroms.items():
            fpath = dirpath / f"{ch_name}{ext}"
            SeqIO.write(SeqRecord(id=ch_name,
                                  seq=seq), 
                        handle=fpath, 
                        format="fasta")      
    
    @classmethod
    def from_fasta(cls, fastapath: Path | str):
        dt = {}
        for rec in SeqIO.parse(fastapath, format="fasta"):
            print(rec)
            dt[rec.name] = rec.seq
        return cls(dt)
    
    def to_fasta(self, fastapath: Path | str):
        records = (SeqRecord(id=ch_name, 
                             seq=seq) for ch_name, seq in self.chroms.items())
        SeqIO.write(records, 
                    handle=fastapath,
                    format="fasta")