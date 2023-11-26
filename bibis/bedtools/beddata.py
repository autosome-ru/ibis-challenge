import heapq
import tempfile


from bibis.seq.genome import Genome

from .bedentry import BedEntry
from .bedtoolsexecutor import BedClosestMode, BedtoolsExecutor
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Tuple, TypeVar, Union, Callable, Optional, Iterable, ClassVar
from ..utils import END_LINE_CHARS

from dataclasses import dataclass, field 

import numpy as np 
from numpy.random import Generator

U = TypeVar('U')


@dataclass
class BedData:
    entries : list[BedEntry] = field(repr=False, default_factory=list)
    sorted: bool = False
    _executor: Optional[BedtoolsExecutor] = field(default=None, repr=False)
    
    BED_SEP: ClassVar[str] = BedEntry.BED_SEP
    REQUIRED_FIELDS: ClassVar[Tuple[str, str, str, str]] = ("chr", "start", "end", "peak")
    
    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, ind) -> BedEntry:
        return self.entries[ind]

    def __iter__(self) -> Iterator[BedEntry]:
        return iter(self.entries)

    @property
    def executor(self) -> BedtoolsExecutor:
        if self._executor is None:
            if BedtoolsExecutor.DEFAULT_EXECUTOR is None:
                raise Exception("Bedtools executor wans't set")
            self._executor = BedtoolsExecutor.DEFAULT_EXECUTOR
        return self._executor 

    @classmethod
    def from_file(cls, path: Union[Path, str], sort=False, presorted=False, header=False):
        if isinstance(path, str):
            path = Path(path)
        entries = []
        with path.open() as infile:
            if header:
                names = infile.readline().split(cls.BED_SEP)
                for ind, rf in enumerate(cls.REQUIRED_FIELDS):
                    if names[ind] != rf:
                        raise Exception(f"{ind} field must have name {rf}")
                
            for line in infile:
                line = line.rstrip(END_LINE_CHARS)
                entry = BedEntry.from_line(line)
                entries.append(entry)

        self = cls(entries, sorted=sort)
        if sort and not presorted:
            self.sort()
        return self
    
    def write(self, path:Union[Path, str], write_peak: bool=True):
        if isinstance(path, str):
            path = Path(path)
        with path.open("w") as output:
            for e in self.entries:
                line = e.to_line(include_peak=write_peak)
                print(line, file=output, sep="\t")

    def sort(self):
        self.entries.sort()
        self.sorted = True
    
    def merge(self) -> 'BedData':
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            store = tempdir / 'store.bed'
            self.write(store)
            out_path = tempdir / 'out.bed'
            self.executor.merge(store, out_path=out_path)
            return self.from_file(out_path, presorted=True)

    def subtract(self, other: 'BedData', full: bool) -> 'BedData':
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            tmp1 = tempdir / "store1.bed"
            tmp2 = tempdir / "store2.bed"
            self.write(tmp1)
            other.write(tmp2)
            out_path = tempdir / "out.bed"
            self.executor.subtract(tmp1, tmp2, full=full, out_path=out_path)
            return self.from_file(out_path, presorted=True)

    def closest(self, other: 'BedData', how: BedClosestMode) -> list[int]:
        '''
        for each feature in self distance to closest feature in other
        if no such feature exists - return None
        '''
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            store1 = tempdir / "store1.bed"
            store2 = tempdir / "store2.bed"
            self.write(store1)
            other.write(store2)
            out_path = tempdir / "out.bed"
            self.executor.closest(store1, 
                                  store2, 
                                  how=how,
                                  out_path=out_path)
            
            lst = []
            with open(out_path) as inp:
                fields = inp.readline().split()
                dist_ind = len(fields) - 1
                chrom_ind = len(fields) // 2
                inp.seek(0)
                for line in inp:
                    fields = line.rstrip(END_LINE_CHARS).split()
                    chrom = fields[chrom_ind]
                    if chrom == ".":
                        lst.append(None)
                    else:
                        dist = int(fields[dist_ind])
                        if dist < 0 and how is BedClosestMode.DOWNSTREAM:
                            lst.append(None)
                        else:
                            lst.append(dist)
        return lst

    def flank(self, genomesizes: str | Path, size: int) -> 'BedData':
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            store = tempdir / "store.bed"
            self.write(store)
            out_path = tempdir / "out.bed"
            self.executor.flank(store, genomesizes, size, out_path=out_path)
            return BedData.from_file(out_path)
        
    def complement(self, genomesizes: str | Path) -> 'BedData':
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            store = tempdir / "store.bed"
            self.write(store)
            out_path = tempdir / "out.bed"
            self.executor.complement(path=store, 
                                     genomesizes=genomesizes, 
                                     out_path=out_path)
            return BedData.from_file(out_path)
          
    def slop(self, genomesizes: str | Path, shift: int) -> 'BedData':
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            store = tempdir / "store.bed"
            self.write(store)
            out_path = tempdir / "out.bed"
            self.executor.slop(path=store, 
                               genomesizes=genomesizes,
                               shift=shift, 
                               out_path=out_path)
            return BedData.from_file(out_path)

    def append(self, e: BedEntry) -> None:
        self.entries.append(e)
        self.sorted = False

    def pop(self, ind) -> 'BedEntry':
        return self.entries.pop(ind)

    def size(self) -> int:
        return sum(len(e) for e in self.entries)

    def apply(self, fn : Callable[[BedEntry], Optional[BedEntry]]) -> 'BedData':
        new_entries = []
        for s in self.entries:
            s = fn(s)
            if s is not None:
                new_entries.append(s)
        return BedData(new_entries)

    def filter(self, predicate: Callable[[BedEntry], bool]) -> 'BedData':
        flt_entries = []
        for s in self.entries:
            if predicate(s):
                flt_entries.append(s)
        return BedData(flt_entries)

    def global2local(self, global_ind: int) -> tuple[int, int]:
        ind = global_ind
        for si, e in enumerate(self.entries):
            if ind < len(e):
                break
            ind = ind - len(e)
        else: # no break
            raise IndexError("global segmentset index is out of range")
        return si, ind

    def retrieve(self, global_ind: int) -> int:
        si, ind = self.global2local(global_ind)
        return self.entries[si][ind]
    
    def sample(self, k: int, rng: Generator | None = None) -> 'BedData':
        if rng is None:
            rng = np.random.default_rng()
        k = min(k, len(self))
        ch_id = rng.choice(len(self.entries), size=k, replace=False)
        ch_entries = [self.entries[i] for i in ch_id]
        
        return BedData(ch_entries)

    def sample_shades(self, seqsize,  genome: Genome, k: int=1, rng: Generator | None = None) -> list[BedEntry]:
        if rng is None:
            rng = np.random.default_rng()
        shift = seqsize // 2
        other = deepcopy(self)
        smpls = []
        
        while len(smpls) != k:        
            n = other.size()
            if n == 0:
                break
            
            g_ind = rng.choice(n)
            si, ind = other.global2local(g_ind)
            entry = other.pop(si)
            coord = entry[ind]
            
            new_entry = BedEntry.from_center(entry.chr, 
                                             coord, 
                                             shift,
                                             metainfo=entry.metainfo, 
                                             genome=genome)
            if len(new_entry) == seqsize:
                smpls.append(new_entry)
            
            en1, en2 = entry.split(ind)
            en1 = en1.truncate(shift, how='right')
            en2 = en2.truncate(shift, how="left")
            if len(en1) > 0:
                other.append(en1)
            if len(en2) > 0:
                other.append(en2)
        return smpls
    
    def sort_by(self, key_fn) -> 'BedData':
        cls = type(self)
        return cls(sorted(self.entries, key=key_fn), sorted=False)

def join_bed(beds: Iterable[BedData],  sort=True) -> BedData:
    if any(not x.sorted for x in beds):
        entries = []
        for b in beds:
            entries.extend(b.entries)
        if sort:
            entries.sort()
        return BedData(entries, sorted=sort)
    entries = list(heapq.merge(*(b.entries for b in beds)))
    return BedData(entries, sorted=True)