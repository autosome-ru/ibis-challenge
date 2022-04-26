import genome 
import random
import heapq

from attr import define, field 
from bedentry import BedEntry
from bedtoolsexecutor import BedClosestMode, BedtoolsExecutor
from copy import deepcopy
from pathlib import Path
from typing import ClassVar, Iterator, TypeVar, Union, Callable, Optional, Iterable
from utils import END_LINE_CHARS, temporary_file

 
from genome import Genome

U = TypeVar('U')

@define
class BedData:
    entries : list[BedEntry] = field(repr=False, factory=list)
    sorted: bool = False
    _executor: Optional[BedtoolsExecutor] = field(default=None, repr=False)

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
                infile.readline()
            for line in infile:
                line = line.rstrip(END_LINE_CHARS)
                entry = BedEntry.from_line(line)
                entries.append(entry)
        self = cls(entries, sorted=sort)
        if sort and not presorted:
            self.sort()
        return self

    def sort(self):
        self.entries.sort()
        self.sorted = True

    def write(self, path:Union[Path, str], write_peak: bool=True):
        if isinstance(path, str):
            path = Path(path)
        with path.open("w") as output:
            for e in self.entries:
                line = e.to_line(include_peak=write_peak)
                print(line, file=output, sep="\t")

    def merge(self) -> 'BedData':
        store = temporary_file()
        self.write(store)
        out_path = temporary_file()
        out_path = self.executor.merge(store)
        return self.from_file(out_path, presorted=True)

    def subtract(self, other: 'BedData', full: bool) -> 'BedData':
        tmp1 = temporary_file()
        tmp2 = temporary_file()
        self.write(tmp1)
        other.write(tmp2)
        out_path = self.executor.subtract(tmp1, tmp2, full=full)
        return self.from_file(out_path, presorted=True)

    def closest(self, other: 'BedData', how: BedClosestMode) -> list[int]:
        '''
        for each feature in self distance to closest feature in other
        if no such feature exists - return None
        '''
        store1 = temporary_file()
        store2 = temporary_file()
        self.write(store1)
        other.write(store2)
        out_path = self.executor.closest(store1, store2, how=how)
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

    def flank(self, genome: Union[str, Path, genome.Genome], size: int) -> 'BedData':
        if isinstance(genome, Genome):
            genomesizes = temporary_file()
            genome.write_bed_genome_file(genomesizes)
            genome = genomesizes
        elif isinstance(genome, str):
            genome = Path(genome)
        elif not isinstance(genome, Path):
            raise Exception(f"Wrong type of genome argument: {type(genome)}")
        
        store = temporary_file()
        self.write(store)
        out_path = self.executor.flank(store, genome, size)
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
    
    def map(self, fn : Callable[[BedEntry], Optional[U]]) -> list[U]:
        lst = []
        for s in self.entries:
            s = fn(s)
            if s is not None:
                lst.append(s)
        return lst

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
    
    def sample(self, k: int) -> 'BedData':
        k = min(k, len(self))
        return BedData(random.sample(self.entries, k))

    def sample_shades(self, seg_size, k: int=1) -> list[BedEntry]:
        shift = seg_size // 2
        other = deepcopy(self)
        smpls = []
        for _ in range(k):        
            n = other.size()
            if n == 0:
                break

            g_ind = random.sample(range(n), k=1)[0]
            si, ind = other.global2local(g_ind)
            entry = other.pop(si)
            coord = entry[ind]
            
            new_entry = BedEntry.from_center(entry.chr, coord, shift)
            smpls.append(new_entry)
            
            en1, en2 = entry.split(ind)
            en1 = en1.truncate(shift, how='right')
            en2 = en2.truncate(shift, how="left")
            if len(en1) > 0:
                other.append(en1)
            if len(en2) > 0:
                other.append(en2)
        return smpls

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