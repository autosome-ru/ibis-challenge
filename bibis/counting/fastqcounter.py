import json
import gzip 
import sys 
import tqdm 
import heapq

from itertools import groupby
from multiprocessing import Pool
from Bio import SeqIO
from copy import copy
from collections import Counter 
from dataclasses import dataclass, asdict
from typing import ClassVar
from pathlib import Path 
from copy import copy 
from ..utils import END_LINE_CHARS

@dataclass(slots=True, order=True)
class CounterEntry:
    seq: str
    file_ind: int
    cnt: int 

@dataclass
class FastqGzReadsCounter:
    index: list[str]
    db: dict[str, str]
    mapdir: Path
    n_jobs: int

    READS_EXT: ClassVar[str] = '.fastq.gz'
    COUNTS_EXT: ClassVar[str] = ".cnts.txt"
    FIELD_SEP = "\t"

    @classmethod
    def create(cls, mapdir: str, n_jobs: int):
        return cls(index=[], 
                   db={},
                   mapdir=Path(mapdir), 
                   n_jobs=n_jobs)
    
    def to_dict(self):
        dt = asdict(self)
        dt['mapdir'] = str(self.mapdir)
        return dt 
    
    @classmethod
    def from_dict(cls, dt):
        dt = copy(dt) 
        dt['mapdir'] = Path(dt['mapdir'])
        return cls(**dt)
    
    def dump(self, outp):
        dt = self.to_dict()
        with open(outp, "w") as out:
            json.dump(dt, out, indent=4)

    @classmethod
    def load(cls, inp):
        with open(inp, "r") as infd:
            dt = json.load(infd)
        return cls.from_dict(dt)

    def parse_entry(self, line): 
        seq, cnt = line.strip(END_LINE_CHARS).split(self.FIELD_SEP)
        cnt = int(cnt)
        return seq, cnt

    def read_reads(self, in_path) -> list[str]:
        seqs = []
        with gzip.open(in_path, "rt") as inp:
            for rec in SeqIO.parse(inp, 'fastq'):
                seqs.append(rec.seq)
        return seqs 
    
    def get_count_path(self, p: str | Path) -> str:
        nm = Path(p).name.replace(self.READS_EXT, '')
        out_p = self.mapdir / f"{nm}{self.COUNTS_EXT}"
        return str(out_p)
    
    def single_count(self, in_path: Path | str) -> tuple[Path, Path]:
        out_path = self.get_count_path(in_path)
        cnt = Counter(self.read_reads(in_path))
        keys = sorted(cnt)
        with open(out_path, "w") as out:
            for key in keys:
                occ = cnt[key]
                print(key, occ, file=out, sep=self.FIELD_SEP)
        return in_path, out_path
    
    def add(self, ent: Path | str | list[Path | str]):
        self.mapdir.mkdir(exist_ok=True, parents=True)
        if not isinstance(ent, list):
            entries = [str(ent)]  
        else:
            entries = [str(e) for e in ent]

        filtered_entries = []
        for en in entries:
            if en in self.db:
                print(f"Skipping {en}, calculated", file=sys.stdout)  
            else:
                filtered_entries.append(en)
        entries = filtered_entries

        if self.n_jobs == 1:
            dones = self.single_count_all(entries)
        else:
            dones = self.parallel_count_all(entries)
        
        for in_path in entries:
            out_path = dones[in_path]
            self.index.append(in_path)
            self.db[in_path] = out_path
            
        return dones

    def single_count_all(self, entries) -> None:
        dones = {}
        pbar = tqdm.tqdm(entries)
        for p in pbar:
            inp, outp = self.single_count(p)
            dones[inp] = outp
            pbar.set_description(f"Processing {p.name}")
        return dones

    def parallel_count_all(self, entries) -> None:
        self.mapdir.mkdir(exist_ok=True, parents=True)
        dones = {}
        pbar = tqdm.tqdm(total=len(entries))

        with Pool(processes=self.n_jobs) as pool:
            for inp, outp in pool.imap_unordered(self.single_count, entries):
                pbar.update(1)
                pbar.set_description(f"Finished {Path(inp).name}")
                dones[inp] = outp
        return dones 

    def create_count_iterator(self, ind: int):
        p = self.index[ind]
        cnts = self.db[p]
        with open(cnts, "r") as infile:
            for line in infile:
                seq, cnt = self.parse_entry(line.rstrip(END_LINE_CHARS))
                yield CounterEntry(seq=seq, 
                                   file_ind=ind, 
                                   cnt=cnt)

    def reduce(self, out_path: str | Path, reduce_fn):
        its = []
        for ind in range(len(self.index)):
            its.append(self.create_count_iterator(ind))
            
        out_p = Path(out_path)
        with open(out_p, "w") as out:
            for seq, g in tqdm.tqdm(groupby(heapq.merge(*its), key=lambda x : x.seq)):
                rd_res = reduce_fn(g)
                print(seq, rd_res, file=out, sep="\t")