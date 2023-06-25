import numpy as np 

from bibis.bedtools.beddata import BedData, BedEntry
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class NarrowPeakEntry:
    chrom: str
    chromStart: int
    chromEnd: int
    name: str
    score: int
    strand: str
    signalValue: float
    pValue: float
    qValue: float
    peak: int
        
        
    @classmethod
    def from_line(cls, line: str):
        line = line.rstrip("\r\n")
        chrom, chromStart, chromEnd, name, score, strand, signalValue, pValue, qValue, peak = line.split()
        chromStart = int(chromStart)
        chromEnd = int(chromEnd)
        score = int(score)
        signalValue = float(signalValue)
        pValue = float(pValue)
        qValue = float(qValue)
        peak = int(peak)
        assert peak != -1
        return cls(chrom, chromStart, chromEnd, name, score, strand, signalValue, pValue, qValue, peak)
    
    def to_bedentry(self, score='qValue'):
        metainfo = dict(name=self.name,
                        pValue=str(self.pValue),
                        qValue=str(self.qValue))
        return BedEntry(chr=self.chrom, 
                        start=self.chromStart,
                        end=self.chromEnd, 
                        peak=self.peak + self.chromStart,
                        metainfo=metainfo) 


@dataclass
class NarrowPeakList:
    entries: list[NarrowPeakEntry] = field(repr=False, default_factory=list)
        
    
    @classmethod
    def read(cls, fl_path: str | Path):
        if isinstance(fl_path, str):
            fl_path = Path(fl_path)
        entries = []
        with fl_path.open("r") as inp:
            for line in inp:
                entry = NarrowPeakEntry.from_line(line)
                entries.append(entry)
        return cls(entries)
    
    def filter(self, cond_fn):
        entrs = []
        for e in self.entries:
            if cond_fn(e):
                entrs.append(e)
        return NarrowPeakList(entrs)
    
    def sort(self, key_fn):
        return NarrowPeakList(sorted(self.entries, key=key_fn))
    
    def __len__(self):
        return len(self.entries)  
    
    def to_beddata(self):
        return BedData([e.to_bedentry() for e in self.entries])
    
    def __getitem__(self, ind):
        return self.entries[ind]
