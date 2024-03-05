from dataclasses import dataclass, field
from typing import ClassVar
from pathlib import Path

from bibis.bedtools.bedentry import BedEntry
from bibis.bedtools.beddata import BedData

@dataclass
class PeakEntry:
    chrom: str
    start: int
    end: int
    peak: int 
    pileup: int 
    log_pvalue: float
    log_qvalue: float
    fold_enrichment: float
    name: str 
    peakcallers: list[str]
    
    CALLERS_SEP: ClassVar[str] = ","
        
        
    @classmethod
    def from_line(cls, line: str):
        line = line.rstrip("\r\n")
        chrom, start, end, peak, pileup, log_pvalue, fold_enrichment, log_qvalue, name, callers = line.split()
        start = int(start)
        end = int(end)
        peak = int(peak)
        pileup = int(pileup)
        log_pvalue = float(log_pvalue)
        log_qvalue = float(log_qvalue)
        fold_enrichment = float(fold_enrichment)
        peakcallers = callers.split(cls.CALLERS_SEP)
        
        return cls(chrom=chrom,
                   start=start, 
                   end=end,
                   peak=peak,
                   pileup=pileup,
                   log_pvalue=log_pvalue,
                   log_qvalue=log_qvalue,
                   fold_enrichment=fold_enrichment,
                   name=name,
                   peakcallers=peakcallers)
        
    def to_bedentry(self):
        metainfo = dict(name=self.name,
                        pValue=self.log_pvalue,
                        qValue=self.log_qvalue)
        return BedEntry(chr=self.chrom, 
                        start=self.start,
                        end=self.end, 
                        peak=self.peak, 
                        metainfo=metainfo)  #type: ignore 


@dataclass
class PeakList:
    entries: list[PeakEntry] = field(repr=False, default_factory=list)
    
    COMMENT_CHAR: ClassVar[str] = "#"
    
    @classmethod
    def read(cls, fl_path: str | Path):
        if isinstance(fl_path, str):
            fl_path = Path(fl_path)
        entries = []
        with fl_path.open("r") as inp:
            for line in inp:
                if line.startswith(cls.COMMENT_CHAR):
                    continue
                entry = PeakEntry.from_line(line)
                entries.append(entry)
        return cls(entries) #type: ignore
    
    def filter(self, cond_fn):
        entrs = []
        for e in self.entries:
            if cond_fn(e):
                entrs.append(e)
        return PeakList(entrs)
    
    def sort(self, key_fn):
        return PeakList(sorted(self.entries, key=key_fn))
    
    def __len__(self):
        return len(self.entries)  
    
    def __getitem__(self, ind):
        return self.entries[ind]
    
    def to_beddata(self):
        return BedData([e.to_bedentry() for e in self.entries])
