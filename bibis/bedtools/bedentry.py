from typing import ClassVar
from copy import deepcopy
from dataclasses import dataclass, field 
from ..seq.genome import Genome
from ..logging import get_bibis_logger

logger = get_bibis_logger()

@dataclass(order=True, slots=True)
class BedEntry:
    chr: str
    start: int
    end: int
    peak: int | None = field(default=None, compare=False)
    metainfo: dict[str, str] | None = field(default=None, compare=False)
    
    NONE_REPR: ClassVar[str] = '.'
    BED_SEP: ClassVar[str] = "\t"

    def __len__(self) -> int:
        return max(0, self.end - self.start) 

    def __getitem__(self, ind: int) -> int:
        i = self.start + ind
        if i >= self.end:
            raise IndexError('segment object index out of range')
        return i
    
    @classmethod
    def peak2str(cls, peak: int | None) -> str:
        if peak is None:
            return cls.NONE_REPR
        else: # isinstance(peak, int):
            return str(peak)
    
    @classmethod
    def str2peak(cls, s: str) -> int | None:
        if s == cls.NONE_REPR:
            return None
        try:
            return int(s)
        except ValueError:
            pass
        raise Exception("Wrong peak format")

    @classmethod
    def from_line(cls, line: str):
        fields = line.split(cls.BED_SEP)
        if len(fields) < 3:
            raise Exception("Wrong bed format")
        if len(fields) == 3:
            chr, st, end = fields
            return cls(chr, int(st), int(end))
        # len(fields) >= 4
        chr, st, end, peak = fields[0:4]
        return cls(chr, int(st), int(end), cls.str2peak(peak))

    def to_line(self, include_peak: bool=True) -> str:
        fields = [self.chr, str(self.start), str(self.end)]
        if include_peak:
            fields.append(self.peak2str(self.peak))
        return self.BED_SEP.join(fields)

    def truncate(self, shift: int, how: str="both", copy_meta: bool=False) -> 'BedEntry':
        '''
        how - ['left', 'rigth', 'both']
        '''
        s = self.start
        if how != "right":
            s += shift
        e = self.end 
        if how != 'left':
            e -= shift
        if s >= e:
            return self.default_entry()

        if self.peak is None or self.peak < s or self.peak > e:
            peak = None
        else:
            peak = self.peak
        other = self.copy(copy_meta)
        other.start = s
        other.end = e
        other.peak = peak
        if copy_meta:
            other.metainfo = self.metainfo
        return other
    
    def split(self, ind: int, copy_meta: bool=False) -> tuple['BedEntry', 'BedEntry']:
        m = self[ind]
        s1, e1 = self.start, m
        s2, e2 = m, self.end

        if self.peak is None:
            p1, p2 = None, None
        else:
            if self.peak <= m:
                p1, p2 = self.peak, None
            else:
                p1, p2 = None, self.peak

        other1 = self.copy(copy_meta=copy_meta)
        other1.start = s1
        other1.end = e1
        other1.peak = p1
        other2 = self.copy(copy_meta=copy_meta)
        other2.start = s2
        other2.end = e2
        other2.peak = p2
        return other1, other2

    @classmethod
    def from_center(cls, chr: str, cntr: int, radius: int, genome: Genome, metainfo: dict | None = None):
        if metainfo is None:
            metainfo = {}
        st = cntr - radius
        end = cntr + 1 + radius
        st = max(0, st)
        end = min(end, len(genome.chroms[chr]))
        return cls(chr=chr, start=st, end=end, peak=cntr, metainfo=metainfo)

    def copy(self, copy_meta: bool=False) -> 'BedEntry':
        cls = type(self)
        other = cls(chr=self.chr,
                    start=self.start, 
                    end=self.end,
                    peak=self.peak, 
                    metainfo=deepcopy(self.metainfo) if copy_meta else None)
        return other
    
    # extend entry if it is less then width. Otherwise -- return copy
    # done symmetrically 
    def to_min_width(self, width: int, genome: Genome, copy_meta: bool = False) -> 'BedEntry':
        other = self.copy(copy_meta=copy_meta)
        size = len(self)
        if size >= width:
            return other
        rest = width - size
        dv, md = divmod(rest, 2)
        ls, rs = dv + md, dv
        other.start = max(0, other.start - ls)
        other.end = min(len(genome.chroms[other.chr]), other.end+rs)
        if len(other) < width:
            logger.warning("Failed to resize entry to requested size")

        return other

    @classmethod
    def default_entry(cls) -> 'BedEntry':
        if not hasattr(cls, "default"):
            cls.default = cls(chr='', 
                              start=0, 
                              end=0,
                              peak=None,
                              metainfo=None)
        return cls.default