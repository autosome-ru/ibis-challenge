from attr import define, field
from typing import Optional, ClassVar, Union

@define(order=True, slots=True)
class BedEntry:
    chr: str
    start: int
    end: int = field(eq=False)
    peak: Optional[int] = field(default=None, eq=False)

    NONE_REPR: ClassVar[str] = '.'
    BED_SEP: ClassVar[str] = "\t"

    @classmethod
    def peak2str(cls, peak: Union[int, None]) -> str:
        if peak is None:
            return cls.NONE_REPR
        else: # isinstance(peak, int):
            return str(peak)
    
    @classmethod
    def str2peak(cls, s: str) -> Union[int, None]:
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

    def __len__(self) -> int:
        return max(0, self.end - self.start) 

    def __getitem__(self, ind: int) -> int:
        i = self.start + ind
        if i >= self.end:
            raise IndexError('segment object index out of range')
        return i

    def truncate(self, shift: int, how: str="both") -> 'BedEntry':
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
            return BedEntry('', 0, 0, None)

        if self.peak is None or self.peak < s or self.peak > e:
            peak = None
        else:
            peak = self.peak
        return BedEntry(self.chr, s, e, peak) 
    
    def split(self, ind: int) -> tuple['BedEntry', 'BedEntry']:
        m = self[ind]
        s1, e1 = self.start, m
        s2, e2 = m, self.end

        if self.peak is None:
            p1, p2 = None, None
        else:
            if self.peak < m:
                p1, p2 = self.peak, None
            else:
                p1, p2 = None, self.peak

        return BedEntry(self.chr, s1, e1, p1), BedEntry(self.chr, s2, e2, p2)

    def expand(self, shift: int) -> 'BedEntry':
        st, end = self.start, self.end
        st -= shift
        end += shift
        return BedEntry(self.chr, st, end, self.peak)

    @classmethod
    def from_center(cls, chr: str, cntr: int, radius: int):
        st = cntr - radius
        end = cntr + 1 + radius
        return cls(chr, st, end, cntr)