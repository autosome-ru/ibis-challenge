from dataclasses import dataclass
from copy import copy 

from ..seq.seqentry import SeqEntry
from ..seq.genome import Genome
from ..bedtools.beddata import BedData, join_bed
from ..bedtools.bedentry import BedEntry
from ..sampling.gc import SetGCSampler, GenomeGCSampler

def _cut_to_window(bed: BedData, window_size: int, genome: Genome) -> BedData:
        cut_peaks = []
        for p in bed:
            if p.peak is None:
                cnt = (p.end + p.start) // 2
                print("Entry has no peak, will use the middle of the interval")
            else:
                cnt = p.peak
            entry = BedEntry.from_center(chr=p.chr, 
                                         cntr=cnt, 
                                         radius=window_size // 2, 
                                         genome=genome)
            if len(entry) == window_size:
                cut_peaks.append(entry)
        return BedData(cut_peaks)
        

@dataclass
class ChIPForeignSampler:
    window_size: int
    sampler: SetGCSampler
    positives: list[SeqEntry]
    sample_per_object: int = 1
    seed: int = 777 
    
    @classmethod
    def make(cls, 
             window_size: int, 
             genome: Genome,
             tf_peaks: list[BedData],
             real_peaks: list[BedData],
             friend_peaks: list[BedData] | None = None,
             black_list_regions: BedData | None = None,
             sample_per_object: int = 1,
             seed = 777):
        assert window_size % 2 == 1, "Window size must be odd"
        prohibit_peaks = copy(tf_peaks)
        if friend_peaks is not None:
            prohibit_peaks.extend(friend_peaks)
        if black_list_regions is not None:
            prohibit_peaks.append(black_list_regions)
        prohibit = join_bed(prohibit_peaks)
        real = join_bed(real_peaks)
        foreign = real.subtract(prohibit, 
                                full=True)
       
        
        foreign = _cut_to_window(foreign,
                                 window_size=window_size,
                                 genome=genome) 
            
        foreign_seqs = genome.cut(foreign)
        
        sampler = SetGCSampler.make(negatives=foreign_seqs,
                  sample_per_object=sample_per_object,
                  seed=seed)
        pos_peaks = max(tf_peaks, key=lambda x: len(x))
        pos_peaks = _cut_to_window(pos_peaks, 
                                       window_size=window_size,
                                       genome=genome)
        positives = genome.cut(pos_peaks)
        
        
        return cls(window_size=window_size,
                   sampler=sampler, 
                   positives=positives, 
                   sample_per_object=sample_per_object, 
                   seed=seed)
    
    def sample(self) -> list[SeqEntry]:
        negs = self.sampler.sample(self.positives) # type: ignore 
        for n in negs:
            n.sequence = n.sequence.upper() # type: ignore 
        return negs # type: ignore 
        
    
from ..sampling.shades import ShadesSampler

@dataclass
class ChIPShadesSampler:
    sampler: ShadesSampler
    genome: Genome 
    window_size: int
    max_dist: int
    sample_per_object: int = 1
    seed: int = 777
    
    @classmethod
    def make(cls, 
             window_size: int, 
             genome: Genome, 
             tf_peaks: list[BedData],
             friend_peaks: list[BedData] | None = None,
             black_list_regions: BedData | None = None,
             sample_per_object: int = 1,
             max_dist=301,
             seed = 777):
        assert window_size % 2 == 1, "Window size must be odd"
        prohibit_peaks = copy(tf_peaks)
        if friend_peaks is not None:
            prohibit_peaks.extend(friend_peaks)
        if black_list_regions is not None:
            prohibit_peaks.append(black_list_regions)
        prohibit = join_bed(prohibit_peaks)
        
        pos_peaks = max(tf_peaks, key=lambda x: len(x))
       
      
        sampler = ShadesSampler.make(pos_ds = pos_peaks,
                                     black_regions=prohibit,
                                     max_dist=max_dist,
                                     peak_size=window_size,
                                     genome=genome,
                                     sample_per_peak=sample_per_object)
        
        return cls(sampler=sampler,
                   genome=genome,
                   window_size=window_size,
                   max_dist=max_dist,
                   sample_per_object=sample_per_object,
                   seed=seed)

    def sample(self) -> list[SeqEntry]:
        smpls = self.sampler.sample()
        return self.genome.cut(smpls)

@dataclass
class ChIPGenomeSampler:
    window_size: int
    genome: Genome 
    sampler: GenomeGCSampler
    positives: BedData
    sample_per_object: int
    seed: int
    exact: bool
    
    
    @classmethod
    def make(cls, 
             window_size: int, 
             genome: Genome,
             tf_peaks: list[BedData],
             friend_peaks: list[BedData] | None = None,
             black_list_regions: BedData | None = None,
             sample_per_object: int = 1,
             exact: bool = True,
             max_overlap: int | None = None,
             n_procs: int = 1,
             seed = 777):
        assert window_size % 2 == 1, "Window size must be odd"
        
        prohibit_peaks = copy(tf_peaks)
        if friend_peaks is not None:
            prohibit_peaks.extend(friend_peaks)
        if black_list_regions is not None:
            prohibit_peaks.append(black_list_regions)
            

        prohibit = join_bed(prohibit_peaks)
        
        sampler = GenomeGCSampler.from_bed(genome=genome,
                                           blacklist_regions=prohibit,
                                           window_size=window_size,
                                           exclude_positives=False,
                                           max_overlap=max_overlap, 
                                           sample_per_object=sample_per_object,
                                           exact=exact,
                                           seed=seed,
                                           n_procs=n_procs)
        
        pos_peaks = max(tf_peaks, key=lambda x: len(x))
        pos_peaks = _cut_to_window(pos_peaks, 
                                   window_size=window_size,
                                   genome=genome)
        
        
        return cls(window_size=window_size,
                   genome=genome,
                   sampler=sampler,
                   positives=pos_peaks,
                   sample_per_object=sample_per_object,
                   exact=exact,
                   seed=seed)
        
    def sample(self) -> list[SeqEntry]:
        return self.sampler.sample(self.positives)
        