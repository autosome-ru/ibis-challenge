import tempfile
from dataclasses import dataclass
from copy import copy 
from pathlib import Path


from ..seq.seqentry import SeqEntry, drop_duplicates
from ..seq.genome import Genome
from ..bedtools.beddata import BedData, join_bed
from ..bedtools.bedentry import BedEntry
from ..sampling.gc import SetGCSampler, GenomeGCSampler
from ..logging import get_bibis_logger

logger = get_bibis_logger()

def cut_to_window(bed: BedData, window_size: int, genome: Genome) -> BedData:
    cut_peaks = []
    for p in bed:
        if p.peak is None:
            cnt = (p.end + p.start) // 2
            logger.info("Entry has no peak, will use the middle of the interval")
        else:
            cnt = p.peak
        entry = BedEntry.from_center(chr=p.chr, 
                                     cntr=cnt, 
                                     radius=window_size // 2, 
                                     genome=genome,
                                     metainfo=p.metainfo)
        if len(entry) == window_size:
            cut_peaks.append(entry)
        else:
            logger.info("Skipping entry as it out of genome after resizing")
    return BedData(cut_peaks)

@dataclass
class PeakForeignSampler:
    window_size: int
    sampler: SetGCSampler
    positives: list[SeqEntry]
    min_dist: int
    sample_per_object: int
    seed: int = 777 
    
    @classmethod
    def make(cls, 
             window_size: int, 
             genome: Genome,
             tf_peaks: list[BedData],
             real_peaks: list[BedData],
             friend_peaks: list[BedData] | None = None,
             black_list_regions: BedData | None = None,
             min_dist: int = 300,
             sample_per_object: int = 1,
             seed = 777):
        assert window_size % 2 == 1, "Window size must be odd"
        assert min_dist >= 0, "Min dist must >= 0"
        if min_dist == 0:
            prohibit_peaks = copy(tf_peaks)
        else:
            tf_peak = join_bed(tf_peaks)
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                genomesizes = tempdir / "genomesizes.txt"
                genome.write_bed_genome_file(genomesizes)
                prohibit_peaks = [tf_peak.slop(genomesizes=genomesizes,
                                              shift=min_dist)]
        if friend_peaks is not None:
            prohibit_peaks.extend(friend_peaks)
        if black_list_regions is not None:
            prohibit_peaks.append(black_list_regions)
        prohibit = join_bed(prohibit_peaks)
        real = join_bed(real_peaks)
        foreign = real.subtract(prohibit, 
                                full=True)
        foreign = cut_to_window(foreign,
                                 window_size=window_size,
                                 genome=genome) 
        foreign_seqs = genome.cut(foreign)
        foreign_seqs = drop_duplicates(foreign_seqs)
        
        sampler = SetGCSampler.make(negatives=foreign_seqs,
                                    sample_per_object=sample_per_object,
                                    seed=seed)
        pos_peaks = max(tf_peaks, key=lambda x: len(x))
        pos_peaks = cut_to_window(pos_peaks, 
                                       window_size=window_size,
                                       genome=genome)
        positives = genome.cut(pos_peaks)
        
        
        return cls(window_size=window_size,
                   sampler=sampler, 
                   positives=positives, 
                   min_dist=min_dist,
                   sample_per_object=sample_per_object, 
                   seed=seed)
    
    def sample_bed(self) -> BedData:
        negs = self.sampler.sample(self.positives) 
        
        bed = BedData([BedEntry(chr=n.metainfo["chr"], # type: ignore
                        start=n.metainfo["start"], # type: ignore
                        end=n.metainfo["end"]) for n in negs]) # type: ignore
        bed.sort()
        return bed
    
from ..sampling.shades import ShadesSampler

@dataclass
class PeakShadesSampler:
    sampler: ShadesSampler
    genome: Genome 
    window_size: int
    min_dist: int
    max_dist: int
    sample_per_object: int
    seed: int
    
    @classmethod
    def make(cls, 
             window_size: int, 
             genome: Genome, 
             tf_peaks: list[BedData],
             friend_peaks: list[BedData] | None = None,
             black_list_regions: BedData | None = None,
             min_dist=300,
             max_dist=600,
             sample_per_object: int = 1,
             seed = 777):
        assert window_size % 2 == 1, "Window size must be odd"
        assert min_dist >= 0, "Min dist must >= 0"
        if min_dist == 0:
            prohibit_peaks = copy(tf_peaks)
        else:
            tf_peak = join_bed(tf_peaks)
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                genomesizes = tempdir / "genomesizes.txt"
                genome.write_bed_genome_file(genomesizes)
                prohibit_peaks = [tf_peak.slop(genomesizes=genomesizes,
                                              shift=min_dist)]

        if friend_peaks is not None:
            prohibit_peaks.extend(friend_peaks)
        if black_list_regions is not None:
            prohibit_peaks.append(black_list_regions)
        prohibit = join_bed(prohibit_peaks)
        
        pos_peaks = max(tf_peaks, key=lambda x: len(x))
       
      
        sampler = ShadesSampler.make(pos_ds=pos_peaks,
                                     black_regions=prohibit,
                                     min_dist=min_dist, 
                                     max_dist=max_dist,
                                     peak_size=window_size,
                                     genome=genome,
                                     sample_per_peak=sample_per_object,
                                     seed=seed)
        
        return cls(sampler=sampler,
                   genome=genome,
                   window_size=window_size,
                   min_dist=min_dist,
                   max_dist=max_dist,
                   sample_per_object=sample_per_object,
                   seed=seed)

    def sample(self) -> list[SeqEntry]:
        smpls = self.sampler.sample()
        return self.genome.cut(smpls)
    
    def sample_bed(self) -> BedData:
        smpls = self.sampler.sample()
        return smpls

@dataclass
class PeakGenomeSampler:
    window_size: int
    genome: Genome 
    sampler: GenomeGCSampler
    positives: BedData
    min_dist: int
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
             min_dist: int = 300,
             sample_per_object: int = 1,
             exact: bool = True,
             max_overlap: int | None = None,
             n_procs: int = 1,
             precalc_profile: bool = False,
             seed = 777):
        assert window_size % 2 == 1, "Window size must be odd"
        assert min_dist >= 0, "Min dist must >= 0"
        if min_dist == 0:
            prohibit_peaks = copy(tf_peaks)
        else:
            tf_peak = join_bed(tf_peaks)
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                genomesizes = tempdir / "genomesizes.txt"
                genome.write_bed_genome_file(genomesizes)
                prohibit_peaks = [tf_peak.slop(genomesizes=genomesizes,
                                              shift=min_dist)]
        if friend_peaks is not None:
            prohibit_peaks.extend(friend_peaks)
        if black_list_regions is not None:
            prohibit_peaks.append(black_list_regions)

        prohibit = join_bed(prohibit_peaks)
        
        sampler = GenomeGCSampler.from_bed(genome=genome,
                                           blacklist_regions=prohibit,
                                           window_size=window_size,
                                           exclude_positives=False, # already excluded 
                                           max_overlap=max_overlap,
                                           sample_per_object=sample_per_object,
                                           exact=exact,
                                           precalc_profile=precalc_profile,
                                           seed=seed,
                                           n_procs=n_procs)
        
        pos_peaks = max(tf_peaks, key=lambda x: len(x))
        pos_peaks = cut_to_window(pos_peaks, 
                                   window_size=window_size,
                                   genome=genome)
        
        
        return cls(window_size=window_size,
                   genome=genome,
                   sampler=sampler,
                   positives=pos_peaks,
                   min_dist= min_dist, 
                   sample_per_object=sample_per_object,
                   exact=exact,
                   seed=seed)
        
    def sample_bed(self) -> BedData:
        return self.sampler.sample(self.positives)
        