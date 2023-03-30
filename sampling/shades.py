import tempfile
import numpy as np
from numpy.random import Generator

from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import tempfile

from ..bedtools.beddata import BedData
from ..bedtools.bedentry import BedEntry
from ..seq.genome import Genome

@dataclass
class ShadesSampler:
    genome: Genome 
    positives: dict[tuple[str, int], BedEntry]
    sample_regions: dict[tuple[str, int], BedData]
    peak_size: int
    rng: Generator
    sample_per_peak: int = 1
     
    
    @classmethod
    def _key_from_entry(cls, entry: BedEntry) -> tuple[str, int]:
        if entry.peak is None:
            raise Exception("All positive entries must have peak")
        return (entry.chr, entry.peak)
    
    @classmethod
    def make(cls, 
             genome: Genome, 
             pos_ds: BedData, 
             black_regions: BedData, 
             max_dist: int, 
             peak_size: int,
             sample_per_peak: int = 1,
             seed: int = 777):
        positives = {}
        for e in pos_ds:
            key = cls._key_from_entry(e)
            if key in positives:
                raise Exception("Peak field must be unique, {key}", key)
            positives[key] = e
            
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            genomesizes = tempdir / "genomesizes.txt"
            genome.write_bed_genome_file(genomesizes)
            flanks = pos_ds.flank(genomesizes, max_dist + peak_size)
        sbstr = flanks.subtract(black_regions, full=False)
        sub_dt = cls.map_peaks(sbstr)
        
        sample_regions = {}
        for peak_ch, sgmnts in sub_dt.items():
            sgmnts = sgmnts.apply(lambda s: s.truncate(peak_size // 2)).filter(lambda s: len(s) > 0)
            sample_regions[peak_ch] = sgmnts
        
        rng = np.random.default_rng(seed=seed)
        return cls(positives=positives,
                   genome=genome,
                   sample_regions=sample_regions,
                   peak_size=peak_size,
                   rng=rng,
                   sample_per_peak=sample_per_peak)
    
    @classmethod
    def map_peaks(cls, bed: BedData) -> dict[int, BedData]:
        dt = defaultdict(BedData)
        for e in bed:
            key = cls._key_from_entry(e)
            dt[key].append(BedEntry(e.chr, e.start, e.end))
        return dict(dt)
    
    def sample(self, save_metainfo: bool = True) -> BedData:
        entries = []
        for peak_ch, sgmnts in self.sample_regions.items():
            smpl = sgmnts.sample_shades(seqsize=self.peak_size, 
                                        k=self.sample_per_peak,
                                        genome=self.genome)
            if len(smpl) < self.sample_per_peak:
                print(f"Warning: unable to sample more than {len(smpl)} for peak {peak_ch}")
            if save_metainfo:
                positive = self.positives[peak_ch]
                for e in smpl:
                    e.metainfo = positive.metainfo
            entries.extend(smpl)

        return BedData(entries)