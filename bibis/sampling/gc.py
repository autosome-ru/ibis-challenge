from __future__ import annotations
import numpy as np 

from numpy.random import Generator

import tempfile
from dataclasses import dataclass
import heapq
from collections import defaultdict
from pathlib import Path
from functools import partial
import multiprocessing
import tqdm 

import sys
from copy import copy

from .sample_bed import sample_from_bed

from .disjoint import DisjointSet
from .gcprofile import GCProfile
from ..seq.seqentry import SeqEntry
from ..bedtools.beddata import BedData, join_bed
from ..bedtools.bedentry import BedEntry
from ..bedtools.constants import CHROM_ORDER
from ..seq.genome import Genome

def filter_unique(seqs: list[SeqEntry]) -> list[SeqEntry]:
    unique = dict()
    for s in seqs:
        if s.sequence not in unique:
            unique[s.sequence] = s
    return list(unique.values()) 

@dataclass
class SetGCSampler:
    negatives: list[SeqEntry]
    negatives_gc: np.ndarray
    rng: np.random.Generator
    sample_per_object: int = 1
    
    @classmethod
    def make(cls,
             negatives: list[SeqEntry],
             sample_per_object: int = 1,
             seed: int =777) -> 'SetGCSampler':
        negatives = filter_unique(negatives)
        
        neg = [-np.inf]
        neg.extend(s.gc for s in negatives)
        neg.append(np.inf)
        negatives_gc = np.array(neg)
        negative_sort_idx = np.argsort(negatives_gc)
        negatives_gc = negatives_gc[negative_sort_idx]
        negatives = [negatives[negative_sort_idx[i] - 1] for i in range(1, len(negatives)+1)] # +-1 due to added inf
        rng = np.random.default_rng(seed=seed)
        return cls(negatives=negatives,
                   negatives_gc=negatives_gc,
                   sample_per_object=sample_per_object, 
                   rng=rng)
        
    
    def sample(self, 
               positive: list[SeqEntry],
               save_metainfo: bool = True,
               return_loss: bool = False) -> list[SeqEntry] | tuple[list[SeqEntry], float]:
        requested_size = len(positive) * self.sample_per_object
        #if requested_size > len(self.negatives):
        #    raise Exception(f"Can't sample: number of negatives: {len(self.negatives)}, requested: { requested_size}")
        
        if requested_size > len(self.negatives):  
            print(f"Can't sample more than total negatives num: number of negatives: {len(self.negatives)}, requested: {requested_size}", file=sys.stderr)
            loss = 0 
            neg_sampled = [copy(n) for n in self.negatives]
            if save_metainfo:
                for n in neg_sampled:
                    if n.metainfo is None:
                        n.metainfo = {}
                    else:
                        n.metainfo = copy(n.metainfo)
                        
                    n.metainfo["pos_cor"] = "__NO_CORRESPONDENCE"
        else:
            positive_gc = np.array([s.gc for s in positive])
            negative_gc = self.negatives_gc
            N, M = negative_gc.shape[0], positive_gc.shape[0]

            disjoint = DisjointSet.from_negative_gc(negative_gc) 

            heap = []

            pos_clusters = np.searchsorted(negative_gc, positive_gc) - 1

            for i in range(M):
                val = positive_gc[i]
                cl = pos_clusters[i]
                p = disjoint.root(cl)
                l = disjoint.left(p)
                r = disjoint.right(p)

                d1 = abs(negative_gc[l] - val)
                d2 = abs(negative_gc[r] - val)

                if d1 < d2:
                    pair = (d1, i, l)
                elif d1 > d2:
                    pair = (d2, i, r)
                else:
                    if self.rng.random() < 0.5:
                        pair = (d1, i, l)
                    else:
                        pair = (d2, i, r)

                heap.append(pair)

            heapq.heapify(heap)
            samples = defaultdict(list)

            loss = 0
            while len(heap) != 0:
                d, i, pos = heapq.heappop(heap)

                if not disjoint.is_taken[pos]:
                    loss += d
                    p = disjoint.take(pos)
                    samples[i].append(pos)

                    if len(samples[i]) == self.sample_per_object:
                        continue
                cl = pos_clusters[i]
                p = disjoint.root(cl)
                val = positive_gc[i]
                l = disjoint.left(p)
                r = disjoint.right(p)
                d1 = abs(negative_gc[l] - val)
                d2 = abs(negative_gc[r] - val)

                if d1 < d2:
                    pair = (d1, i, l)
                elif d1 > d2:
                    pair = (d2, i, r)
                else:
                    if self.rng.random() < 0.5:
                        pair = (d1, i, l)
                    else:
                        pair = (d2, i, r)
                heapq.heappush(heap, pair)
                
            neg_sampled = []
            for j, inds in samples.items():
                for i in inds:
                    im = i - 1 # due to added -np.inf  !!!!
                    neg = self.negatives[im]
                    if save_metainfo:
                        if positive[j].metainfo is not None: 
                            if neg.metainfo is None:
                                neg.metainfo = {}
                            
                            neg.metainfo['pos_cor'] = copy(positive[j].metainfo)
                    neg_sampled.append(neg)
                
        
        if return_loss:
            return neg_sampled, loss
        else:
            return neg_sampled


@dataclass
class GenomeGCSampler:
    genome: Genome
    negative_regions: BedData
    genome_profile: dict[str, GCProfile] | None 
    window_size: int 
    min_point_dist: int 
    exclude_positives: bool 
    rng: np.random.Generator
    sample_per_object: int = 1
    n_procs: int = 1
    exact: bool = True
    
    
    @staticmethod
    def _exact_profile(genome: Genome,
                 regions: BedData,
                 chr: str,
                 window_size: int,
                 rng: Generator) -> GCProfile:
        return GCProfile.from_bed(
            genome=genome,  
            regions=regions, 
            chr=chr, 
            window_size=window_size,
            rng=rng)
        
    @staticmethod
    def _not_exact_profile(genome: Genome,
                 regions: BedData,
                 chr: str,
                 window_size: int,
                 min_point_dist: int,
                 rng: Generator) -> GCProfile:
    
        regions = sample_from_bed(regions, 
                                  window=window_size, 
                                  min_dist=min_point_dist,
                                  genome=genome,
                                  rng=rng)
        return GCProfile.from_bed(
            genome=genome,  
            regions=regions, 
            chr=chr, 
            window_size=window_size,
            rng=rng)
        
    def get_chrom_profile(self, ch: str) -> GCProfile:
        if self.genome_profile is not None:
            return self.genome_profile[ch]
        return self._calc_profile(chr=ch,
                                  genome=self.genome, 
                                  regions=self.negative_regions,
                                  window_size=self.window_size,
                                  rng=self.rng,
                                  min_point_dist=self.min_point_dist,
                                  exact=self.exact)
        
       
    
    @classmethod
    def _calc_profile(cls,
                      chr: str,
                      genome: Genome,
                      regions: BedData,
                      window_size: int,
                      rng: Generator,
                      min_point_dist: int,
                      exact: bool) -> GCProfile:
        if exact:
            return cls._exact_profile(genome=genome,
                                      regions=regions,
                                      chr=chr,
                                      window_size=window_size,
                                      rng=rng)
        else:
            return cls._not_exact_profile(genome=genome,
                                          regions=regions,
                                          chr=chr,
                                          window_size=window_size,
                                          min_point_dist=min_point_dist,
                                          rng=rng)
        
    @classmethod
    def default_prohibited_regions(cls, genome: Genome, window_size:int) -> BedData:
        entries = []
        for ch in CHROM_ORDER:
            seq = genome.chroms.get(ch, None)
            if seq is None:
                continue
            size = len(seq)
            entries.append(BedEntry(chr=ch,
                                    start=0,
                                    end=window_size // 2))
            entries.append(BedEntry(chr=ch,
                                    start=size-window_size-1,
                                    end=size))
        return BedData(entries)
    
    @classmethod
    def from_bed(cls,
                  genome: Genome, 
                  blacklist_regions: BedData,
                  window_size: int,
                  exclude_positives: bool,
                  max_overlap: int | None = None,
                  sample_per_object: int = 1,
                  exact: bool = True,
                  precalc_profile: bool = False,
                  seed: int = 777,
                  n_procs: int = 1) -> 'GenomeGCSampler':
        if max_overlap is None:
            max_overlap = window_size // 2
        min_point_dist = window_size - max_overlap
        
        
        
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            chroms_path = tempdir / "chromsizes.txt"
            genome.write_bed_genome_file(chroms_path)
            blacklist_regions = blacklist_regions.slop(genomesizes=chroms_path, 
                                                       shift=window_size // 2)
            blacklist_regions = join_bed([blacklist_regions, 
                                          cls.default_prohibited_regions(genome=genome,
                                                                         window_size=window_size)])
            negative_regions = blacklist_regions.complement(chroms_path)

        rng = np.random.default_rng(seed=seed)
        
        if precalc_profile:
            if n_procs == 1:
                genome_profile = cls._calc_genome_profile_noparallel(genome=genome,
                                                                    negative_regions=negative_regions,
                                                                    window_size=window_size,
                                                                    rng=rng,
                                                                    min_point_dist=min_point_dist,
                                                                    exact=exact)
            else:
                genome_profile = cls._calc_genome_profile_parallel(n_procs=n_procs,
                                                                genome=genome,
                                                                negative_regions=negative_regions,
                                                                window_size=window_size,
                                                                rng=rng,
                                                                min_point_dist=min_point_dist,
                                                                exact=exact)
        else:
            genome_profile = None
                     
        return cls(genome=genome,
                   negative_regions=negative_regions,
                   genome_profile=genome_profile, 
                   window_size=window_size,
                   min_point_dist=min_point_dist,
                   exclude_positives=exclude_positives,
                   sample_per_object=sample_per_object,
                   rng=rng,
                   n_procs=n_procs, 
                   exact=exact)

    @classmethod
    def _calc_genome_profile_noparallel(cls, 
                                 genome: Genome, 
                                 negative_regions: BedData,
                                 window_size: int, 
                                 rng: Generator,
                                 min_point_dist: int,
                                 exact: bool):
        genome_profile = {}
        for ch in genome.chroms.keys():
            profile = cls._calc_profile(genome=genome,
                                        regions=negative_regions,
                                        chr=ch,
                                        window_size=window_size,
                                        rng=rng,
                                        exact=exact,
                                        min_point_dist=min_point_dist)
            genome_profile[ch] = profile
        
        return genome_profile
    
    @classmethod
    def _calc_genome_profile_parallel(cls,
                                     n_procs: int,
                                     genome: Genome,
                                     negative_regions: BedData,
                                     window_size: int,
                                     rng: Generator,
                                     min_point_dist: int,
                                     exact: bool):
        calculator = partial(cls._calc_profile, 
                             genome=genome,
                             regions=negative_regions,
                             window_size=window_size,
                             rng=rng,
                             exact=exact,
                             min_point_dist=min_point_dist)
        with multiprocessing.Pool(processes=n_procs) as pool:
            profiles = pool.map(calculator, genome.chroms.keys())
        genome_profile = {ch: pr for ch, pr in zip(genome.chroms.keys(), 
                                                   profiles, 
                                                   strict=True)}
        return genome_profile
  
        
    def _sample_chromosome(self, positives: BedData, chr: str) -> BedData:
        positives = positives.filter(lambda e: e.chr == chr)
        if len(positives) == 0:
            return BedData()
        pos_seqs = self.genome.cut(positives)
        positive_gc = np.array([s.gc for s in pos_seqs])
        
        chr_profile = self.get_chrom_profile(ch=chr)
       
        negative_gc, neg_positions = chr_profile.gc, chr_profile.idx
        
        if positive_gc.shape[0] * self.sample_per_object > negative_gc.shape[0]:
            raise Exception(f"Can't sample: requested sample size is smaller, then negatives number: {positive_gc.shape[0] * self.sample_per_object}, {negative_gc.shape[0]}")
        
        negative_gc = np.concatenate([[-np.inf],  negative_gc, [np.inf]])
        
        M = positive_gc.shape[0]

        disjoint = DisjointSet.from_negative_gc(negative_gc) 
        
        heap = []

        pos_clusters = np.searchsorted(negative_gc, positive_gc) - 1
        
        real_position_mapping = np.full(max(neg_positions) + 2, fill_value=-2, dtype=np.int32)

        for i in range(neg_positions.shape[0]):
            real_position_mapping[neg_positions[i]] = i 
            
        if self.exclude_positives:
            for e in positives:
                sur_start = max(e.start - self.min_point_dist, 0)
                sur_end = max(e.end + self.min_point_dist + 1, real_position_mapping.shape[0])
                for p in range(sur_start, sur_end):
                    mapped_pos = real_position_mapping[p] + 1  # due to -np.inf
                    if mapped_pos > 0: # if position exist 
                        _ = disjoint.take(mapped_pos)
        
        for i in range(M):
            val = positive_gc[i]
            cl = pos_clusters[i]
            p = disjoint.root(cl)
            l = disjoint.left(p)
            r = disjoint.right(p)

            d1 = abs(negative_gc[l] - val)
            d2 = abs(negative_gc[r] - val)

            if d1 < d2:
                pair = (d1, i, l)
            elif d1 > d2:
                pair = (d2, i, r)
            else:
                if self.rng.random() > 0.5:
                    pair = (d1, i, l)
                else:
                    pair = (d2, i, r)

            heap.append(pair)

        heapq.heapify(heap)
        samples = defaultdict(list)

       
        loss = 0
        while len(heap) != 0:
            d, i, pos = heapq.heappop(heap)

            if not disjoint.is_taken[pos]:
                
                loss += d
                _ = disjoint.take(pos)
                
                samples[i].append(pos)
                
                # added
                
                if self.exact: # else no such positions exist and we can skip this step 
                    real_pos = neg_positions[pos-1] # due to -np.inf
                    
                    sur_start = max(real_pos -self.min_point_dist, 0)
                    for sur_pos in range(sur_start, real_pos):
                        mapped_sur_pos = real_position_mapping[sur_pos] + 1 # due to -np.inf
                        if mapped_sur_pos > 0:
                            _ = disjoint.take(mapped_sur_pos)
                        
                    sur_end = min(real_pos + self.min_point_dist + 1, real_position_mapping.shape[0])
                    for sur_pos in range(real_pos + 1,  sur_end):
                        mapped_sur_pos = real_position_mapping[sur_pos] + 1 # due to -np.inf
                        if mapped_sur_pos > 0: # if position exist 
                            _ = disjoint.take(mapped_sur_pos)
                
                
                if len(samples[i]) == self.sample_per_object:
                    continue
                         
            cl = pos_clusters[i]
            p = disjoint.root(cl)
            val = positive_gc[i]
            l = disjoint.left(p)
            r = disjoint.right(p)
            d1 = abs(negative_gc[l] - val)
            d2 = abs(negative_gc[r] - val)

            if d1 < d2:
                pair = (d1, i, l)
            elif d1 > d2:
                pair = (d2, i, r)
            else: # d1 == d2
                if self.rng.random() > 0.5:
                    pair = (d1, i, l)
                else:
                    pair = (d2, i, r)
            heapq.heappush(heap, pair)


        for key, value in samples.items():
            restored = [neg_positions[v - 1] for v in value]
            samples[key] = restored 
        
        part_window = self.window_size // 2
        entries = []
        for pos_ind, negs in samples.items():
            pos_entry = positives[pos_ind]
            
            for n in negs:

                neg_entry = BedEntry.from_center(pos_entry.chr, 
                                                 cntr=n, 
                                                 radius=part_window,
                                                 metainfo=copy(pos_entry.metainfo), 
                                                 genome=self.genome)
                if len(neg_entry) != self.window_size:
                    raise Exception("Error while sampling occured")
                entries.append(neg_entry)
            
        return BedData(entries)
    
    def _to_seqs(self, bed: BedData):
        seqs = self.genome.cut(bed)
        for i, s in enumerate(seqs):
            s.metainfo = bed[i].metainfo
            if s.metainfo is None:
                s.metainfo = {}
                s.metainfo["kind"] = "negative"
                s.metainfo['chr'] = bed[i].chr
                s.metainfo['start'] = bed[i].start # type: ignore
                s.metainfo['end'] = bed[i].end # type: ignore
        return seqs
    
    
    def _sample_noparallel(self, positives: BedData) -> BedData:
        beds = []
        for chr in tqdm.tqdm(self.genome.chroms.keys()):
            ch_bed = self._sample_chromosome(positives=positives,
                                                   chr=chr)
            beds.append(ch_bed)
        bed = join_bed(beds, sort=True)
        return bed
    
    def _sample_parallel(self, positives: BedData) -> BedData:
        seqs = []
        calculator = partial(self._sample_chromosome,
                             positives=positives)

        with multiprocessing.Pool(processes=self.n_procs) as pool:
            beds = pool.map(calculator, self.genome.chroms.keys())
        bed = join_bed(beds, sort=True)        
        return bed
            
    def sample(self, positives: BedData) -> BedData:
        if self.n_procs == 1:
            print("No parallel")
            return self._sample_noparallel(positives=positives)
        return self._sample_parallel(positives=positives)