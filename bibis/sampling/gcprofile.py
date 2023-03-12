from __future__ import annotations
from typing import Tuple 
from dataclasses import dataclass
import numpy as np 

from numba import jit

from ..bedtools.bedentry import BedEntry
from ..bedtools.beddata import BedData
from ..seq.genome import Genome 

from numpy.random import Generator

@jit(nopython=True)
def calc_gc_profile(seq: str, window_size: int):
    assert window_size % 2 == 1
    assert len(seq) >= window_size
    seq = seq.upper()
    profile = np.zeros(len(seq) - window_size + 1, dtype=np.float32)
    gc = 0
    for i in range(0, window_size):
        if seq[i] == "G" or seq[i] == "C":
            gc += 1
    profile[0] = gc / window_size
    for i in range(window_size, len(seq)):
        if seq[i] == "G" or seq[i] == "C":
            gc += 1
        if seq[i-window_size] == "G" or seq[i-window_size] == "C":
            gc -= 1
        profile[i-window_size+1] = gc / window_size
    
    idx = np.arange(window_size // 2, len(seq) - window_size // 2)
        
    return idx, profile

def calc_gc_profile_for_bedentry(bedentry: BedEntry, genome: Genome, window_size: int):
    seq = str(genome[bedentry])
    idx, gc = calc_gc_profile(seq=seq, window_size=window_size)
    idx += bedentry.start
    return idx, gc 

def calc_gc_profile_for_beddata(beddata: BedData, chr: str, genome: Genome, window_size: int):
    beddata = beddata.filter(lambda e: e.chr == chr)
    idx_ar = []
    gc_ar = []
    if len(beddata) != 0:
        for e in beddata:
            if len(e) < window_size:
                continue
            idx, gc = calc_gc_profile_for_bedentry(bedentry=e, 
                                                genome=genome, 
                                                window_size=window_size)
            idx_ar.append(idx)
            gc_ar.append(gc)
        idx = np.concatenate(idx_ar)
        gc = np.concatenate(gc_ar)
    else:
        idx = np.array(idx_ar)
        gc = np.array(gc_ar)
    return idx, gc

@dataclass
class GCProfile:
    idx: np.ndarray
    gc: np.ndarray
    
    @staticmethod 
    def _shuffle_and_sort(idx: np.ndarray, gc: np.ndarray, rng: Generator) -> Tuple[np.ndarray, np.ndarray]:
        shuffle_idx = rng.permutation(gc.shape[0]) 
        gc = gc[shuffle_idx] 
        idx =  idx[shuffle_idx] 
        
        order = np.argsort(gc) 
        idx =  idx[order] 
        gc = gc[order] 
        return idx, gc
    
    @classmethod
    def from_bed(cls,
                 genome: Genome,
                 regions: BedData,
                 chr: str,
                 window_size: int,
                 rng: Generator):
        idx, gc = calc_gc_profile_for_beddata(beddata=regions,
                                              chr=chr, 
                                              genome=genome,
                                              window_size=window_size)
        idx, gc = cls._shuffle_and_sort(idx=idx, 
                                        gc=gc,
                                        rng=rng)
            
        return cls(idx=idx, gc=gc)