import numpy as np 
from ..bedtools.beddata import BedData
from ..bedtools.bedentry import BedEntry
from ..seq.genome import Genome 
from numpy.random import Generator

def sample_segments(start: int, 
                    end: int, 
                    min_dist: int, 
                    max_dist: int | None = None,
                    rng: Generator | None = None) -> np.ndarray:
    '''
    start - inclusive
    end - exclusive 
    min_dist - inclusive
    max_dist - exclusive, min_dist * 2 by default
    '''
    
    if rng is None:
        rng = np.random.default_rng()
    
    N = end - start
    if N <= 0:
        return np.array([])
    if N <= min_dist:
        return rng.integers(start, end, size=1)
    
    if max_dist is None:
        max_dist = min_dist * 2
    w = max_dist - min_dist
    if N >= max_dist:
        p = 2 / (max_dist + min_dist - 1)
        probs = np.concatenate([np.full(shape=min_dist, fill_value=p),
                               (1-np.arange(1, w)/w) * p])
        start = start + rng.choice(max_dist -1, p=probs)
        K = N // min_dist
        dists = rng.integers(low=min_dist, high=max_dist, size=K)
        idx = [start]
        for i in range(K):
            start += dists[i]
            if start >= end:
                break
            idx.append(start)
    else: #  min_dist < N < max_dist 
        probs = np.concatenate([np.full(shape=min_dist, fill_value=1),
                                (1 - np.arange(1, N - min_dist + 1) / w )])
        probs = probs / probs.sum()
        start = start + rng.choice(N, p=probs)
        K = N // min_dist
        dists = rng.integers(low=min_dist, high=max_dist, size=K)
        idx = [start]
        for i in range(K):
            start += dists[i]
            if start >= end:
                break
            idx.append(start)
    return np.array(idx)

def sample_from_bed(beddata: BedData, window: int, min_dist: int, genome: Genome, rng: Generator):
    """
    Draw samples windows from beddata, with min distance between point centers equal to min_dist
    """
    entries = []
    radius = window // 2
    for entry in beddata:
        s = entry.start + radius
        e = entry.end - radius - 1 
        if s < e:
            centers = sample_segments(start=s, 
                                      end=e,
                                      min_dist=min_dist,
                                      max_dist=min_dist * 2,
                                      rng=rng)
            new_entries = [BedEntry.from_center(entry.chr, 
                                           cntr=c, 
                                           radius=radius,
                                           genome=genome) for c in centers]
            entries.extend(new_entries)
    return BedData(entries)
                  
    