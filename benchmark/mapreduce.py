from typing import IO, Iterable, Optional
from collections import Counter
from pathlib import Path
import pyfastx
import glob
import parse
import os 
from utils import END_LINE_CHARS
import shutil
import heapq
from itertools import groupby

FIELD_SEP = "\t"

def count(in_path: Path, out_path: Path):
    cnt = Counter()
    f = pyfastx.Fastq(str(in_path), full_name=True, build_index=False)
    cnt = Counter(seq for _, seq, _ in f)
    keys = sorted(cnt)
    with out_path.open("w") as out:
        for key in keys:
            occ = cnt[key]
            print(key, occ, file=out, sep=FIELD_SEP)

def parse_entry(line):
    seq, cnt = line.split(FIELD_SEP)
    cnt = int(cnt)
    return seq, cnt

MAP_DIR = Path(".MAP")
#shutil.rmtree(MAP_DIR)
MAP_DIR.mkdir(exist_ok=True)
READS_DIR = Path('/home_local/vorontsovie/greco-bit-data-processing/source_data/HTS/reads')

def reduce_cnt(inputs: Iterable[Path], out_p: Path):
    its = []
    for r in inputs:
        r = Path(r)
        its.append(parse_entry(line.rstrip(END_LINE_CHARS)) for line in r.open())
    
    with out_p.open("w") as out:
        for name, g in tqdm.tqdm(groupby(heapq.merge(*its), key=lambda x : x[0])):
                print(name, sum( e[1] for e in g ), file=out)
import tqdm

from time import perf_counter
if __name__ == "__main__":
    print("first solution")
    reads = glob.glob(str(READS_DIR / "*.gz"))[:250]
    cnt_paths = []
    for r in tqdm.tqdm(reads):
        r = Path(r)
        nm = r.name.replace('.fastq.gz', '')
        out_p = MAP_DIR / f"{nm}_count.txt"
        if not out_p.exists():
            count(r, out_p)
        cnt_paths.append(out_p)

    outp = Path('ex.txt')
    reduce_cnt(cnt_paths, outp)
    print("second solution")
    
    cnt = Counter()
    for fl in tqdm.tqdm(reads):
        f = pyfastx.Fastq(fl, full_name=True, build_index=False)
        cnt.update(seq for _, seq, _ in f)

    cnt2 = {}
    for line in open('ex.txt'):
        seq, c = line.split()
        cnt2[seq] = int(c) 

    for key, value in cnt.items():
        if value != cnt2[key]:
            print(key, value)
            break
        