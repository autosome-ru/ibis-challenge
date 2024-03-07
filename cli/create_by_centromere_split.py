import sys 

sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd 
from bibis.seq.genome import Genome

@dataclass(order=True)
class Region:
    chr: str
    start: int
    end: int


def write_regions(regions: list[Region], out_path: str | Path):
    with open(out_path, "w") as out:
        for reg in regions:
            print(reg.chr, reg.start, reg.end, sep="\t", file=out)

parser = argparse.ArgumentParser()
parser.add_argument("--centromers_file",
                    required=True, 
                    type=str)
parser.add_argument("--genome",
                    required=True, 
                    type=str)
parser.add_argument("--out_path",
                    required=True,
                    type=str)

args = parser.parse_args()

out_dir = Path(args.out_path)
out_dir.mkdir(exist_ok=True, parents=True)

t = pd.read_csv(args.centromers_file,
                sep="\t", 
                header=None, 
                names=['chr', 'start', 'end', 'id'],
                index_col=False)


centromers = []
for _, row in t.iterrows():
    cent = Region(row.chr, row.start, row.end)
    centromers.append(cent)
centromers.sort()

from itertools import groupby

centromers_joined = []
for ch, cents in groupby(centromers, key = lambda x: x.chr):
    cents = list(cents)
    start = min(c.start for c in cents)
    end = max(c.end for c in cents)
    cent = Region(chr=ch, start=start, end=end)
    centromers_joined.append(cent)

joined_centromers_path = out_dir / "joined_centromers.bed"
write_regions(centromers_joined, joined_centromers_path)

genome = Genome.from_dir(args.genome)
ghts_hide = []     
chs_hide = []
for cent in centromers_joined:
    try:
        ind = int(cent.chr.replace("chr", ""))
    except ValueError: #skip random chromosomes and autosomes
        continue
    chr_seq = genome.chroms[cent.chr]
    before = Region(chr=cent.chr, 
                      start=0, 
                      end=cent.start)
    after = Region(chr=cent.chr,
                      start=cent.end,
                      end=len(chr_seq))
    if ind % 2 == 0:
        chs_hide.append(before)
        ghts_hide.append(after)
    else:
        chs_hide.append(after)
        ghts_hide.append(before)

chs_hide_path = out_dir / "chs_hide.bed"
ghts_hide_path = out_dir / "ghts_hide.bed"

write_regions(chs_hide, chs_hide_path)
write_regions(ghts_hide, ghts_hide_path)
