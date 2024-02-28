import gzip 
from Bio import SeqIO
from typing import overload
from pathlib import Path
import tqdm

END_LINE_CHARS = "\r\n"

@overload
def replace_path2str(obj: list) -> list:
    pass

@overload
def replace_path2str(obj: dict) -> dict:
    pass

def replace_path2str(obj: list | dict) -> list | dict:
    if isinstance(obj, dict):
        new_dt = {}
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                new_dt[key] = replace_path2str(value)
            else:
                if isinstance(value, Path):
                    value = str(value)
                new_dt[key] = value
        return new_dt
    else:
        new_lst = []
        for el in obj:
            if isinstance(el, (dict, list)):
                new_lst.append(replace_path2str(el))
            else:
                if isinstance(el, Path):
                    el = str(el)
                new_lst.append(el)
        return new_lst
    
def merge_fastqgz(in_paths, out_path):
    recs = []
    for path in in_paths:
        with gzip.open(path, "rt") as inp:
            for rec in tqdm.tqdm(SeqIO.parse(inp, 'fastq')):
                #print(rec.id, rec.description, rec.seq)
                recs.append(rec)
    with  gzip.open(out_path, "wt") as out:
        SeqIO.write(recs, out, 'fastq')
        

def merge_fastqgz_unique(in_paths, out_path):
    recs = []
    seq_set = set()
    for path in tqdm.tqdm(in_paths):
        with gzip.open(path, "rt") as inp:
            for rec in tqdm.tqdm(SeqIO.parse(inp, 'fastq')):
                if rec.seq not in seq_set:
                    seq_set.add(rec.seq)                    
                    recs.append(rec)
    with gzip.open(out_path, "wt") as out:
        SeqIO.write(recs, out, 'fastq')