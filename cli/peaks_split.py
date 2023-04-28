from dataclasses import dataclass, asdict
import sys 
import json
from pathlib import Path

import argparse
import sys


def filter_chrom(peak_file: str | Path,
                 out_file: str | Path,
                 chroms: set[str] | list[str]):
    with open(peak_file, 'r') as inp, open(out_file, "w") as out:
        for line in inp:
            if line.startswith("#"):
                print(line, end="", file=out)
            else:
                ch, _ = line.split("\t", maxsplit=1)
                if ch in chroms:
                    print(line, end="", file=out)
                

parser = argparse.ArgumentParser()

parser.add_argument("--benchmark_out_dir", 
                    required=True, 
                    type=str)
parser.add_argument("--tagdb_cfg",
                    required=True,
                    type=str)
parser.add_argument("--config_file", 
                    required=True, 
                    type=str)
parser.add_argument("--type", 
                    choices=['Leaderboard', 'Final'], 
                    required=True, type=str)
parser.add_argument("--n_procs", 
                    default=1,
                    type=int)
parser.add_argument("--bedtools", 
                    default="/home_local/dpenzar/bedtools2/bin",
                    type=str)
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)

args = parser.parse_args()

sys.path.append(args.bibis_root) # temporary solution while package is in development

from bibis.seqdb.config import DBConfig 
from bibis.chipseq.config import ChipSeqConfig, ChipSeqDatasetConfig
from bibis.chipseq.peakfile import ChIPPeakList
from bibis.bedtools.beddata import BedData
from bibis.chipseq.samplers import (ChIPForeignSampler, 
                                    ChIPShadesSampler,
                                    ChIPGenomeSampler,
                                    cut_to_window)
from bibis.seq.genome import Genome
from bibis.seq.seqentry import write as seq_write
from bibis.bedtools.bedtoolsexecutor import BedtoolsExecutor

BedtoolsExecutor.set_defaull_executor(args.bedtools)

BENCH_SEQDB_CFG = Path(args.tagdb_cfg)

CHS_BENCH_DIR = Path(args.benchmark_out_dir)
CHS_BENCH_DIR.mkdir(parents=True, exist_ok=True)

cfg = ChipSeqConfig.load(args.config_file)

tf_peaks = [ChIPPeakList.read(t) for t in cfg.tf_path]
tf_beds = [f.to_beddata() for f in tf_peaks]


if "train" in cfg.splits:
    split = cfg.splits["train"]
    if "test" in cfg.splits:
        ind, _ = max(enumerate(tf_beds), key=lambda x: len(x[1]))
        train_peaks_paths = list(cfg.tf_path)
        train_peaks_paths.pop(ind)
    else:
        train_peaks_paths = list(cfg.tf_path)
    
    train_dir = CHS_BENCH_DIR / "train" / cfg.tf_name
    train_dir.mkdir(exist_ok=True, parents=True)
    
    #filter_chrom
    
    for ind, peak_path in enumerate(train_peaks_paths, 1):
        fl_name = Path(peak_path).name
        replic_path = train_dir / fl_name
        filter_chrom(peak_file=peak_path, 
                     out_file=replic_path,
                     chroms=split.chroms)

if "test" in cfg.splits:
    ind, valid_bed = max(enumerate(tf_beds), key=lambda x: len(x[1]))
    train_beds = list(tf_beds)
    train_beds.pop(ind)
    
    valid_dir = CHS_BENCH_DIR / "valid" / cfg.tf_name
    valid_dir.mkdir(exist_ok=True, parents=True)
    
    validation_replic_file = cfg.tf_path[ind]
    print(f"Selected {cfg.tf_path[ind]} as validation replic")
    
    sel_path = valid_dir / "selected.json"
    with open(sel_path, "w") as out:
        json.dump(obj={"validation_replic": validation_replic_file},
                  fp=out,
                  indent=4)
    
    
    split = cfg.splits["test"]
    foreign_peaks = [ChIPPeakList.read(t) for t in cfg.foreign_cfg.foreigns_path]
    foreign_beds = [f.to_beddata() for f in foreign_peaks]
    
    valid_bed = valid_bed.filter(lambda e: e.chr in split.chroms) # type: ignore
    train_beds = [bed.filter(lambda e: e.chr in split.chroms) for bed in train_beds]
    foreign_beds = [bed.filter(lambda e: e.chr in split.chroms) for bed in foreign_beds]

    if cfg.black_list_path is not None:
        black_list = BedData.from_file(cfg.black_list_path)
    else:
        black_list = None
    
    
    if black_list is not None:
        valid_black_list = black_list.filter(lambda e: e.chr in split.chroms)
    else:
        valid_black_list = None

    friends_peaks = list(train_beds)
    print("Downloading genome")
    gpath = Path(cfg.genome_path)
    if gpath.is_dir():
        genome = Genome.from_dir(gpath)
    else:
        genome = Genome.from_fasta(gpath)
    
    samples: dict[str, BedData] = {"positives": cut_to_window(bed=valid_bed, 
                                                              window_size=cfg.window_size,
                                                              genome=genome)}
    
    
    print("Creating shades")    
    shades_sampler = ChIPShadesSampler.make(window_size=cfg.window_size,
                                    genome=genome,
                                    tf_peaks=[valid_bed], # type: ignore
                                    friend_peaks=friends_peaks,
                                    black_list_regions=valid_black_list,
                                    sample_per_object=cfg.shades_cfg.balance,
                                    max_dist=cfg.shades_cfg.max_dist,
                                    seed=cfg.seed)
    samples['shades'] = shades_sampler.sample_bed()
    friends_peaks.append(samples['shades'])
    
    print("Creating aliens") 
    foreign_sampler = ChIPForeignSampler.make(window_size=cfg.window_size,
                                            genome=genome,
                                            tf_peaks=[valid_bed], # type: ignore
                                            real_peaks=foreign_beds,
                                            friend_peaks=friends_peaks,
                                            black_list_regions=valid_black_list,
                                            sample_per_object=cfg.foreign_cfg.balance,
                                            seed=cfg.seed)

    samples['aliens'] = foreign_sampler.sample_bed()
    friends_peaks.append(samples['aliens'])
    
    print("Creating random genome samples")
    genome_sampler = ChIPGenomeSampler.make(window_size=cfg.window_size,
                                    genome=genome,
                                    tf_peaks=[valid_bed], # type: ignore
                                    friend_peaks=friends_peaks,
                                    black_list_regions=valid_black_list,
                                    sample_per_object=cfg.genome_sample_cfg.balance,
                                    exact=cfg.genome_sample_cfg.exact,
                                    max_overlap=cfg.genome_sample_cfg.max_overlap,
                                    n_procs=args.n_procs,
                                    seed=cfg.seed)
    
    samples['random'] = genome_sampler.sample_bed()
    friends_peaks.append(samples['random'])
    
    
    db = DBConfig.load(BENCH_SEQDB_CFG).build()
    
    # benchmark part files
    parts_dir = valid_dir / 'parts'
    parts_dir.mkdir(exist_ok=True)
    for name, bed in samples.items():
        fas = genome.cut(bed)
        fas = db.taggify_entries(fas)
        
        table_path = parts_dir / f"{name}.tsv"
        with open(table_path, "w") as out:
            for entry, seq in zip(bed, fas):
                print(entry.chr, 
                      entry.start,
                      entry.end, 
                      seq.tag, 
                      sep="\t",
                      file=out)

        seq_path = parts_dir / f"{name}.fasta"        
        seq_write(fas, seq_path)
        
    ds_config_file = valid_dir / f"config.json"
    ds = ChipSeqDatasetConfig(tf_name=cfg.tf_name, 
                              tf_path=str(valid_dir))
    ds.save(ds_config_file)
    
    # answer files for benchmark
    answer_dir = valid_dir / 'answer'
    answer_dir.mkdir(exist_ok=True)
    for background in ('shades', 'aliens', 'random'):
        path_pref = str(answer_dir / f"{cfg.tf_name}_{background}")
        ds_dir = answer_dir / background
        ds_dir.mkdir(exist_ok=True)
        ds_info = ds.make_ds(background=background,
                             path_pref=str(ds_dir / "data"),
                             hide_labels=False)
        
        filepath = answer_dir / background / f"config.json"
        ds_info.save(filepath)
    
    #participants files
    participants_dir = valid_dir / "participants"
    participants_dir.mkdir(exist_ok=True)
    participants_file_path = participants_dir / "submission"
    ds_info = ds.make_full_ds(path_pref=str(participants_file_path), 
                              hide_labels=True)
    participants_file_path = participants_dir / "submission.tsv"
    ds_info.write_tsv(participants_file_path)
    
    
    
    
    
        
