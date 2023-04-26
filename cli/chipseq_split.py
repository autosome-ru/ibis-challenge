from dataclasses import dataclass, asdict
import sys 
import json
from pathlib import Path

import argparse
import sys

sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")
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

BEDTOOLS_PATH: Path = Path("/home_local/dpenzar/bedtools2/bin")
BedtoolsExecutor.set_defaull_executor(BEDTOOLS_PATH)

BENCH_PROCESSED_DIR = Path("/home_local/dpenzar/BENCHMARK_PROCESSED")
BENCH_SEQDB_CFG = BENCH_PROCESSED_DIR / "tag.json"
CHS_BENCH_DIR = BENCH_PROCESSED_DIR / "CHS"
CHS_BENCH_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--config_file", required=True, type=str)
parser.add_argument("--n_procs", default=1, type=int)

args = parser.parse_args()

cfg = ChipSeqConfig.load(args.config_file)

tf_peaks = [ChIPPeakList.read(t) for t in cfg.tf_path]
tf_beds = [f.to_beddata() for f in tf_peaks]


        
        
if "train" in cfg.splits:
    if "test" in cfg.splits:
        ind, _ = max(enumerate(tf_beds), key=lambda x: len(x[1]))
        train_beds = list(tf_beds)
        train_beds.pop(ind)
    else:
        train_beds = tf_beds
    
    split = cfg.splits['train']
    train_beds = [bed.filter(lambda e: e.chr in split.chroms) for bed in train_beds]
    train_dir = CHS_BENCH_DIR / "train" / cfg.tf_name
    train_dir.mkdir(exist_ok=True, parents=True)
    
    for ind, bed in enumerate(train_beds, 1):
        replic_path = train_dir / f"replic_{ind}.bed"
        bed.write(path=train_dir / replic_path, write_peak=True)

if "test" in cfg.splits:
    ind, valid_bed = max(enumerate(tf_beds), key=lambda x: len(x[1]))
    train_beds = list(tf_beds)
    train_beds.pop(ind)
    
    
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
    
    print("Creating foreigns") 
    foreign_sampler = ChIPForeignSampler.make(window_size=cfg.window_size,
                                            genome=genome,
                                            tf_peaks=[valid_bed], # type: ignore
                                            real_peaks=foreign_beds,
                                            friend_peaks=friends_peaks,
                                            black_list_regions=valid_black_list,
                                            sample_per_object=cfg.foreign_cfg.balance,
                                            seed=cfg.seed)

    samples['foreigns'] = foreign_sampler.sample_bed()
    friends_peaks.append(samples['foreigns'])
    
    print("Creating genome samples")
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
    
    samples['genome'] = genome_sampler.sample_bed()
    friends_peaks.append(samples['genome'])
    
    valid_dir = CHS_BENCH_DIR / "valid" / cfg.tf_name
    valid_dir.mkdir(exist_ok=True, parents=True)
    
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
    for background in ('shades', 'foreigns', 'genome'):
        filepath = answer_dir / f"{cfg.tf_name}_{background}.fasta"
        ds_info = ds.make_ds(background=background,
                   path=filepath,
                   hide_labels=False)
        filepath = answer_dir / f"{cfg.tf_name}_{background}.json"
        ds_info.save(filepath)
    
    #participants files
    participants_dir = valid_dir / "participants"
    participants_dir.mkdir(exist_ok=True)
    participants_file_path = participants_dir / "submission.fasta"
    ds_info = ds.make_full_ds(participants_file_path, hide_labels=True)
    participants_file_path = participants_dir / "submission.tsv"
    ds_info.write_tsv(participants_file_path)
    
    
    
    
    
        
