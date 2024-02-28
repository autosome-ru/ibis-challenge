from collections import defaultdict
import sys 
import json
from pathlib import Path
from copy import copy
import tqdm 

import argparse
import sys

from Bio.Seq import Seq
import random
import numpy as np 

parser = argparse.ArgumentParser()

def sample_count_conv(arg):
    if arg == "all":
        return arg
    else:
        return int(arg)
parser.add_argument("--benchmark_out_dir", 
                    required=True, 
                    type=str)
parser.add_argument("--seq_length",
                    default=40, 
                    type=int)
parser.add_argument("--tagdb_cfg",
                    required=True,
                    type=str)
parser.add_argument("--config_file", 
                    required=True, 
                    type=str)
parser.add_argument("--type", 
                    choices=['Leaderboard', 'Final'], 
                    required=True, type=str)
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
parser.add_argument("--keep_left_cnt",
                    default=18,
                    type=int)
parser.add_argument("--keep_right_cnt",
                    default=14,
                    type=int)
parser.add_argument("--sample_count",
                    default=200_000,
                    type=sample_count_conv)
parser.add_argument("--foreign_neg2pos_ratio",
                    default=2,
                    type=int)
parser.add_argument("--zero_neg2pos_ratio",
                    default=2,
                    type=int)
parser.add_argument("--recalc",
                    action='store_true')
parser.add_argument("--seed",
                    action='store_true',
                    default=777)


args = parser.parse_args()


sys.path.append(args.bibis_root) # temporary solution while package is in development

from bibis.sampling.gc import GCProfileMatcher, SetGCSampler
from bibis.benchmark.dataset import DatasetInfo
from bibis.seq.seqentry import SeqEntry, write as seq_write
from bibis.scoring.label import NO_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL
from bibis.seqdb.config import DBConfig 
from bibis.hts.config import HTSRawConfig
from bibis.hts.dataset import HTSRawDataset
from bibis.hts.seqentry import SeqAssignEntry
from bibis.hts.utils import dispatch_samples
from bibis.utils import merge_fastqgz
from bibis.sampling.reservoir import (AllSelector,  
                                      PredefinedSizeUniformSelector, 
                                      UniformSampler, 
                                      WeightSampler)

EPS = 1e-10
BENCH_SEQDB_CFG = Path(args.tagdb_cfg)
CYCLE_CNT = 4
main_seed = args.seed

HTS_BENCH_DIR = Path(args.benchmark_out_dir)
HTS_BENCH_DIR.mkdir(parents=True, exist_ok=True)

cfg = HTSRawConfig.load(args.config_file)

assert cfg.split in ("Train", "Test", "Train/Test"), "wrong split"


def split_datasets(cfg: HTSRawConfig):
    datasets = cfg.datasets
    groups = defaultdict(lambda: defaultdict(dict))

    for ds in datasets:
        groups[ds.exp_tp][ds.rep][ds.cycle] = ds

    train_datasets = {}
    test_datasets = {}

    cur = test_datasets
    nxt = train_datasets
    for _, rep_dt in groups.items():
        for rep, ds_cycles in rep_dt.items():
            cur[rep] = ds_cycles
            cur, nxt = nxt, cur
    return train_datasets, test_datasets

def split_by_rep(datasets: list[HTSRawDataset]):
    spl = defaultdict(dict)
    for ds in datasets:
        spl[ds.rep][ds.cycle] = ds
    return spl
    

datasets = copy(cfg.datasets)
#print(datasets)

if cfg.split == "Train":
    train_datasets = split_by_rep(datasets)
    test_datasets = None
else:
    if cfg.split == "Train/Test":
        train_datasets, test_datasets = split_datasets(cfg)
    else: # cfg.split == "Test"
        train_datasets = None
        test_datasets = split_by_rep(datasets) 



if train_datasets is not None:
    print("Train datasets: ", train_datasets.keys())
    train_dir = HTS_BENCH_DIR / "train" / cfg.tf_name  
    train_dir.mkdir(parents=True, exist_ok=True)

    for rep_ind, (rep, rep_info) in enumerate(train_datasets.items()):
        for cycle, ds in rep_info.items():
            
            out_path = train_dir / f"{cfg.tf_name}_R{rep_ind}_C{ds.cycle}_lf{ds.left_flank}_rf{ds.right_flank}.fastq.gz"
            if not out_path.exists() or args.recalc:
                merge_fastqgz(in_paths=ds.raw_paths, 
                            out_path=out_path)

if test_datasets is None:
    exit(0) # nothing to done

print("Test datasets", test_datasets.keys())
valid_dir = HTS_BENCH_DIR / "valid" / cfg.tf_name  
valid_dir.mkdir(exist_ok=True, parents=True)
answer_valid_dir = valid_dir / "answer"
answer_valid_dir.mkdir(exist_ok=True)
participants_valid_dir = valid_dir / "participants"
participants_valid_dir.mkdir(exist_ok=True)
positives_path = answer_valid_dir / "positives.seqs"
foreigns_path = answer_valid_dir / "foreigns.seqs"
inputs_path = answer_valid_dir / "inputs.seqs"



if not (positives_path.exists() and foreigns_path.exists() and inputs_path.exists()) or args.recalc:

    positive_cycle_sizes = {cycle: 0 for cycle in range(1, CYCLE_CNT+1)}
    test_rep_ids = set()
    for _, (_, rep_info) in enumerate(test_datasets.items()):
        for _, ds in rep_info.items():
            test_rep_ids.add(ds.rep_id) 
            positive_cycle_sizes[ds.cycle] += ds.size

    main_seed += sum(positive_cycle_sizes.values())

    total_cycle_sizes = positive_cycle_sizes
    if args.sample_count != 'all':
        positive_cycle_sizes = dispatch_samples(positive_cycle_sizes, args.sample_count)
        positive_selectors = {cycle: PredefinedSizeUniformSelector(sample_size=size,
                                                                   total_size=total_cycle_sizes[cycle],
                                                                   seed=main_seed+cycle)\
                                    for cycle, size in positive_cycle_sizes.items()}
    else:
        positive_selectors = {cycle: AllSelector() for cycle in positive_cycle_sizes.keys()}
    
    main_seed += CYCLE_CNT

    print(total_cycle_sizes)
    print(positive_cycle_sizes)
    print(test_rep_ids)

    gc_bins = np.arange(0, args.seq_length+EPS, 0.5) / args.seq_length

    positive_gc_profiles = {cycle: dict.fromkeys(gc_bins, 0) for cycle in range(1, CYCLE_CNT+1)}
    total_positive_gc_profile = dict.fromkeys(gc_bins, 0)

    foreigns_gc_profiles = {cycle: dict.fromkeys(gc_bins, 0) for cycle in range(1, CYCLE_CNT+1)}
    zero_gc_profile = dict.fromkeys(gc_bins, 0)

    with open(ds.path, 'r') as assign, open(positives_path, 'w') as positive_fd:
        for line in tqdm.tqdm(assign):
            entry = SeqAssignEntry.from_line(line)
            if entry.rep_ind in test_rep_ids:
                to_take = positive_selectors[entry.cycle].add(entry)
                if to_take:
                    positive_gc_profiles[entry.cycle][entry.gc_content] += 1
                    total_positive_gc_profile[entry.gc_content] += 1
                    print(entry.to_line(), file=positive_fd)
            elif entry.cycle == 0:
                zero_gc_profile[entry.gc_content] += 1
            elif entry.tf_ind != cfg.tf_id and entry.stage_ind == cfg.stage_id: # foreigns
                foreigns_gc_profiles[entry.cycle][entry.gc_content] += 1
    for cycle, sel in positive_selectors.items():
        if sel.count != sel.total_size:
            raise Exception(f"Positive selector for cycle {cycle} has not recieved all {sel.total_size} entries: {sel.count}")

    
    print('Positive', positive_gc_profiles)
    print('Zero', zero_gc_profile)
    print('Foreigns', foreigns_gc_profiles)


    foreign_matcher = GCProfileMatcher.make(sample_per_object=args.foreign_neg2pos_ratio)
    foreigns_samplers = {}
    print("Foreigns")
    for cycle, pos_gc_counts in positive_gc_profiles.items():
        foreign_gc_counts = foreigns_gc_profiles[cycle]
        foreign_assign = foreign_matcher.match(positives_profile=pos_gc_counts, 
                                            negatives_profile=foreign_gc_counts)
        foreigns_samplers[cycle] = {gc: PredefinedSizeUniformSelector(sample_size=size,
                                                                      total_size=foreign_gc_counts[gc],
                                                                      seed=main_seed+ind)\
                                                                        for ind, (gc, size)\
                                                                         in enumerate(foreign_assign.items())}
        main_seed += len(foreign_assign)

    inputs_matcher = GCProfileMatcher.make(sample_per_object=args.zero_neg2pos_ratio)
    inputs_assign = inputs_matcher.match(positives_profile=total_positive_gc_profile,
                                        negatives_profile=zero_gc_profile)
    input_samplers = {gc: PredefinedSizeUniformSelector(sample_size=size,
                                                        total_size=zero_gc_profile[gc],
                                                        seed=main_seed+ind)\
                                                            for ind, (gc, size) in enumerate(inputs_assign.items())}
    main_seed += len(inputs_assign)
    with open(ds.path, 'r') as assign, open(foreigns_path, 'w') as foreigns_fd, open(inputs_path, 'w') as inputs_fd:
        for line in tqdm.tqdm(assign):
            entry = SeqAssignEntry.from_line(line)
            if entry.cycle == 0:
                to_take = input_samplers[entry.gc_content].add(entry)
                if to_take:
                    print(entry.to_line(), file=inputs_fd)
            elif entry.tf_ind != cfg.tf_id and entry.stage_ind == cfg.stage_id: # foreigns
                to_take = foreigns_samplers[entry.cycle][entry.gc_content].add(entry)
                if to_take:
                    print(entry.to_line(), file=foreigns_fd)
    for gc, sel in input_samplers.items():
        if sel.count != sel.total_size:
            raise Exception(f"Zeros selector for gc {gc} has not recieved all {sel.total_size} entries: {sel.count}")
    
    for cycle, gc_samplers in foreigns_samplers.items():
        for gc, sel in gc_samplers.items():
            if sel.count != sel.total_size:
                raise Exception(f"Foreigns selector for cycle {cycle} for gc {gc} has not recieved all {sel.total_size} entries: {sel.count}")
            else:
                print(cycle, gc, sel.count, sel.total_size)

left_flank = ds.left_flank[1:]
right_flank = ds.right_flank[1:]

left_flank = left_flank[:args.keep_left_cnt] + (len(left_flank) - args.keep_left_cnt) * 'N'
right_flank = (len(right_flank) - args.keep_right_cnt) * 'N' + right_flank[-args.keep_right_cnt:]
print(left_flank, args.keep_left_cnt)
print(right_flank, args.keep_right_cnt)
db = DBConfig.load(BENCH_SEQDB_CFG).build()

# benchmark part files
with open(positives_path, "r") as inp:
    pos_samples = []
    for line in inp:
        entry = SeqAssignEntry.from_line(line)

        seq_entry = SeqEntry(sequence=Seq(entry.seq), label=entry.cycle)
        pos_samples.append(seq_entry)
        
pos_samples = db.taggify_entries(pos_samples)

user_known_samples: list[SeqEntry] = []
user_known_samples.extend(pos_samples)


# foreign 
with open(foreigns_path, "r") as inp:
    neg_samples = []
    for line in inp:
        entry = SeqAssignEntry.from_line(line)

        seq_entry = SeqEntry(sequence=Seq(entry.seq), label=0) # always zero cycle 
        neg_samples.append(seq_entry)


neg_samples = db.taggify_entries(neg_samples)
user_known_samples.extend(neg_samples)

foreign_ds_dir = answer_valid_dir /  "foreign"
foreign_ds_dir.mkdir(parents=True, exist_ok=True)
samples: list[SeqEntry] = pos_samples + neg_samples    

fasta_path = foreign_ds_dir  / "data.fasta"

flanked_samples = []
for entry in samples:
    flanked_seq = left_flank + str(entry.sequence) + right_flank
    flanked_entry = SeqEntry(sequence=Seq(flanked_seq),
                             tag=entry.tag,
                             label=entry.label)
    flanked_samples.append(flanked_entry)
seq_write(flanked_samples, fasta_path)

        
answer = {pe.tag: pe.label for pe in samples}
answer_path = foreign_ds_dir   / "data_answer.json"
with open(answer_path, "w") as out:
    json.dump(answer, fp=out, indent=4)

config_path = foreign_ds_dir / "config.json"
ds_info = DatasetInfo(name = f"{cfg.tf_name}_foreign", 
                      tf = cfg.tf_name,
                      background="foreign",
                      fasta_path=str(fasta_path),
                      answer_path=str(answer_path),
                      left_flank=left_flank,
                      right_flank=right_flank)
ds_info.save(config_path)

# zero seqs 
with open(inputs_path, "r") as inp:
    neg_samples = []
    for line in inp:
        entry = SeqAssignEntry.from_line(line)

        seq_entry = SeqEntry(sequence=Seq(entry.seq), label=0) # always zero cycle 
        neg_samples.append(seq_entry)

neg_samples = db.taggify_entries(neg_samples)
user_known_samples.extend(neg_samples)

zeros_ds_dir =  answer_valid_dir / "input"
zeros_ds_dir.mkdir(parents=True, exist_ok=True)
samples: list[SeqEntry] = pos_samples + neg_samples    

fasta_path = zeros_ds_dir  / "data.fasta"
flanked_samples = []

for entry in samples:
    flanked_seq = left_flank + str(entry.sequence) + right_flank
    flanked_entry = SeqEntry(sequence=Seq(flanked_seq),
                             tag=entry.tag,
                             label=entry.label)
    flanked_samples.append(flanked_entry)
seq_write(flanked_samples, fasta_path)
        
answer = {pe.tag: pe.label for pe in samples}
answer_path = zeros_ds_dir   / "data_answer.json"
with open(answer_path, "w") as out:
    json.dump(answer, fp=out, indent=4)

config_path = zeros_ds_dir/ "config.json"
ds_info = DatasetInfo(name = f"{cfg.tf_name}_input", 
                      tf = cfg.tf_name,
                      background="input",
                      fasta_path=str(fasta_path),
                      answer_path=str(answer_path),
                      left_flank=left_flank,
                      right_flank=right_flank)
ds_info.save(config_path)

# write sequences for user
participants_fasta_path = participants_valid_dir / "submission.fasta"
random.shuffle(user_known_samples)
for entry in user_known_samples:
    entry.label = NO_LABEL
    entry.metainfo = {}
seq_write(user_known_samples, participants_fasta_path)
