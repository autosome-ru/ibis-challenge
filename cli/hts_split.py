import sys 
import json
from pathlib import Path
import tqdm 

import argparse
import sys

from Bio.Seq import Seq
import random
import numpy as np 

def load_ds2flanks(path):
    with open(path) as inp:
        dt = json.load(inp)
    ds2flanks = {}
    for rep_id, rep_info in dt.items():
        rep_rest = {}
        for cycle, flanks in rep_info.items():
            rep_rest[int(cycle)] = flanks
        ds2flanks[int(rep_id)] = rep_rest
    return ds2flanks

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
parser.add_argument("--flanks", 
                    required=True, 
                    type=str)
parser.add_argument("--type", 
                    choices=['Leaderboard', 'Final'], 
                    required=True, type=str)
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
parser.add_argument("--sample_count",
                    default=100_000,
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

from bibis.sampling.gc import GCProfileMatcher
from bibis.benchmark.dataset import DatasetInfo
from bibis.seq.seqentry import SeqEntry, write as seq_write
from bibis.scoring.label import NO_LABEL
from bibis.seqdb.config import DBConfig 
from bibis.hts.config import HTSRawConfig
from bibis.hts.seqentry import SeqAssignEntry
from bibis.hts.utils import dispatch_samples
from bibis.utils import merge_fastqgz
from bibis.sampling.reservoir import (AllSelector,  
                                      PredefinedSizeUniformSelector)
from bibis.logging import get_bibis_logger

logger = get_bibis_logger()
    

EPS = 1e-10
BENCH_SEQDB_CFG = Path(args.tagdb_cfg)
CYCLE_CNT = 4
main_seed = args.seed

HTS_BENCH_DIR = Path(args.benchmark_out_dir)
HTS_BENCH_DIR.mkdir(parents=True, exist_ok=True)

cfg = HTSRawConfig.load(args.config_file)

#print(datasets)
train_datasets = cfg.splits.get('train')
test_datasets = cfg.splits.get('test')

if train_datasets is not None:
    logger.info(f"For factor {cfg.tf_name} the following replics are going to train:")
    train_dir = HTS_BENCH_DIR / "train" / cfg.tf_name  
    train_dir.mkdir(parents=True, exist_ok=True)

    for rep_ind, (rep, rep_info) in enumerate(train_datasets.items()):
        logger.info(f"\t{rep}")
        for cycle, ds in rep_info.items():
            
            out_path = train_dir / f"{cfg.tf_name}_R{rep_ind}_C{ds.cycle}_lf{ds.left_flank}_rf{ds.right_flank}.fastq.gz"
            if not out_path.exists() or args.recalc:
                merge_fastqgz(in_paths=ds.raw_paths, 
                            out_path=out_path)
else:
    logger.info(f"For factor {cfg.tf_name} no replics are going to train")

if test_datasets is None:
    logger.info(f"For factor {cfg.tf_name} no replics are going to test")
    exit(0) # nothing to done
else:
    logger.info(f"For factor {cfg.tf_name} the following replics are going to test:")
    for rep_ind, (rep, rep_info) in enumerate(test_datasets.items()):
        logger.info(f"\t{rep}")

valid_dir = HTS_BENCH_DIR / "valid" / cfg.tf_name  
valid_dir.mkdir(exist_ok=True, parents=True)
answer_valid_dir = valid_dir / "answer"
answer_valid_dir.mkdir(exist_ok=True)
participants_valid_dir = valid_dir / "participants"
participants_valid_dir.mkdir(exist_ok=True)
positives_path = answer_valid_dir / "positives.seqs"
foreigns_path = answer_valid_dir / "foreigns.seqs"
inputs_path = answer_valid_dir / "inputs.seqs"

logger.info("Creating positives and negatives .seqs files")
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

    gc_bins = np.arange(0, args.seq_length+EPS, 0.5) / args.seq_length

    positive_gc_profiles = {cycle: dict.fromkeys(gc_bins, 0) for cycle in range(1, CYCLE_CNT+1)}
    total_positive_gc_profile = dict.fromkeys(gc_bins, 0)

    foreigns_gc_profiles = {cycle: dict.fromkeys(gc_bins, 0) for cycle in range(1, CYCLE_CNT+1)}
    zero_gc_profile = dict.fromkeys(gc_bins, 0)
    
    logger.info("Processing positives")
    with open(ds.path, 'r') as assign, open(positives_path, 'w') as positive_fd:
        for line in tqdm.tqdm(assign):
            entry = SeqAssignEntry.from_line(line)
            if entry.stage_ind == cfg.stage_id:
                if entry.rep_ind in test_rep_ids:
                    to_take = positive_selectors[entry.cycle].add(entry)
                    if to_take:
                        positive_gc_profiles[entry.cycle][entry.gc_content] += 1
                        total_positive_gc_profile[entry.gc_content] += 1
                        print(line, file=positive_fd, end="")
                elif entry.cycle == 0:
                    zero_gc_profile[entry.gc_content] += 1
                elif entry.tf_ind != cfg.tf_id: # foreigns
                    foreigns_gc_profiles[entry.cycle][entry.gc_content] += 1
                else:
                    raise Exception(f"Wrong entry in assign file: {entry}")
            else:
                raise Exception(f"Assign file contains irrelevant stage id: {cfg.stage_id}")
    for cycle, sel in positive_selectors.items():
        if sel.count != sel.total_size:
            raise Exception(f"Positive selector for cycle {cycle} has not recieved all {sel.total_size} entries: {sel.count}")


    foreign_matcher = GCProfileMatcher.make(sample_per_object=args.foreign_neg2pos_ratio)
    foreigns_samplers = {}

    logger.info("Processing negatives")
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
            if entry.stage_ind == cfg.stage_id:
                if entry.cycle == 0:
                    to_take = input_samplers[entry.gc_content].add(entry)
                    if to_take:
                        print(line, file=inputs_fd, end="")
                elif entry.tf_ind != cfg.tf_id: # foreigns
                    to_take = foreigns_samplers[entry.cycle][entry.gc_content].add(entry)
                    if to_take:
                        print(line, file=foreigns_fd, end="")
            else:
                raise Exception(f"Wrong entry in assign file: {entry}")
    for gc, sel in input_samplers.items():
        if sel.count != sel.total_size:
            raise Exception(f"Zeros selector for gc {gc} has not recieved all {sel.total_size} entries: {sel.count}")
    
    for cycle, gc_samplers in foreigns_samplers.items():
        for gc, sel in gc_samplers.items():
            if sel.count != sel.total_size:
                raise Exception(f"Foreigns selector for cycle {cycle} for gc {gc} has not recieved all {sel.total_size} entries: {sel.count}")
else:
    logger.info("Skipping step as files already exists")


ds2flanks = load_ds2flanks(args.flanks)

db = DBConfig.load(BENCH_SEQDB_CFG).build()

seq_datasets: dict[str, list[SeqEntry]] = {}

user_known_samples: list[SeqEntry] = []
logger.info("Collectiong positives dataset")
# benchmark part files
with open(positives_path, "r") as inp:
    pos_samples: list[SeqEntry] = []
    for line in inp:
        entry = SeqAssignEntry.from_line(line)
        lf, rf = ds2flanks[entry.rep_ind][entry.cycle]
        seq_entry = SeqEntry(sequence=Seq(entry.seq), 
                             label=entry.cycle, 
                             metainfo={'rep': entry.rep_ind,
                                       'flanks': (lf, rf)})
        pos_samples.append(seq_entry)
pos_samples = db.taggify_entries(pos_samples)
user_known_samples.extend(pos_samples)

seq_datasets['positives'] = pos_samples

logger.info("Collectiong aliens dataset")
# foreign 
with open(foreigns_path, "r") as inp:
    alien_samples = []
    for line in inp:
        entry = SeqAssignEntry.from_line(line)
        lf, rf = ds2flanks[entry.rep_ind][entry.cycle]
        seq_entry = SeqEntry(sequence=Seq(entry.seq), 
                             label=0, 
                             metainfo={'rep': None,
                                       'flanks': (lf,rf) }) # always zero cycle 
        alien_samples.append(seq_entry)

alien_samples = db.taggify_entries(alien_samples)
user_known_samples.extend(alien_samples)

seq_datasets['alien'] = pos_samples + alien_samples

logger.info("Collectiong input dataset")
zero_flanks = [en.metainfo['flanks'] for en in pos_samples] * args.zero_neg2pos_ratio
rng = random.Random(args.seed)
rng.shuffle(zero_flanks)
with open(inputs_path, "r") as inp:
    input_samples = []
    for ind, line in enumerate(inp):
        entry = SeqAssignEntry.from_line(line)
        seq_entry = SeqEntry(sequence=Seq(entry.seq), 
                             label=0,
                             metainfo={'rep': None,
                                       'flanks': zero_flanks[ind]}) # always zero cycle 
        input_samples.append(seq_entry)

input_samples = db.taggify_entries(input_samples)
user_known_samples.extend(input_samples)
seq_datasets['input'] = pos_samples + input_samples

for dataset_name, samples in seq_datasets.items():
    logger.info(f"Writing {dataset_name} dataset")
    ds_dir = answer_valid_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = ds_dir / 'data.fasta'

    flanked_samples = []
    for entry in samples:
        lf, rf = entry.metainfo['flanks']
        flanked_seq = lf[1:] + str(entry.sequence) + rf[1:] 
        flanked_entry = SeqEntry(sequence=Seq(flanked_seq),
                                 tag=entry.tag,
                                 label=entry.label)
        flanked_samples.append(flanked_entry)
    seq_write(flanked_samples, fasta_path)

    answer = {'labels': {pe.tag: pe.label for pe in samples},
              'groups': {pe.tag: pe.metainfo['rep'] for pe in samples}}

    answer_path = ds_dir   / "data_answer.json"
    with open(answer_path, "w") as out:
        json.dump(answer, fp=out, indent=4)

    config_path = ds_dir / "config.json"
    ds_info = DatasetInfo(name = f"{cfg.tf_name}_{dataset_name}", 
                          tf = cfg.tf_name,
                          background=dataset_name,
                          fasta_path=str(fasta_path),
                          answer_path=str(answer_path))
    ds_info.save(config_path)

logger.info(f"Writing participants sequence file")
# write sequences for user
participants_fasta_path = participants_valid_dir / "submission.fasta"
random.shuffle(user_known_samples)
for entry in user_known_samples:
    entry.label = NO_LABEL
    entry.metainfo = {}
seq_write(user_known_samples, participants_fasta_path)