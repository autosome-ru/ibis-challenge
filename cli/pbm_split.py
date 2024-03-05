from dataclasses import dataclass, asdict
import sys 
import json
import shutil 
from pathlib import Path

import argparse
import sys


parser = argparse.ArgumentParser()

parser.add_argument("--benchmark_out_dir", 
                    required=True, 
                    type=str)
# pbms already have tags 
#parser.add_argument("--tagdb_cfg",
#                    required=True,
#                   type=str)
parser.add_argument("--config_file", 
                    required=True, 
                    type=str)
parser.add_argument("--type", 
                    choices=['Leaderboard', 'Final'], 
                    required=True, 
                    type=str)
parser.add_argument("--n_procs", 
                    default=1,
                    type=int)
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
parser.add_argument("--remove_grey_zone",
                    action="store_true")
parser.add_argument("--log_path",
                    default='log.txt')
parser.add_argument("--logger_name",
                    default="pbm_split")

args = parser.parse_args()

sys.path.append(args.bibis_root) # temporary solution while package is in development

from bibis.pbm.config import PBMConfig 
from bibis.seq.seqentry import SeqEntry, write as seq_write
from bibis.sampling.gc import SetGCSampler
from bibis.benchmark.dataset import DatasetInfo, entries2tsv
from bibis.pbm.pbm import PBMExperiment
from bibis.pbm.pbm_protocol import IbisProtocol
from bibis.logging import get_logger, BIBIS_LOGGER_CFG
BIBIS_LOGGER_CFG.set_path(path=args.log_path)
logger = get_logger(name=args.logger_name, path=args.log_path)


PBM_BENCH_DIR = Path(args.benchmark_out_dir)
PBM_BENCH_DIR.mkdir(parents=True, exist_ok=True)

cfg = PBMConfig.load(args.config_file)

if len(cfg.train_paths) != 0:
    logger.info("Writing train datasets")
    train_dir = PBM_BENCH_DIR / "train" / cfg.tf_name
    train_dir.mkdir(exist_ok=True, parents=True)
    for path in cfg.train_paths:
        shutil.copy(path, train_dir / Path(path).name)

    # just copy file without any modification

if cfg.protocol == "ibis":
    protocol = IbisProtocol()
else:
    raise Exception(f"Protocol is not implemented: {cfg.protocol}")

if len(cfg.test_paths) == 0:
    exit(0)

valid_dir = PBM_BENCH_DIR / "valid" / cfg.tf_name
valid_dir.mkdir(exist_ok=True, parents=True)

joined_pos_tags = set()

datasets: dict[str, dict[str, list[SeqEntry]]] = {}
experiments: list[PBMExperiment] = []
logger.info("Selecting positives")
for test_path in cfg.test_paths:
    experiment = PBMExperiment.read(path=test_path)
    if "/QNZS/" in test_path:
        preprocessing = "QNZS"
    elif "/SD/" in test_path:
        preprocessing = "SD"
    else:
        raise Exception(f"Cant deduce preprocessing type for file {test_path}")
    pos_entries, neg_entries = protocol.process_pbm(experiment, preprocessing=preprocessing)
    
    name = Path(test_path).name.replace(".tsv", "")
    name = f"{preprocessing}_{name}"
    if name in datasets:
        raise Exception(f"Duplicated info for {name} dataset")
    
    experiments.append(experiment)
    datasets[name] = {"pos": pos_entries, "neg": neg_entries}
    joined_pos_tags.update(pe.tag for pe in pos_entries)

if args.remove_grey_zone:
    logger.info("Removing tags appearing as positives in at least one dataset from possible negatives")
    for ds in datasets.values():
        ds["neg"] = [pe for pe in ds["neg"] if not pe.tag in joined_pos_tags]


logger.info("Selecting negatives and writing final datasets")
answers_dir = valid_dir / 'answer'
answers_dir.mkdir(exist_ok=True, parents=True)
for name, ds in datasets.items():
    ds_dir = answers_dir /  name
    ds_dir.mkdir(exist_ok=True, parents=True)

    negative_sampler = SetGCSampler.make(negatives=ds["neg"],
                                            sample_per_object=cfg.neg2pos_ratio)
    pos_samples = ds["pos"]
    neg_samples = negative_sampler.sample(positive=pos_samples, return_loss=False)
    samples = pos_samples + neg_samples        
    fasta_path = ds_dir  / "data.fasta"
    seq_write(samples, fasta_path)


    answer = {"labels": {pe.tag: pe.label for pe in samples}}
    answer_path = ds_dir / "data_answer.json"
    with open(answer_path, "w") as out:
        json.dump(answer, fp=out, indent=4)

    config_path = ds_dir / "config.json"
    ds_info = DatasetInfo(name = f"{cfg.tf_name}_{name}", 
                            tf = cfg.tf_name,
                            background=name,
                            fasta_path=str(fasta_path),
                            answer_path=str(answer_path))
    ds_info.save(config_path)

logger.info("Collecting tags used")
tags = {rec.id_probe for rec in experiments[0]}
for exp in experiments[1:]:
    other_tags = {rec.id_probe for rec in exp}
    if tags != other_tags:
        raise Exception(f"Experiments have different tags {tags.symmetric_difference(tags, other_tags)}")

logger.info("Writing participants file")
participants_dir = valid_dir / "participants"
participants_dir.mkdir(exist_ok=True, parents=True)
participants_file_path = participants_dir / "submission.tsv"
entries2tsv([pe.to_seqentry() for pe in experiments[0]],
                path=participants_file_path,
                kind="PBM")

participants_file_path = participants_dir / "submission.fasta"
entries = [pe.to_seqentry() for pe in experiments[0]]
seq_write(entries, participants_file_path)