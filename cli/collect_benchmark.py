import argparse

import glob
import json
import sys
import random
import numpy as np

from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--benchmark_root", 
                    type=str,
                    required=True)
parser.add_argument("--benchmark_name", 
                    type=str,
                    required=True)
parser.add_argument("--benchmark_kind", 
                    type=str,
                    required=True)
parser.add_argument("--scorers", 
                    type=str,
                    required=True)
parser.add_argument("--out_dir",
                    type=str,
                    required=True)
parser.add_argument("--pwmeval",
                    type=str,
                    default="/home_local/dpenzar/PWMEval/pwm_scoring")
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
parser.add_argument("--log_path",
                    default="log.txt",
                    type=str)
parser.add_argument("--logger_name",
                    default=None,
                    type=str)
args = parser.parse_args()

sys.path.append(args.bibis_root)

from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.dataset import DatasetInfo, entries2tsv, seqentry2interval_key
from bibis.scoring.scorer import ScorerInfo
from bibis.benchmark.score_submission import ScoreSubmission
from bibis.benchmark.pwm_submission import PWMSubmission
from bibis.seq.seqentry import SeqEntry, read_fasta, write_fasta
from bibis.logging import get_logger, BIBIS_LOGGER_CFG
BIBIS_LOGGER_CFG.set_path(path=args.log_path)
log_name = args.logger_name
if log_name is None:
    log_name = f"collect_{args.benchmark_kind}"
logger = get_logger(name=log_name, path=args.log_path)

benchmark = Path(args.benchmark_root)

HTS_CYCLE_CNT = 4
logger.info(f"Collecting benchmark {args.benchmark_kind} {args.benchmark_name} ")
if args.benchmark_kind in ("GHTS", "CHS", "PBM", "SMS", "HTS"):   
    ds_cfg_paths = benchmark / "valid" / "*" / "answer" / "*" / "config.json"

    config_paths = glob.glob(str(ds_cfg_paths))
    assert len(config_paths) > 0
else:
    raise Exception(f"No config collection implemented for benchmark {args.benchmark_kind}")

logger.info("Collecting scorers info")
with open(args.scorers, "r") as out:
    scorers_dt = json.load(out)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

datasets = [DatasetInfo.load(p) for p in config_paths]

logger.info(f"Reading participants information")
if args.benchmark_kind in ("GHTS", "CHS", "PBM", "SMS", "HTS"):          
    sub_fasta_paths = glob.glob(str(benchmark / "valid" / "*" / "participants" / "*.fasta"))
    assert len(sub_fasta_paths) > 0
    unique_entries: dict[str, SeqEntry] = {}
    for path in sub_fasta_paths:
        entries = read_fasta(path)
        for e in entries:
            unique_entries[e.tag] = e
else:
    raise Exception("No sequence collection implemented for benchmark {args.benchmark_kind}")

logger.info("Shuffling benchmark entries")
final_entries = list(unique_entries.values())
if args.benchmark_kind in ("GHTS", "CHS"): 
    final_entries.sort(key=seqentry2interval_key)
elif args.benchmark_kind == "PBM":
    final_entries.sort(key=lambda pe: pe.tag)
elif args.benchmark_kind in ("SMS", 'HTS'):
    random.shuffle(final_entries)
else:
    raise Exception("No ordering implemented for benchmark {args.benchmark_kind}")

entries_tags = [e.tag for e in final_entries]

logger.info("Writing benchmark entries")
participants_fasta_path = out_dir / "participants.fasta"
write_fasta(entries=final_entries, 
            handle=participants_fasta_path)

participants_tsv_path = out_dir / "participants.bed"

entries2tsv(entries=final_entries, 
            path=participants_tsv_path,
            kind=args.benchmark_kind)

# WRITING BENCHMARK CONFIG
tags = {}
answers = {}
tfs = set()

logger.info("Reading benchmark labels")
ds_names = set()
for ds in datasets:
    tfs.add(ds.tf)
    ans = ds.answer()['labels']
    for tag, label in ans.items():
        tags[tag] = label
        answers[(ds.tf, tag)] = label
    if ds.name in ds_names: 
        raise Exception(f"Some datasets in benchmark have the same name {ds.name}")
    ds_names.add(ds.name)

if args.benchmark_kind in ("GHTS", "CHS", "SMS", 'HTS'):   
    all_tags = list(tags.keys())
elif args.benchmark_kind  == "PBM":
    all_tags = list(unique_entries.keys()) 
else:
    raise Exception("No config collection implemented for benchmark {args.benchmark_kind}")

assert set(entries_tags) == set(all_tags), "Answer tags and entries task must match"


logger.info("Writing benchmark config")
cfg = BenchmarkConfig(
    name=args.benchmark_name,
    kind=args.benchmark_kind,
    datasets=datasets,
    scorers=[ScorerInfo.from_dict(sc) for sc in scorers_dt],
    pwmeval_path=args.pwmeval,
    tfs=list(tfs),
    tags=entries_tags,
    metainfo={}    
)

cfg_path = out_dir / f"benchmark.json"
cfg.save(cfg_path)

## WRITING TEMPLATES
logger.info("Writing template submissions")
score_template = ScoreSubmission.template(tag_col_name="tag",
                                          tf_names=cfg.tfs,
                                          tags=cfg.tags)
aaa_template_path = out_dir / "aaa_template.tsv"
score_template.write(aaa_template_path)

for tf in score_template.tf_names:
    for tag in score_template.tags:
        score_template[tf][tag] = np.random.random()
random_aaa_path = out_dir / "aaa_random.tsv"
score_template.write(random_aaa_path)

for tf in score_template.tf_names:
    for tag in score_template.tags:
        score = answers.get((tf, tag), 0)
        if args.benchmark_kind == "HTS":
            score = score / HTS_CYCLE_CNT
        score_template[tf][tag] = score

ideal_aaa_path = out_dir / "aaa_ideal.tsv"
score_template.write(ideal_aaa_path)    

pwm_submission_path = out_dir / "pwm_submission.txt"
with open(pwm_submission_path, "w") as out:
    for ind, tf in enumerate(tfs):
        for i in range(PWMSubmission.MAX_PWM_PER_TF):
            tag = f"{tf}_motif{i+1}"
            print(f">{tf} {tag}", file=out)
            for i in range(np.random.randint(PWMSubmission.MIN_PWM_LENGTH,
                                             PWMSubmission.MAX_PWM_LENGTH)):
                a, t, g, c = np.random.dirichlet([1,1,1,1])
                p = PWMSubmission.MAX_PRECISION
                print(f"{a:.0{p}f} {t:.0{p}f} {g:.0{p}f} {c:.0{p}f}", file=out)
            print(file=out)