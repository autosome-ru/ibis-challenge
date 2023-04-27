import argparse

import glob
import json
import sys
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
args = parser.parse_args()

sys.path.append(args.bibis_root)

from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.dataset import DatasetInfo
from bibis.scoring.scorer import ScorerInfo
#from bibis.benchmark.benchmark import Benchmark
from bibis.seq.seqentry import read_fasta
from bibis.benchmark.score_submission import ScoreSubmission
from bibis.benchmark.pwm_submission import PWMSubmission

#~/BENCHMARK_PROCESSED_NEW/CHS/Leaderboard

benchmark = Path(args.benchmark_root)

ds_cfg_paths = benchmark / "valid" / "*" / "answer" / "*.json"

config_paths = glob.glob(str(ds_cfg_paths))

with open(args.scorers, "r") as out:
    scorers_dt = json.load(out)

cfg = BenchmarkConfig(
    name=args.benchmark_name,
    kind=args.benchmark_kind,
    datasets=[DatasetInfo.load(p) for p in config_paths],
    scorers=[ScorerInfo.from_dict(sc) for sc in scorers_dt],
    pwmeval_path=args.pwmeval,
    metainfo={}    
)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

cfg_path = out_dir / f"benchmark.json"
cfg.save(cfg_path)

# collect tags 
tags = {}
answers = {}
tfs = set()

g_obs = set()
for ds in cfg.datasets:
    tfs.add(ds.tf)
    fs = read_fasta(ds.path)
    for e in fs:
        tags[e.tag] = e.label
        answers[(ds.tf, e.tag)] = e.label

score_template = ScoreSubmission.template(tag_col_name="peaks",
                                          tf_names=list(tfs),
                                          tags=list(tags.keys()))
aaa_template_path = out_dir / "aaa_template.txt"
score_template.write(aaa_template_path)

for tf in score_template.tf_names:
    for tag in score_template.tags:
        score_template[tf][tag] = np.random.random()
random_aaa_path = out_dir / "aaa_random.txt"
score_template.write(random_aaa_path)

for tf in score_template.tf_names:
    for tag in score_template.tags:
        score_template[tf][tag] = answers.get((tf, tag), 0)

ideal_aaa_path = out_dir / "aaa_ideal.txt"
score_template.write(ideal_aaa_path)    

pwm_submission_path = out_dir / "pwm_submission.txt"
with open(pwm_submission_path, "w") as out:
    for ind, tf in enumerate(tfs):
        tag = f"tag{ind+1}"
        print(f">{tf} {tag}", file=out)
        for i in range(np.random.randint(3, 11)):
            a, t, g, c = np.random.dirichlet([1,1,1,1])
            p = PWMSubmission.MAX_PRECISION
            print(f"{a:.0{p}f} {t:.0{p}f} {g:.0{p}f} {c:.0{p}f}", file=out)
        print(file=out)
        
    
        


