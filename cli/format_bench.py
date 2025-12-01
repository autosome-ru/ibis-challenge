from importlib.metadata import requires
import tempfile
import sys
import argparse
from pathlib import Path
 
from bibis.benchmark.benchmark import Benchmark
from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.score_submission import ScoreSubmission, ScoreSubmissionFormatException
from bibis.benchmark.pwm_submission import PWMSubmission, PWMSubmissionFormatException


parser = argparse.ArgumentParser(description="Python script to adapt benchmark config to a new location")

parser.add_argument("--in_benchmark_path", 
                    help="benchmark config file (all dataset paths must be valid)",
                    required=True)
parser.add_argument("--root_dir", 
                    help="path to directiry with benchmark files",
                    required=True)
parser.add_argument("--pwmeval_path",
                    help="path to pwmeval",
                    required=True)
parser.add_argument("--out_benchmark_path",
                    help="Path to write output")
parser.add_argument("--root_label",
                    default="__ROOTDIR__")

args = parser.parse_args()

SUCCESS_CODE = 0
FORMAT_ERROR_CODE = 1
INTERNAL_ERROR_CODE = 2


cfg = BenchmarkConfig.from_json(args.in_benchmark_path)

if args.root_dir.endswith('/'):
    args.root_dir = args.root_dir[:-1]
if args.root_label.endswith('/'):
    args.root_label = args.root_label[:-1]

for ds in cfg.datasets:
    if ds.answer_path is None:
        raise Exception("Wrong config: no answers path speciifed")
    if args.root_label not in ds.answer_path:
        raise Exception("Wrong root label: not found in config answer path")
    ds.answer_path = ds.answer_path.replace(args.root_label, args.root_dir, 1)
    if args.root_label not in ds.fasta_path:
        raise Exception("Wrong root label: not found in config fasta path")
    ds.fasta_path = ds.fasta_path.replace(args.root_label, args.root_dir, 1)
cfg.pwmeval_path = args.pwmeval_path

cfg.save(args.out_benchmark_path)