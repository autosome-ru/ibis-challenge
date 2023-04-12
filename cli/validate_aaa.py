import sys
from pathlib import Path


sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

import argparse

from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.score_submission import ScoreSubmission, ScoreSubmissionFormatException

SUCCESS_CODE = 0
FORMAT_ERROR_CODE = 1
INTERNAL_ERROR_CODE = 2

parser = argparse.ArgumentParser(description="Python script to locally validate PWM submission")

parser.add_argument("--benchmark", 
                    help="benchmark config file (possible to provide file without valid dataset paths)",
                    required=True)

parser.add_argument("--aaa_sub", 
                    help="path to AAA submission file",
                    required=True)

args = parser.parse_args()

cfg = BenchmarkConfig.from_json(args.benchmark)
bench_tfs = set([ds.tf for ds in cfg.datasets])

try:
    submission = ScoreSubmission.load(args.aaa_sub)
    submission.validate(tfs=bench_tfs )
except ScoreSubmissionFormatException as exc:
    print(f"Format error detected: {exc}", file=sys.stderr)
    sys.exit(FORMAT_ERROR_CODE)
except Exception as exc:
    print(f"Uknown error occured: {exc}", file=sys.stderr)
    sys.exit(INTERNAL_ERROR_CODE)
else:
    print("Validation is successful", file=sys.stderr)
    sys.exit(SUCCESS_CODE)

    

