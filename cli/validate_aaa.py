import sys
from pathlib import Path


sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

import argparse

from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.score_submission import ScoreSubmission, ScoreSubmissionFormatException, ValidationResult

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

try:
    submission = ScoreSubmission.load(args.aaa_sub)
except ScoreSubmissionFormatException as exc:
    print(f"Format error detected: {exc}", file=sys.stderr)
    sys.exit(FORMAT_ERROR_CODE)
except Exception as exc:
    print(f"Uknown error occured: {exc}", file=sys.stderr)
    sys.exit(INTERNAL_ERROR_CODE)
else:
    val_res = submission.validate(cfg=cfg)
    if len(val_res.errors) != 0:
        for er in val_res.errors:
            print(f"Error detected: {er}", file=sys.stderr)
            
        if len(val_res.warnings) != 0:
            for war in val_res.warnings:
                print(f"Warning: {war}")
        sys.exit(FORMAT_ERROR_CODE)
       
    if len(val_res.warnings) != 0:
        for war in val_res.warnings:
            print(f"Warning: {war}")
    print("Validation is successful", file=sys.stderr)
    sys.exit(SUCCESS_CODE)

