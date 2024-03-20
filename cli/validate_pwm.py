import sys
from pathlib import Path



import argparse


SUCCESS_CODE = 0
FORMAT_ERROR_CODE = 1
INTERNAL_ERROR_CODE = 2

parser = argparse.ArgumentParser(description="Python script to locally validate PWM submission")

parser.add_argument("--benchmark", 
                    help="benchmark config file (possible to provide file without valid dataset paths)",
                    required=True)

parser.add_argument("--pwm_sub", 
                    help="path to pwm submission file",
                    required=True)

parser.add_argument("--bibis_root",
                    required=True,
                    help="Path to dir with bibis package (example - /home_local/dpenzar/bibis_git/ibis-challenge)",
                    type=str)

args = parser.parse_args()

sys.path.append(args.bibis_root)

from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.pwm_submission import PWMSubmission, PWMSubmissionFormatException

cfg = BenchmarkConfig.from_json(args.benchmark)

subm = PWMSubmission(name="to_validate",
                     path=args.pwm_sub,
                     available_tfs=cfg.tfs)

try:
    val_res = subm.validate(cfg) 
except PWMSubmissionFormatException as exc:
    print(f"Format error detected: {exc}", file=sys.stderr)
    sys.exit(FORMAT_ERROR_CODE)
except Exception as exc:
    print(f"Uknown error occured: {exc}", file=sys.stderr)
    sys.exit(INTERNAL_ERROR_CODE)
else:
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