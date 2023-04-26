import sys
import tempfile
sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

import argparse
from pathlib import Path

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)

from bibis.benchmark.benchmark import Benchmark
from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.score_submission import ScoreSubmission, ScoreSubmissionFormatException
from bibis.benchmark.pwm_submission import PWMSubmission, PWMSubmissionFormatException

SUCCESS_CODE = 0
FORMAT_ERROR_CODE = 1
INTERNAL_ERROR_CODE = 2

parser = argparse.ArgumentParser(description="Python script to locally validate PWM submission")

parser.add_argument("--benchmark", 
                    help="benchmark config file (all dataset paths must be valid)",
                    required=True)

parser.add_argument("--sub", 
                    help="path to submission file",
                    required=True)

parser.add_argument("--sub_type",
                    choices=["aaa", "pwm"],
                    help="submission type")

parser.add_argument("--scores_path",
                    default=sys.stdout,
                    help="Path to write output. By default - stdout")

args = parser.parse_args()

cfg = BenchmarkConfig.from_json(args.benchmark)
bench_tfs = set([ds.tf for ds in cfg.datasets])

if args.sub_type == "aaa":
    try:
        submission = ScoreSubmission.load(args.sub)
        submission.validate(tfs=bench_tfs )
    except ScoreSubmissionFormatException as exc:
        print(f"Format error detected: {exc}", file=sys.stderr)
        sys.exit(FORMAT_ERROR_CODE)
    except Exception as exc:
        print(f"Uknown error occured: {exc}", file=sys.stderr)
        sys.exit(INTERNAL_ERROR_CODE)
elif args.sub_type == "pwm":
    submission = PWMSubmission(name="to_validate",
                        path=args.sub,
                        available_tfs=bench_tfs)
    try:
        submission.validate() 
    except PWMSubmissionFormatException as exc:
        print(f"Format error detected: {exc}", file=sys.stderr)
        sys.exit(FORMAT_ERROR_CODE)
    except Exception as exc:
        print(f"Uknown error occured: {exc}", file=sys.stderr)
        sys.exit(INTERNAL_ERROR_CODE)
else:
    print("Wrong submission type", file=sys.stderr)
    sys.exit(INTERNAL_ERROR_CODE)

with tempfile.TemporaryDirectory() as tempdir:
    tempdir = Path(tempdir)
    bench = Benchmark.from_cfg(cfg, 
                            results_dir=tempdir)
    bench.submit(submission)
    scores = bench.run()
    
if args.sub_type == "aaa":
    scores = scores[['tf', 'background', 'score', 'value']]    
    scores.columns = ['tf', 'background', 'score_type', 'value']
elif args.sub_type == "pwm":
    scores = scores[['tf', 'part_name', 'scoring_type', 'background', 'score', 'value']]    
    scores.columns = ['tf', 'matrix_name', 'scoring_type', 'background', 'score_type', 'value']
else:
    print("Wrong submission type", file=sys.stderr)
    sys.exit(INTERNAL_ERROR_CODE)

BACKGROUND_OUTER_NAMING = {
    "shades":"shades", 
    "foreigns":"aliens",
    "genome":"random"
}
scores['background'] = scores['background'].apply(lambda x: BACKGROUND_OUTER_NAMING[x])

scores.to_csv(args.scores_path, index=False, sep="\t")
