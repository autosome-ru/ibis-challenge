import sys
import tempfile
sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

import argparse
from pathlib import Path


from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
import numpy as np 
rpy2_logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description="Python script to locally validate PWM submission")

parser.add_argument("--benchmark", 
                    help="benchmark config file (all dataset paths must be valid)",
                    required=True)
parser.add_argument("--sub", 
                    help="path to submission file",
                    required=True)
parser.add_argument("--sub_type",
                    choices=["aaa", "pwm"],
                    help="submission type",
                    required=True)
parser.add_argument("--scores_path",
                    default=sys.stdout,
                    help="Path to write output. By default - stdout")
parser.add_argument("--bibis_root",
                    required=True,
                    help="Path to dir with bibis package (example - /home_local/dpenzar/bibis_git/ibis-challenge)",
                    type=str)

sys.path.append(sys.path.append(str(Path(args.bibis_root).resolve())))



from bibis.benchmark.benchmark import Benchmark
from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.score_submission import ScoreSubmission, ScoreSubmissionFormatException
from bibis.benchmark.pwm_submission import PWMSubmission, PWMSubmissionFormatException

SUCCESS_CODE = 0
FORMAT_ERROR_CODE = 1
INTERNAL_ERROR_CODE = 2


args = parser.parse_args()

cfg = BenchmarkConfig.from_json(args.benchmark)
bench_tfs = set([ds.tf for ds in cfg.datasets])

if args.sub_type == "aaa":
    try:
        submission = ScoreSubmission.load(args.sub)
        submission.validate(cfg=cfg)
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
        submission.validate(cfg=cfg) 
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
    if bench.kind == "PBM":
        scores = scores[['tf', 'background', 'experiment_id', 'score', 'value', 'metainfo']]    
        scores.columns = ['tf', 'preprocessing', 'experiment_id', 'score_type', 'value', 'metainfo']
    else:
        scores = scores[['tf', 'background', 'score', 'value', 'metainfo']]    
        scores.columns = ['tf', 'background', 'score_type', 'value', 'metainfo']
elif args.sub_type == "pwm":
    if bench.kind == "PBM":
        scores = scores[['tf', 'part_name', 'scoring_type', 'background', 'experiment_id', 'score', 'value', 'metainfo']]    
        scores.columns = ['tf', 'matrix_name', 'scoring_type', 'background', 'experiment_id', 'score_type', 'value', 'metainfo']
    else:
         scores = scores[['tf', 'part_name', 'scoring_type', 'background', 'score', 'value', 'metainfo']]    
         scores.columns = ['tf', 'matrix_name', 'scoring_type', 'background', 'score_type', 'value', 'metainfo']
else:
    print("Wrong submission type", file=sys.stderr)
    sys.exit(INTERNAL_ERROR_CODE)

scores.to_csv(args.scores_path, 
              index=False,
              sep="\t")
