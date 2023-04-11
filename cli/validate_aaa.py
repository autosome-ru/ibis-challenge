import sys
from pathlib import Path


sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

import argparse

from bibis.benchmark.benchmarkconfig import BenchmarkConfig
from bibis.benchmark.pwm_submission import PWMSubmission, PWMSubmissionFormatException

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