# Project structure

* bibis - main package, all tools for benchmarking, sampling, etc
* safe_examples - toy examples without any possibility of data leakage
* ipynb - jupyter-notebooks with examples how to build submissions, benchmark configs and etc
* cli - command-line-wrappers

# PWMEval

To be able to run pwm scoring you need to download and install PWMEval from https://github.com/gio31415/PWMEval
and update the path to PWMEval in benchmark_example.json

# CLI: 

* validate_aaa.py - script to validate aaa submission 
example run:

python cli/validate_aaa.py --benchmark safe_examples/benchmark_example.json --aaa_sub safe_examples/example_score_sub.txt 

* validate_pwm.py - script to validate pwm submission 
example run:
python validate_pwm.py --benchmark safe_examples/benchmark_example.json --pwm_sub safe_examples/pwm_submission.txt 

Both scripts write errors and warning to stderr. They return non-zero error code if any error occured
1 - if format error occured (wrong submission format)
2 - if internal error occured (some possible bug in benchmark or unhandled format-error)

* run_bench

Run pwm submission benchmarking:

python cli/run_bench.py --benchmark safe_examples/benchmark_example.json --sub safe_examples/pwm_submission.txt --sub_type pwm

Run aaa submission benchmarking:

python cli/run_bench.py --benchmark safe_examples/benchmark_example.json --sub safe_examples/example_score_sub.txt --sub_type aaa

Script write errors and warning to stderr. They return non-zero error code if any error occured
1 - if format error occured (wrong submission format)
2 - if internal error occured (some possible bug in benchmark or unhandled format-error)

By default, it will write scores to stdout. This behaviour can be changed using --scores_path

python cli/run_bench.py --benchmark safe_examples/benchmark_example.json --sub safe_examples/pwm_submission.txt --sub_type pwm --scores_path out.tsv