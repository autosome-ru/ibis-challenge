# Project structure

* bibis - main package, benchmarking tools, data sampling methods, etc
* cli - command-line-wrappers for various tools, including an offline validator for IBIS submissions

# Offline validation scripts

Offline validator scripts for PWM and AAA predictions are available in the ibis-challenge GitHub repository. The respective scripts can be used on a Linux machine. 

Start by running: 
```console
git clone https://github.com/autosome-ru/ibis-challenge
```

Then create and activate a conda environment:
```console
conda create --name ibis --file spec-file.txt
conda activate ibis
```

The json benchmark configuration files and examples for offline validation are downloadable at the IBIS challenge website.
The respective archive file should be unpacked to the same folder.

Finally, you can validate your AAA leaderboard submissions:
```console
python cli/validate_aaa.py --benchmark leaderboard_examples/{EXP_TYPE}_benchmark.json --aaa_sub leaderboard_examples/example_{EXP_TYPE}_sub.tsv 
```

```{EXP_TYPE}``` can be SMS, PBM, CHS, GHTS, or HTS.

For PWMs, validation against a single json covering all benchmarks and TFs is sufficient.

```console
python validate_pwm.py --benchmark leaderboard_examples/example_PWM_benchmark.json --pwm_sub leaderboard_examples/pwm_submission.txt
```

To satisfy curious participants, the software implementation of the train-test data preparation and benchmarking protocols are available on GitHub in the same repo.

# Final

The data for the final evaluation will be organized in the same fashion as the data for the Leaderboard. You will be able to use the same validation scripts, and the necessary configuration files will be provided along with the test data.

## Errors and Wardning

The validation scripts report errors and warnings to stderr. 
They return a non-zero error code in case of errors:

1 - known format error: misformatted submission file

2 - internal errors: bugs in the validation code or unhandled formatting errors

# IBIS Benchmarking (for information purposes)

## ! Note, that benchmarking data is hidden during the IBIS challenge hence you won't be able to fully replicate the analysis until the challenge ends (i.e. the command line examples won't work w/o complete test data).

The main benchmarking script is ```run_bench.py```. Examples of correct json configuration files are available at the IBIS challenge website.

Run PWM submission benchmarking:

```console
python cli/run_bench.py --benchmark safe_examples/benchmark_example.json --sub safe_examples/pwm_submission.txt --sub_type pwm
```

Run AAA submission benchmarking:

```console
python cli/run_bench.py --benchmark safe_examples/benchmark_example.json --sub safe_examples/example_score_sub.txt --sub_type aaa
```

The benchmarking script writes errors and warnings to stderr, and returns non-zero error codes if any error occurs:
1 - format error (wrong submission format);
2 - internal error occurred (possible bug or unhandled format error).

By default, the benchmarking scores are written to stdout. 
This behavior can be changed using the --scores_path parameter.

```console
python cli/run_bench.py --benchmark safe_examples/benchmark_example.json --sub safe_examples/pwm_submission.txt --sub_type pwm --scores_path out.tsv
```

# External dependencies

To be able to run PWM scoring you need to download and compile PWMEval (pwm_scoring.c) from https://github.com/gio31415/PWMEval
and update the path to PWMEval in benchmark_example.json

The most up-to-date version of PWMEval is included in the PWMScan package: https://gitlab.sib.swiss/EPD/pwmscan

Note that the PWMEval way of handling Ns in nucleotide sequences is not fully predictable and we explicitly avoided such sequences in the IBIS data.
