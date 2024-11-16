# This repository provides the benchmarking suite <br/> of *the IBIS Challenge*
Codebook/GRECO-BIT open challenge in Inferring Binding Specificities <br/>of human transcription factors from multiple experimental data types

![IBIS logo](https://github.com/user-attachments/assets/212f4edd-131d-4023-989a-a828e5e9dff4)

# Detailed challenge documentation and accompanying data

The detailed documentation of the IBIS challenge is available at the [IBIS](https://ibis.autosome.org) website.

The complete IBIS data package including train, test, and benchmarking-ready data are available at [ZENODO](https://).

# Project structure

* bibis - main package, benchmarking tools, data sampling methods, etc
* cli - command-line-wrappers for various tools, including an offline validator for IBIS submissions

* To satisfy curious participants, along with the benchmarking protocols, here we also provide the software implementation of the train-test data preparation.

# External dependencies

For PWM scanning, the IBIS benchmarking suite relies on PWMEval (https://github.com/gio31415/PWMEval). 
We have included a precompiled PWMEval binary executable for x86-64 Linux distributions in the `cli/` folder. 

If you are using a different system or architecture, please follow these steps to be able to run PWM scoring:
- Download the PWMEval source code from https://github.com/gio31415/PWMEval and navigate to the downloaded PWMEval repository folder.
- Compile PWMEval (`pwm_scoring.c`) using the following command: `make -f Makefile` in the source code directory of PWMEval.
- Update the path to PWMEval in benchmark_example.json (more details in the section "[Gathering benchmark files](#gathering-benchmark-files)").

The most up-to-date version of PWMEval is included in the PWMScan package: https://gitlab.sib.swiss/EPD/pwmscan

Note that the PWMEval way of handling Ns in nucleotide sequences is not fully predictable and we explicitly avoided such sequences in the IBIS data.

# Offline validation scripts

These scripts were intended to be used during the challenge to allow participants to ensure that their files are technically valid.
The respective scripts are designed to be used on a Linux machine. The scripts are applicable to both the Leaderboard and Final stages of IBIS,
although the `leaderboard` should be replaced with `final` in the respective paths.

Start by running: 
```console
git clone https://github.com/autosome-ru/ibis-challenge
cd ibis-challenge
```

Then create and activate a conda environment:
```console
conda env create -f environment.yml
conda activate bibis
```

Now install bibis package required for benchmark scripts
```
pip install -e .
```

The json 'minimal' configuration files and submission examples for offline validation are downloadable from the ZENODO repo.
The respective archive file should be unpacked and placed in the same folder.

Finally, you can validate the AAA leaderboard submissions:
```console
python cli/validate_aaa.py --benchmark leaderboard_examples/${EXP_TYPE}_benchmark.json --aaa_sub leaderboard_examples/example_${EXP_TYPE}_sub.tsv
```

where ```EXP_TYPE``` can be SMS, PBM, CHS, GHTS, or HTS.

For PWMs, validation against a single json file covering all benchmarks and TFs is sufficient.

```console
python cli/validate_pwm.py --benchmark leaderboard_examples/example_PWM_benchmark.json --pwm_sub leaderboard_examples/pwm_submission.txt
```

## Errors and Warrnings

The validation scripts report errors and warnings to stderr. 
They return a non-zero error code in case of errors:

1 - known format error: misformatted submission file

2 - internal errors: bugs in the validation code or unhandled formatting errors

# IBIS benchmarking scripts

The benchmarking data labels were hidden during the IBIS challenge but are now accessible in the dedicated ZENODO repository (see above).
The benchmarking zip-file provided on ZENODO includes both the contents of this repository AND the necessary configuration files
and labeled test data to estimate the performance metrics.

## Unpacking and benchmark files

First, please set up the conda environment as for the offline validation scripts, see above.
Next, run the following command to fix benchmark configs if you have x86-64 Linux:

```console
bash cli/format_bench.sh ${PATH_TO_BENCHMARK_DIR} cli/pwm_scoring
```

Otherwise, if you use a different system or a custom-compiled PWMEval, check  the "[External dependencies](#external-dependencies)" section, and  replace the default path with your custom path to the PWMEval executable: 
```console
bash cli/format_bench.sh ${PATH_TO_BENCHMARK_DIR} ${PATH_TO_PWMEval}/pwm_scoring
```
`PATH_TO_BENCHMARK_DIR` should point to the folder where you unpacked the benchmarking suite archive from ZENODO.
Particularly, it must contain the `BENCHMARK_CONFIGS` and `BENCHMARK_PROCESSED` folders, which contain the benchmark configuration files and the labeled test data.

## How to run

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
