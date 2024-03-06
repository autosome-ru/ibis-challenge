import argparse
import sys
import numpy as np

from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--benchmark_root", 
                    type=str,
                    required=True)
parser.add_argument("--templates_dir", 
                    type=str,
                    required=True)
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
args = parser.parse_args()

sys.path.append(args.bibis_root)

from bibis.benchmark.pwm_submission import PWMSubmission
benchmark = Path(args.benchmark_root).expanduser()
templates_dir = Path(args.templates_dir)

print(benchmark)
for stage in ('Leaderboard', 'Final'):
    stage_tfs = list(set(p.name for p in benchmark.glob("*/Leaderboard/valid/*")))

    pwm_submission_path = templates_dir / f"{stage}_pwm_submission.txt"

    with open(pwm_submission_path, "w") as out:
        for ind, tf in enumerate(stage_tfs):
            for i in range(PWMSubmission.MAX_PWM_PER_TF):
                tag = f"{tf}_motif{i+1}"
                print(f">{tf} {tag}", file=out)
                for i in range(np.random.randint(5, 31)):
                    a, t, g, c = np.random.dirichlet([1,1,1,1])
                    p = PWMSubmission.MAX_PRECISION
                    print(f"{a:.0{p}f} {t:.0{p}f} {g:.0{p}f} {c:.0{p}f}", file=out)
                print(file=out)