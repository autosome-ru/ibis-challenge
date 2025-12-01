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
parser.add_argument("--log_path",
                    default="out.log",
                    type=str)
args = parser.parse_args()


from bibis.benchmark.pwm_submission import PWMSubmission
from bibis.benchmark.benchmarkconfig import BenchmarkConfig
benchmark = Path(args.benchmark_root).expanduser()
templates_dir = Path(args.templates_dir)

print(benchmark)
for stage in ('Leaderboard', 'Final'):
    stage_tfs = list(set(p.name for p in benchmark.glob(f"*/{stage}/valid/*")))

    pwm_config_path = templates_dir / f"{stage}_pwm_pseudocfg.cfg"

    template_cfg = BenchmarkConfig(
        name="PWM_PSEUDO_CFG",
        kind=stage,
        datasets=[],
        scorers=[],
        pwmeval_path="",
        tfs=stage_tfs,
        tags=[],
        metainfo={}    
    )

    template_cfg.save(pwm_config_path)

    pwm_submission_path = templates_dir / f"{stage}_pwm_submission.txt"

    with open(pwm_submission_path, "w") as out:
        for ind, tf in enumerate(stage_tfs):
            for i in range(PWMSubmission.MAX_PWM_PER_TF):
                tag = f"{tf}_motif{i+1}"
                print(f">{tf} {tag}", file=out)
                for i in range(np.random.randint(PWMSubmission.MIN_PWM_LENGTH,
                                             PWMSubmission.MAX_PWM_LENGTH)):
                    a, t, g, c = np.random.dirichlet([1,1,1,1])
                    p = PWMSubmission.MAX_PRECISION
                    print(f"{a:.0{p}f} {t:.0{p}f} {g:.0{p}f} {c:.0{p}f}", file=out)
                print(file=out)