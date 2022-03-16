from pathlib import Path
from _benchmark import BenchmarkConfig
from pwm import PWMEvalPredictor, PWMMode 
from prediction import Prediction
import pandas as pd

BENCHMARK_CONFIG = Path("/home_local/dpenzar/ibis-challenge/benchmark/benchmark.json")

if __name__ == '__main__':
    benchmark = BenchmarkConfig\
                .from_json(BENCHMARK_CONFIG)\
                .make_benchmark()
    ideal_prediction_path = Path("/home_local/dpenzar/ideal_predictions.tsv")
    benchmark.write_ideal_model(ideal_prediction_path)
    prediction = Prediction.load(ideal_prediction_path)
    scores = benchmark.score_prediction(prediction)
    pd.DataFrame(scores).to_csv("/home_local/dpenzar/results_ideal.tsv", sep="\t")

    PWMEval_path = Path("/home_local/dpenzar/PWMEval/pwm_scoring")
    pwm_path = Path("/home_local/dpenzar/ibis-challenge/benchmark/example.pwm")
    model = PWMEvalPredictor(PWMEval_path, pwm_path, PWMMode.BEST_HIT)
    prediction = model.score(benchmark.datasets[0:1])
    scores = benchmark.score_prediction(prediction)
    pd.DataFrame(scores).to_csv("/home_local/dpenzar/results_motif_part.tsv", sep="\t")

    prediction = model.score(benchmark.datasets[0:2])
    scores = benchmark.score_prediction(prediction)
    pd.DataFrame(scores).to_csv("/home_local/dpenzar/results_motif.tsv", sep="\t")