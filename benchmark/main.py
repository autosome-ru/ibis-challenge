
from pathlib import Path
from _benchmark import BenchmarkConfig
from prediction import Prediction
import pandas as pd

BENCHMARK_CONFIG = Path("/home_local/dpenzar/ibis-challenge/benchmark/benchmark.json")

if __name__ == '__main__':
    benchmark = BenchmarkConfig\
                .from_json(BENCHMARK_CONFIG)\
                .make_benchmark()
    print(benchmark)
    ideal_prediction_path = Path("/home_local/dpenzar/ideal_predictions.tsv")
    benchmark.write_ideal_model(ideal_prediction_path)
    prediction = Prediction.load(ideal_prediction_path)
    scores = benchmark.score_prediction(prediction)
    pd.DataFrame(scores).to_csv("/home_local/dpenzar/resulst.tsv", sep="\t")