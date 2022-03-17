from pathlib import Path
from _benchmark import BenchmarkConfig
from model import RandomPredictor
from pwm import PWMEvalPredictor, PWMEvalMode 
from prediction import Prediction
import pandas as pd

BENCHMARK_CONFIG = Path("/home_local/dpenzar/ibis-challenge/benchmark/benchmark.json")

if __name__ == '__main__':
    # create benchmark 
    benchmark = BenchmarkConfig\
                .from_json(BENCHMARK_CONFIG)\
                .make_benchmark()

    # ideal predictions for this benchmark 
    ideal_prediction_path = Path("/home_local/dpenzar/ideal_predictions.tsv")
    benchmark.write_ideal_model(ideal_prediction_path)
    prediction = Prediction.load(ideal_prediction_path)
    scores = benchmark.score_prediction(prediction)
    pd.DataFrame(scores).to_csv("/home_local/dpenzar/results_ideal.tsv", sep="\t")

    # random model
    model = RandomPredictor(seed=777)
    prediction = model.score(benchmark.datasets[0])
    scores = benchmark.score_prediction(prediction)
    
    # scoring PWM 
    PWMEval_path = Path("/home_local/dpenzar/PWMEval/pwm_scoring")
    pwm_path = Path("/home_local/dpenzar/ibis-challenge/benchmark/example.pwm")
    model = PWMEvalPredictor(PWMEval_path, pwm_path, PWMEvalMode.BEST_HIT)
    prediction = model.score(benchmark.datasets[0])
    scores = benchmark.score_prediction(prediction)

    prediction = model.score_batch(benchmark.datasets)
    scores = benchmark.score_prediction(prediction)

    model = PWMEvalPredictor(PWMEval_path, pwm_path, PWMEvalMode.SUM_SCORE)
    prediction = model.score(benchmark.datasets[0])
    scores = benchmark.score_prediction(prediction)
    
    # using benchmark to score many models simultaneously

    #add predictions
    prediction = Prediction.load(ideal_prediction_path)
    benchmark.add_prediction("ideal", prediction)

    model = RandomPredictor(seed=777)
    prediction = model.score(benchmark.datasets[0])
    benchmark.add_prediction("random_first_only", prediction)

    model = PWMEvalPredictor(PWMEval_path, pwm_path, PWMEvalMode.BEST_HIT) 
    prediction = model.score_batch(benchmark.datasets)
    benchmark.add_prediction("pwm_prediction", prediction)

    #add models 
    model = PWMEvalPredictor(PWMEval_path, pwm_path, PWMEvalMode.BEST_HIT) 
    benchmark.add_model("pwm_add_best", model)

    model = PWMEvalPredictor(PWMEval_path, pwm_path, PWMEvalMode.BEST_HIT) 
    benchmark.add_pwm("pwm_model", pwm_path)
 
    benchmark.add_pwm("pwm_model2", pwm_path, pwmeval_path=PWMEval_path)

    benchmark.run()