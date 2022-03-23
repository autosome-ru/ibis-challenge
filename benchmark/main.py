from pathlib import Path
from benchmarkconfig import BenchmarkConfig
from model import DictPredictor, RandomPredictor
from pwmeval import PWMEvalPFMPredictor, PWMEvalPWMPredictor 
from prediction import Prediction
import pandas as pd

BENCHMARK_CONFIG = Path("/home_local/dpenzar/ibis-challenge/benchmark/benchmark.json")

if __name__ == '__main__':
    # create benchmark 
    benchmark = BenchmarkConfig\
                .from_json(BENCHMARK_CONFIG)\
                .make_benchmark()
    '''
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
    pfm_path = Path("/home_local/dpenzar/ibis-challenge/benchmark/example.pwm")
    model = PWMEvalPFMPredictor.from_pfm(pfm_path=pfm_path, pwmeval_path=PWMEval_path)
    prediction = model.score(benchmark.datasets[0])
    scores = benchmark.score_prediction(prediction)

    prediction = model.score_batch(benchmark.datasets)
    scores = benchmark.score_prediction(prediction)

    model = PWMEvalPWMPredictor.from_pfm(pfm_path=pfm_path, pwmeval_path=PWMEval_path)
    prediction = model.score(benchmark.datasets[0])
    scores = benchmark.score_prediction(prediction)
    
    # using benchmark to score many models simultaneously
    '''
    #add predictions
    #ideal_prediction_path = Path("/home_local/dpenzar/ideal_predictions.tsv")
    #benchmark.write_ideal_model(ideal_prediction_path)
    #prediction = Prediction.load(ideal_prediction_path)
    #benchmark.add_prediction("ideal", prediction)

    model = RandomPredictor(seed=777)
    prediction = model.score(benchmark.datasets[0])
    benchmark.add_prediction("random_first_only", benchmark.datasets[0].motif, prediction)

    PWMEval_path = Path("/home_local/dpenzar/PWMEval/pwm_scoring")
    pfm_path = Path("/home_local/dpenzar/ibis-challenge/benchmark/example.pwm")
    model = PWMEvalPFMPredictor.from_pfm(pfm_path=pfm_path, pwmeval_path=PWMEval_path)
    submission = model.score_batch(benchmark.datasets)
    benchmark.add_submission("pwm_prediction", submission)
    
    #add models 
    model = PWMEvalPWMPredictor.from_pfm(pfm_path=pfm_path, pwmeval_path=PWMEval_path)
    benchmark.add_model("pwm_add_best", model)
    benchmark.add_pfm("pwm_model2", pfm_path, pwmeval_path=PWMEval_path)

    benchmark.run(n_procs=10)
    