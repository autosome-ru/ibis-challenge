from dataset import Dataset
from examples import PBM_EXAMPLE_PATH
from pbm import PBMExperiment
from pathlib import Path
from attrs import define 
from typing import List
from model import Model
from scorer import Scorer
from exceptions import WrongBecnhmarkModeException
from benchmark import BenchmarkMode
from pathlib import Path

@define
class Benchmark:
    datasets: List[Dataset]
    scorers: List[Scorer]

    @classmethod
    def from_config(cls):
        raise NotImplementedError()

    def write_datasets(self, mode: BenchmarkMode):
        if mode is BenchmarkMode.USER:
            self.write_for_user()
        elif mode is BenchmarkMode.ADMIN:
            self.write_for_admin()
        else:
            raise WrongBecnhmarkModeException()
        raise NotImplementedError()

    def write_for_user(self):
        raise NotImplementedError()

    def write_for_admin(self):
        raise NotImplementedError()

    def score_prediction(self, prediction):
        raise NotImplementedError()

    def score_model(self, model: Model):
        raise NotImplementedError()


BENCHMARK_CONFIG = Path("/home_local/dpenzar/ibis-challenge/benchmark/benchmark.json")