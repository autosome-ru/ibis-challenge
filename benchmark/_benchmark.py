import json

import pandas as pd
from pwm import PWMEvalMode, PWMEvalPredictor
from prediction import Prediction
from dataset import Dataset, DatasetType
from datasetconfig import DatasetConfig
from pathlib import Path
from attr import field, define
from enum import Enum 
from typing import ClassVar, List, Optional, Union
from scorer import ScorerInfo, Scorer
from exceptions import BenchmarkConfigException, BenchmarkException, WrongBecnhmarkModeException
from utils import register_enum
from typing import List
from collections.abc import Sequence
from pathlib import Path
from model import Model 

@register_enum
class BenchmarkMode(Enum):
    USER = 1
    ADMIN = 2

@define
class Benchmark:
    name: str
    datasets: Sequence[Dataset]
    scorers: Sequence[Scorer]
    results_dir: Path
    metainfo: dict
    pwmeval_path: Optional[Path] = None
    models: dict[str, Model] = field(factory=dict)
    predictions: dict[str, Prediction] = field(factory=dict)

    def write_datasets(self, mode: BenchmarkMode):
        if mode is BenchmarkMode.USER:
            self.write_datasets_for_user()
        elif mode is BenchmarkMode.ADMIN:
            self.write_datasets_for_admin()
        else:
            raise WrongBecnhmarkModeException()
        raise NotImplementedError()

    def write_datasets_for_user(self, path: Optional[Path]=None):
        if path is None:
            path = Path.cwd()
        path.mkdir(exist_ok=True)
        for ds in self.datasets:
            if ds.type is DatasetType.TRAIN:
                pass

        raise NotImplementedError()

    def write_datasets_for_admin(self):
        raise NotImplementedError()

    def write_ideal_model(self, path: Path):
        '''
        writes real labels for both train and test set
        in the prediction file format
        '''
        with path.open("w") as out:
            for ds in self.datasets:
                for entry in ds:
                    print(entry.tag, entry.label.value, sep="\t", file=out)  # type: ignore
    
    @staticmethod
    def retrieve_prediction(prediction: Prediction, ds: Dataset):
        labels = []
        scores = []
        skip_ds = False
        for e in ds.entries:
            y_real = e.label
            try:
                y_score = prediction[e.tag]
            except KeyError:
                print(f"No information about entry {e.tag} for {ds.name}. Skipping dataset")
                skip_ds = True
                break
            labels.append(y_real)
            scores.append(y_score)
        if skip_ds:
            return None, None
        else:
            return labels, scores

    def add_pwm(self, 
                pref: str, 
                pwm_path: Union[Path, str], 
                modes: Sequence[PWMEvalMode] =(PWMEvalMode.BEST_HIT, PWMEvalMode.SUM_SCORE),
                pwmeval_path: Optional[Path]=None):
        if pwmeval_path is None:
            if self.pwmeval_path is None:
                raise BenchmarkException("Can't add PWM due to unspecified pwmeval_path")
            pwmeval_path = self.pwmeval_path
        if isinstance(pwm_path, str):
            pwm_path = Path(pwm_path)
        for mode in modes:
            model = PWMEvalPredictor(pwmeval_path=pwmeval_path,
                                     pwm_path=pwm_path, 
                                     mode=mode)
            name = f"{pref}_{mode.value}"
            self.add_model(name, model)
        
    def add_model(self, name: str, model: Model):
        self.models[name] = model

    def add_prediction(self, name: str, pred: Prediction):
        self.predictions[name] = pred

    def score(self, labels, scores):
        ds_scores = {}
        for sc in self.scorers:
            score = sc.score(y_real=labels, y_score=scores)
            ds_scores[sc.name] = score
        return ds_scores

    def score_prediction(self, prediction: Prediction) -> dict[str, dict[str, Union[str, float]]]:
        model_scores = {}
        for ds in self.datasets:
            labels, scores = self.retrieve_prediction(prediction, ds)
            if labels is None:
                ds_scores = {sc.name: "skip" for sc in self.scorers}
            else:
                ds_scores = self.score(labels, scores)
            model_scores[ds.name] = ds_scores
        return model_scores

    def score_model(self, model: Model) -> dict[str, dict[str, Union[str, float]]]:
        scores = {'f':{}}
        return scores

    def get_results_file_path(self, name: str):
        return self.results_dir / f"{name}.tsv"

    def run(self):
        self.results_dir.mkdir(exist_ok=True)
        for name, pred in self.predictions.items():
            scores = self.score_prediction(pred)
            path = self.get_results_file_path(name)
            df = pd.DataFrame(scores)
            df.to_csv(path, sep="\t")

        for name, model in self.models.items():
            scores = self.score_model(model)
            path = self.get_results_file_path(name)
            df = pd.DataFrame(scores)
            df.to_csv(path, sep="\t")


        
@define
class BenchmarkConfig:
    name: str
    datasets: Sequence[DatasetConfig]
    scorers: Sequence[ScorerInfo]
    results_dir: Path
    pwmeval_path: Optional[Path] = None
    metainfo: dict = field(factory=dict)
    
    NAME_FIELD: ClassVar[str] = 'name'
    DATASETS_FIELD: ClassVar[str] = 'datasets'
    SCORERS_FIELD: ClassVar[str] = 'scorers'
    PWMEVAL_PATH_FIELD: ClassVar[str] = "pwmeval"
    RESULTS_DIR_FIELD: ClassVar[str] = "results_dir"

    @classmethod
    def validate_benchmark_dict(cls, dt: dict):
        if not cls.NAME_FIELD in dt:
            raise BenchmarkConfigException(
                    f"Benchmark config must has field '{cls.NAME_FIELD}'")
        if not cls.DATASETS_FIELD in dt:
            raise BenchmarkConfigException("No information about datasets found")
        if not cls.SCORERS_FIELD in dt:
            raise BenchmarkConfigException("No information about scorers found")

    @classmethod
    def from_dt(cls, dt: dict):
        cls.validate_benchmark_dict(dt)
        name = dt[cls.NAME_FIELD]
        datasets = [DatasetConfig.from_dict(rec)\
                        for rec in dt[cls.DATASETS_FIELD]]
        scorers = [ScorerInfo.from_dict(rec)\
                        for rec in dt[cls.SCORERS_FIELD]]
        results_dir = dt.get(cls.RESULTS_DIR_FIELD)
        if results_dir is None:
            results_dir = Path("results")
        elif isinstance(results_dir, str):
            results_dir = Path(results_dir)
        results_dir = results_dir.absolute()
        pwmeval_path = dt.get(cls.PWMEVAL_PATH_FIELD)
        metainfo = dt.get('metainfo', {})
        for key, value in dt.items():
            if key not in (cls.NAME_FIELD, cls.DATASETS_FIELD, cls.SCORERS_FIELD, cls.PWMEVAL_PATH_FIELD, cls.RESULTS_DIR_FIELD):
                metainfo[key] = value
        return cls(name, datasets, scorers, results_dir, pwmeval_path, metainfo)

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r") as inp:
            dt = json.load(inp)
        return cls.from_dt(dt)

    def make_benchmark(self):
        datasets = [cfg.make_dataset() for cfg in self.datasets]
        scorers = [cfg.make_scorer() for cfg in self.scorers]
        return Benchmark(self.name, datasets, scorers, self.results_dir, self.metainfo, self.pwmeval_path)