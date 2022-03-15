import json
from prediction import Prediction
from dataset import Dataset, DatasetType
from datasetconfig import DatasetConfig
from pathlib import Path
from attrs import define
from enum import Enum 
from typing import ClassVar, List, Optional
from scorer import ScorerInfo, Scorer
from exceptions import BenchmarkConfigException, WrongBecnhmarkModeException
from utils import register_enum
from attrs import define 
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
    metainfo: dict

    def write_datasets(self, mode: BenchmarkMode):
        if mode is BenchmarkMode.USER:
            self.write_for_user()
        elif mode is BenchmarkMode.ADMIN:
            self.write_for_admin()
        else:
            raise WrongBecnhmarkModeException()
        raise NotImplementedError()

    def write_for_user(self, path: Optional[Path]=None):
        if path is None:
            path = Path.cwd()
        path.mkdir(exist_ok=True)
        for ds in self.datasets:
            if ds.type is DatasetType.TRAIN:
                pass

        raise NotImplementedError()

    def write_for_admin(self):
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
    
    def score_prediction(self, prediction: Prediction):
        model_scores = {}
        for ds in self.datasets:
            labels = []
            scores = []
            for e in ds.entries:
                y_real = e.label
                y_score = prediction[e.tag]
                labels.append(y_real)
                scores.append(y_score)
            ds_scores = {}
            for sc in self.scorers:
                score = sc.score(y_real=labels, y_score=scores)
                ds_scores[sc.name] = score
            model_scores[ds.name] = ds_scores
        return model_scores  

    def score_model(self, model: Model):
        raise NotImplementedError()

    def score_models(self, model_lst: List[Model]):
        raise NotImplementedError()

@define
class BenchmarkConfig:
    name: str
    datasets: List[DatasetConfig]
    scorers: List[ScorerInfo]
    metainfo: dict 
    
    NAME_FIELD: ClassVar[str] = 'name'
    DATASETS_FIELD: ClassVar[str] = 'datasets'
    SCORERS_FIELD: ClassVar[str] = 'scorers'

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
        metainfo = dt.get('metainfo', {})
        for key, value in dt.items():
            if key not in (cls.NAME_FIELD, cls.DATASETS_FIELD, cls.SCORERS_FIELD):
                metainfo[key] = value
        return cls(name, datasets, scorers, metainfo)

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r") as inp:
            dt = json.load(inp)
        return cls.from_dt(dt)

    def make_benchmark(self):
        datasets = [cfg.make_dataset() for cfg in self.datasets]
        scorers = [cfg.make_scorer() for cfg in self.scorers]
        return Benchmark(self.name, datasets, scorers, self.metainfo)