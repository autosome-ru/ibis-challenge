from abc import ABCMeta, abstractmethod
from attrs import define
from utils import auto_convert
from labels import BinaryLabel
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

class Scorer:
    pass

class BinaryScorer(Scorer, metaclass=ABCMeta):
    @abstractmethod
    def score(self, y_score: List[float], y_real: List[BinaryLabel]) -> float:
        raise NotImplementedError

class SklearnScorer(BinaryScorer):
    pass

class SklearnROCAUC(SklearnScorer):
    def score(self, y_score: List[float], y_real: List[BinaryLabel]):
        y_score_arr = np.array(y_score)
        y_real_arr = np.array([y.value for y in y_real])
        return roc_auc_score(y_true=y_real_arr, y_score=y_score_arr)
    
class SklearnPRAUC(SklearnScorer):
    def score(self, y_score: List[float], y_real: List[BinaryLabel]):
        y_score_arr = np.array(y_score)
        y_real_arr = np.array([y.value for y in y_real])
        return average_precision_score(y_true=y_real_arr, y_score=y_score_arr)



@define(field_transformer=auto_convert)
class ScorerInfo:
    name: str
    alias: str = ""
    params: dict = {}

    @classmethod
    def from_dict(cls, dt: dict):
        return cls(**dt)

    def __attrs_post_init__(self):
        if not self.alias:
            self.alias = self.name
    
    def make_scorer(self):
        return Scorer()

