from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from enum import Enum
from exceptions import WrongPRAUCTypeException, WrongScorerException

@dataclass
class Scorer(metaclass=ABCMeta):
    name: str
    @abstractmethod
    def score(self, *args, **kwargs) -> float:
        pass

@dataclass
class ConstantScorer(Scorer):
    const: float
    def score(self, *args, **kwargs) -> float:
        return self.const

class BinaryScorer(Scorer):
    @abstractmethod
    def score(self, y_score: List[float], y_real: List[int]) -> float:
        raise NotImplementedError

class SklearnScorer(BinaryScorer):
    pass

class SklearnROCAUC(SklearnScorer):
    def score(self, y_score: List[float], y_real: List[int]):
        y_score_arr = np.array(y_score)
        y_real_arr = np.array(y_real)
        return roc_auc_score(y_true=y_real_arr, y_score=y_score_arr)
    
class SklearnPRAUC(SklearnScorer):
    def score(self, y_score: List[float], y_real: List[int]):
        y_score_arr = np.array(y_score)
        y_real_arr = np.array(y_real)
        return average_precision_score(y_true=y_real_arr, y_score=y_score_arr)

class PRROCScorer(BinaryScorer):
    pass

def import_PRROC():
    '''
    import PRROC package (https://cran.r-project.org/web/packages/PRROC/index.html)
    '''
    from rpy2.robjects.packages import importr, isinstalled
    if not isinstalled("PRROC"):
        utils = importr("utils")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("PRROC", quiet = True, verbose=False)
    pkg = importr("PRROC")
    return pkg

@dataclass
class PRROC_PRAUC(PRROCScorer):
    type: str

    def score(self, y_score: List[float], y_real: List[int]):
        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector
            labels = FloatVector([x for x in y_real])
            scores = FloatVector(y_score)
            if self.type == "integral":
                auroc = pkg.pr_curve(scores, weights_class0=labels, dg_compute=False)
                auroc = auroc[1][0]
            elif self.type == "davisgoadrich":
                auroc = pkg.pr_curve(scores, weights_class0=labels, dg_compute=True)
                auroc = auroc[2][0]
            else:
                raise WrongPRAUCTypeException()
            return auroc

class PRROC_ROCAUC(PRROCScorer):
    def score(self, y_score: List[float], y_real: List[int]):
        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector
            labels = FloatVector(y_real)
            scores = FloatVector(y_score)
            auroc = pkg.roc_curve(scores, weights_class0=labels)
            auroc = auroc[1][0]
            return auroc

@dataclass
class ScorerInfo:
    name: str
    alias: str = ""
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, dt: dict):
        return cls(**dt)

    def __attrs_post_init__(self):
        if not self.alias:
            self.alias = self.name
    
    def make(self):
        if self.name == "scikit_rocauc":
            return SklearnROCAUC(self.alias)
        elif self.name == "scikit_prauc":
            return SklearnPRAUC(self.alias)
        elif self.name == "prroc_rocauc":
            return PRROC_ROCAUC(self.alias)
        elif self.name == "prroc_prauc":
            tp = self.params.get("type")
            if tp is None:
                raise WrongScorerException("type must be specified for scorers from PRROC package")
            tp = tp.lower()
            return PRROC_PRAUC(self.alias, tp)
        elif self.name == "constant_scorer":
            cons = self.params.get("cons")
            if cons is None:
                raise WrongScorerException("cons must be specified for scorers from PRROC package")
            cons = float(cons)
            return ConstantScorer(self.alias, cons)
        raise WrongScorerException(f"Wrong scorer: {self.name}")