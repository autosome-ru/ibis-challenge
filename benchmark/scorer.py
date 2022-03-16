from abc import ABCMeta, abstractmethod
from attrs import define
from utils import auto_convert, register_enum
from labels import BinaryLabel
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from enum import Enum
from exceptions import WrongPRAUCTypeException, WrongScorerException

@define
class Scorer(metaclass=ABCMeta):
    name: str
    @abstractmethod
    def score(self, *args, **kwargs) -> float:
        pass

@define
class ConstantScorer(Scorer):
    const: float
    def score(self, *args, **kwargs) -> float:
        return self.const

class BinaryScorer(Scorer):
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

@register_enum
class PRAUC_TYPE(Enum):
    INTEGRAL = "integral"
    DG = "davisgoadrich"

@define
class PRROC_PRAUC(PRROCScorer):
    type: PRAUC_TYPE
    def score(self, y_score: List[float], y_real: List[BinaryLabel]):
        pkg = import_PRROC()
        from rpy2.robjects.vectors import FloatVector
        labels = FloatVector([x.value for x in y_real])
        scores = FloatVector(y_score)
        if self.type is PRAUC_TYPE.INTEGRAL:
            auroc = pkg.pr_curve(scores, weights_class0=labels, dg_compute=False)
            auroc = auroc[1][0]
        elif self.type is PRAUC_TYPE.DG:
            auroc = pkg.pr_curve(scores, weights_class0=labels, dg_compute=True)
            auroc = auroc[2][0]
        else:
            raise WrongPRAUCTypeException()
        return auroc

class PRROC_ROCAUC(PRROCScorer):
    def score(self, y_score: List[float], y_real: List[BinaryLabel]):
        pkg = import_PRROC()
        from rpy2.robjects.vectors import FloatVector
        labels = FloatVector([x.value for x in y_real])
        scores = FloatVector(y_score)
        auroc = pkg.roc_curve(scores, weights_class0=labels)
        auroc = auroc[1][0]
        return auroc

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
            tp = PRAUC_TYPE(tp)
            return PRROC_PRAUC(self.alias, tp)
        elif self.name == "constant_scorer":
            cons = self.params.get("cons")
            if cons is None:
                raise WrongScorerException("cons must be specified for scorers from PRROC package")
            cons = float(cons)
            return ConstantScorer(self.alias, cons)
        raise WrongScorerException(f"Wrong scorer: {self.name}")


if __name__ == "__main__":
    scorer = SklearnROCAUC("example")
    s = scorer.score([0.] * 5 + [1.] * 5, [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5)
    print(s)
    s = scorer.score([1.] * 5 + [0.] * 5, [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5)
    print(s)
    scorer = SklearnPRAUC("example")
    s = scorer.score([0.] * 5 + [1.] * 5, [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5)
    print(s)
    s = scorer.score([1.] * 5 + [0.] * 5, [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5)
    print(s)
    scorer = PRROC_PRAUC("example", PRAUC_TYPE.INTEGRAL)
    labels =  [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5
    scores = [0.] * 5 + [1.] * 5
    s = scorer.score(scores, labels)
    print(s)
    labels =  [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5
    scores = [1.] * 5 + [0.] * 5
    s = scorer.score(scores, labels)
    print(s)
    scorer = PRROC_PRAUC("example", PRAUC_TYPE.DG)
    labels =  [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5
    scores = [1.] * 5 + [0.] * 5
    s = scorer.score(scores, labels)
    print(s)
    labels = [BinaryLabel.NEGATIVE] * 100000 + [BinaryLabel.POSITIVE] * 100000
    import random
    scores = [random.random() for _ in labels]
    s = scorer.score(scores, labels)
    print(s)
    scorer = PRROC_ROCAUC("example")
    labels =  [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5
    scores = [1.] * 5 + [0.] * 5
    s = scorer.score(scores, labels)
    print(s)
    print(PRAUC_TYPE("integral"))