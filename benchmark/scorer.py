from abc import ABCMeta, abstractmethod
from tabnanny import verbose
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
        



if __name__ == "__main__":
    scorer = SklearnROCAUC()
    s = scorer.score([0.] * 5 + [1.] * 5, [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5)
    print(s)
    s = scorer.score([1.] * 5 + [0.] * 5, [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5)
    print(s)
    scorer = SklearnPRAUC()
    s = scorer.score([0.] * 5 + [1.] * 5, [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5)
    print(s)
    s = scorer.score([1.] * 5 + [0.] * 5, [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5)
    print(s)
    pkg = import_PRROC()
    from rpy2.robjects.vectors import FloatVector
    labels =  [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5
    labels = FloatVector([x.value for x in labels])
    scores = FloatVector([0.] * 5 + [1.] * 5)
    auroc = pkg.roc_curve(scores, weights_class0=labels)
    print(auroc[1])
    labels =  [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5
    labels = FloatVector([x.value for x in labels])
    scores = FloatVector([1.] * 5 + [0.] * 5)
    auroc = pkg.roc_curve(scores, weights_class0=labels)
    print(auroc[1])
    labels =  [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5
    labels = FloatVector([x.value for x in labels])
    scores = FloatVector([0.] * 5 + [1.] * 5)
    prroc = pkg.pr_curve(scores, weights_class0=labels)
    print(float(prroc[1][0]))
    labels =  [BinaryLabel.NEGATIVE] * 5 + [BinaryLabel.POSITIVE] * 5
    labels = FloatVector([x.value for x in labels])
    scores = FloatVector([1.] * 5 + [0.] * 5)
    prroc = pkg.pr_curve(scores, weights_class0=labels)
    print(float(prroc[1][0]))
    
    


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

