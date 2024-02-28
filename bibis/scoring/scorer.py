from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from scipy.stats import kendalltau

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
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float:
        raise NotImplementedError

class RegressionScorer(Scorer):
    @abstractmethod
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float:
        raise NotImplementedError

class SklearnScorer(BinaryScorer):
    pass

class SklearnROCAUC(SklearnScorer):
    def score(self, y_score: np.ndarray[float], y_real:  np.ndarray[float]) -> float:
        return float(roc_auc_score(y_true=y_real, y_score=y_score))
    
class SklearnPRAUC(SklearnScorer):
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float:
        return float(average_precision_score(y_true=y_real, y_score=y_score))

def import_stats():
    '''
    import stats R package 
    '''
    from rpy2.robjects.packages import importr, isinstalled
    pkg = importr("stats")
    return pkg

class KendallRank(RegressionScorer):
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float: 
        from rpy2.rinterface_lib import openrlib
        from rpy2.robjects.vectors import FloatVector

        mask = np.logical_not(np.isclose(y_real, 0))
        y_score = y_score[mask]
        y_real = y_real[mask]
        with openrlib.rlock:
            y_score = FloatVector(y_score)
            y_real = FloatVector(y_real)
            stats = import_stats()
            corr = stats.cor(y_score, y_real)[0]
        return corr

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

class PRROCScorer(BinaryScorer):
    pass    

@dataclass
class PRROC_PRAUC(PRROCScorer):
    type: str

    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float:
        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector
            labels = FloatVector(y_real)
            scores = FloatVector(y_score)

            if self.type == "integral":
                auroc = pkg.pr_curve(scores, weights_class0=labels, dg_compute=False, sorted=True)
                auroc = auroc[1][0]
            elif self.type == "davisgoadrich":
                auroc = pkg.pr_curve(scores, weights_class0=labels, dg_compute=True, sorted=True)
                auroc = auroc[2][0]
            else:
                raise Exception()
            return auroc

@dataclass
class PRROC_PRAUC_HTSELEX(PRROCScorer):
    type: str
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float:
        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector

            y_real = np.where(np.isclose(y_real, 0), 0, 1) # convert to binary task

            labels = FloatVector(y_real)
            scores = FloatVector(y_score)
            if self.type == "integral":
                auroc = pkg.pr_curve(scores, weights_class0=labels, dg_compute=False, sorted=True)
                auroc = auroc[1][0]
            elif self.type == "davisgoadrich":
                auroc = pkg.pr_curve(scores, weights_class0=labels, dg_compute=True, sorted=True)
                auroc = auroc[2][0]
            else:
                raise Exception()
            return auroc

@dataclass 
class PRROC_ROCAUC(PRROCScorer):
    def score(self, y_score:  np.ndarray[float], y_real:  np.ndarray[float]) -> float:
        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector
            labels = FloatVector(y_real)
            scores = FloatVector(y_score)
            auroc = pkg.roc_curve(scores, weights_class0=labels, sorted=True)
            auroc = auroc[1][0]
            return auroc

@dataclass 
class PRROC_ROCAUC_HTSELEX(PRROCScorer):
    def score(self, y_score:  np.ndarray[float], y_real:  np.ndarray[float]) -> float:
        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector
            y_real = np.where(np.isclose(y_real, 0), 0, 1) # convert to binary task
            labels = FloatVector(y_real)
            scores = FloatVector(y_score)
            auroc = pkg.roc_curve(scores, weights_class0=labels, sorted=True)
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
        elif self.name == "prroc_rocauc_htselex":
            return PRROC_ROCAUC_HTSELEX(self.alias)
        elif self.name == "prroc_prauc":
            tp = self.params.get("type")
            if tp is None:
                raise Exception("type must be specified for prauc scorer from PRROC package")
            tp = tp.lower()
            return PRROC_PRAUC(self.alias, tp)
        elif self.name == "prroc_prauc_htselex":
            tp = self.params.get("type")
            if tp is None:
                raise Exception("type must be specified for prauc scorer from PRROC package")
            tp = tp.lower()
            return PRROC_PRAUC_HTSELEX(self.alias, tp)
        elif self.name == "kendalltau":
            return KendallRank(self.alias)
        elif self.name == "constant_scorer":
            cons = self.params.get("cons")
            if cons is None:
                raise Exception("cons must be specified for constant scorer")
            cons = float(cons)
            return ConstantScorer(self.alias, cons)
        raise Exception(f"Wrong scorer: {self.name}")
    
    def to_dict(self) -> dict:
        dt = {}
        dt['name'] = self.name
        dt['alias'] = self.alias
        dt['params'] = self.params
        return dt