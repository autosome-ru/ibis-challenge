from abc import ABCMeta, abstractmethod
from curses import meta
from dataclasses import dataclass, field, asdict
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from scipy.stats import kendalltau
import pandas as pd 

@dataclass
class ScorerResult:
    value: float 
    metainfo: dict | None = None # any json-compatible dict with metainfo  

@dataclass
class Scorer(metaclass=ABCMeta):
    name: str
    @abstractmethod
    def score(self, *args, **kwargs) -> ScorerResult:
        pass

@dataclass
class ConstantScorer(Scorer):
    const: float
    def score(self, *args, **kwargs) -> float:
        return ScorerResult(value=self.const)

class BinaryScorer(Scorer):
    @abstractmethod
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float], **kwargs) -> float:
        raise NotImplementedError

class RegressionScorer(Scorer):
    @abstractmethod
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float], **kwargs) -> float:
        raise NotImplementedError

class SklearnScorer(BinaryScorer):
    pass

class SklearnROCAUC(SklearnScorer):
    def score(self, y_score: np.ndarray[float], y_real:  np.ndarray[float], **kwargs) -> float:
        val = float(roc_auc_score(y_true=y_real, y_score=y_score))
        return ScorerResult(value=val)
    
class SklearnPRAUC(SklearnScorer):
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float], **kwargs) -> float:
        val = float(average_precision_score(y_true=y_real, y_score=y_score))
        return ScorerResult(value=val)


class KendallRank(RegressionScorer):
    def score(self, 
              y_score: np.ndarray[float], 
              y_real: np.ndarray[float], 
              y_group: np.ndarray[float] | None = None, 
              **kwargs) -> float: 
      
        if y_group is None:
            val =  self._calc(y_score=y_score, 
                              y_real=y_real)
            metainfo = None
        else:
            nonanmask = ~np.isnan(y_group)
            y_score = y_score[nonanmask]
            y_real = y_real[nonanmask]
            y_group = y_group[nonanmask]
            groups = set(y_group)
            scores = {}
            for g in groups:
                gr_mask = y_group == g
                scores[g] = self._calc(y_score=y_score[gr_mask],
                                       y_real=y_real[gr_mask])
            #print(scores, file=sys.stderr)
            val =  sum(scores.values()) / len(groups)
            metainfo = scores
        return ScorerResult(value=val, metainfo=metainfo)
            
    def _calc(self,
              y_score: np.ndarray[float], 
              y_real: np.ndarray[float]) -> float:
        mask = np.logical_not(np.isclose(y_real, 0))
        y_score = y_score[mask]
        y_real = y_real[mask]
        cor =  kendalltau(y_score, y_real).correlation
        if pd.isnull(cor):
            if len(set(y_score)) == 1 or len(set(y_real)) == 1 :
                return 0
            else:
                raise Exception("Unknown bug with correlation calculation occured")
        return cor

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
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float], **kwargs) -> float:
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
        return ScorerResult(value=auroc)
        
@dataclass
class PRROC_PRAUC_TOP25(PRROCScorer):
    type: str
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float], **kwargs) -> float:
        from rpy2.rinterface_lib import openrlib
        neg_mask = np.isclose(y_real, 0) # don't change unless rewriting HTSELEX PRAUC TOP25
        negatives = y_score[neg_mask]
        quart, mod = divmod(negatives.shape[0], 4)
        if mod != 0:
            quart += 1
        negatives = negatives[-quart:] 

        positives = y_score[~neg_mask] 
        quart, mod = divmod(positives.shape[0], 4)
        if mod != 0:
            quart += 1
        positives = positives[-quart:] 

        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector

            if self.type == "integral":
                auroc = pkg.pr_curve(scores_class0=FloatVector(positives), 
                                     scores_class1=FloatVector(negatives),
                                     dg_compute=False, 
                                     sorted=True)
                auroc = auroc[1][0]
            elif self.type == "davisgoadrich":
                auroc = pkg.pr_curve(scores_class0=FloatVector(positives), 
                                     scores_class1=FloatVector(negatives),
                                     dg_compute=True,
                                     sorted=True)
                auroc = auroc[2][0]
            else:
                raise Exception()
        return ScorerResult(value=auroc)

@dataclass
class PRROC_PRAUC_TOP50(PRROCScorer):
    type: str
    def score(self, y_score: np.ndarray[float], y_real: np.ndarray[float], **kwargs) -> float:
        from rpy2.rinterface_lib import openrlib
        neg_mask = np.isclose(y_real, 0) # don't change unless rewriting HTSELEX PRAUC TOP25
        negatives = y_score[neg_mask]
        quart, mod = divmod(negatives.shape[0], 2)
        if mod != 0:
            quart += 1
        negatives = negatives[-quart:] 

        positives = y_score[~neg_mask] 
        quart, mod = divmod(positives.shape[0], 2)
        if mod != 0:
            quart += 1
        positives = positives[-quart:] 

        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector

            if self.type == "integral":
                auroc = pkg.pr_curve(scores_class0=FloatVector(positives), 
                                     scores_class1=FloatVector(negatives),
                                     dg_compute=False, 
                                     sorted=True)
                auroc = auroc[1][0]
            elif self.type == "davisgoadrich":
                auroc = pkg.pr_curve(scores_class0=FloatVector(positives), 
                                     scores_class1=FloatVector(negatives),
                                     dg_compute=True,
                                     sorted=True)
                auroc = auroc[2][0]
            else:
                raise Exception()
        return ScorerResult(value=auroc)

@dataclass 
class PRROC_PRAUC_AVERAGED(PRROCScorer):
    type: str
    inner_scorers: dict[str, PRROCScorer] = field(init=False)

    def __post_init__(self):
        self.inner_scorers = {
            'full': PRROC_PRAUC(name=self.name, type=self.type),
            'top50': PRROC_PRAUC_TOP50(name=self.name, type=self.type),
            'top25': PRROC_PRAUC_TOP25(name=self.name, type=self.type)
        }

    def score(self, y_score: np.ndarray[float], y_real:  np.ndarray[float], **kwargs) -> float:
        inner_scores = {name: scorer.score(y_score=y_score, y_real=y_real) for\
                            name, scorer in self.inner_scorers.items()}
        score = sum(inner_scores.values) / len(inner_scores)
        
        return ScorerResult(value=score,
                            metainfo=inner_scores)


@dataclass 
class PRROC_ROCAUC(PRROCScorer):
    def score(self, y_score:  np.ndarray[float], y_real:  np.ndarray[float], **kwargs) -> float:
        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector
            labels = FloatVector(y_real)
            scores = FloatVector(y_score)
            auroc = pkg.roc_curve(scores, weights_class0=labels, sorted=True)
            auroc = auroc[1][0]
        return ScorerResult(value=auroc)

@dataclass 
class PRROC_ROCAUC(PRROCScorer):
    def score(self, y_score:  np.ndarray[float], y_real:  np.ndarray[float], **kwargs) -> float:
        from rpy2.rinterface_lib import openrlib
        with openrlib.rlock:
            pkg = import_PRROC()
            from rpy2.robjects.vectors import FloatVector
            y_real = np.where(np.isclose(y_real, 0), 0, 1) # convert to binary task
            labels = FloatVector(y_real)
            scores = FloatVector(y_score)
            auroc = pkg.roc_curve(scores, weights_class0=labels, sorted=True)
            auroc = auroc[1][0]
        return ScorerResult(value=auroc)    

@dataclass 
class PRROC_ROCAUC_TOP25(PRROCScorer):
    def score(self, y_score:  np.ndarray[float], y_real:  np.ndarray[float], **kwargs) -> float:
        from rpy2.rinterface_lib import openrlib

        neg_mask = np.isclose(y_real, 0) # don't change unless rewriting HTSELEX ROCAUC TOP25
        negatives = y_score[neg_mask]
        quart, mod = divmod(negatives.shape[0], 4)
        if mod != 0:
            quart += 1
        negatives = negatives[-quart:] 

        positives = y_score[~neg_mask] 
        quart, mod = divmod(positives.shape[0], 4)
        if mod != 0:
            quart += 1
        positives = positives[-quart:] 

        with openrlib.rlock:
            pkg = import_PRROC()

            from rpy2.robjects.vectors import FloatVector
            auroc = pkg.roc_curve(scores_class0=FloatVector(positives), 
                                  scores_class1=FloatVector(negatives), 
                                  sorted=True)
            auroc = auroc[1][0]
        return ScorerResult(value=auroc)

@dataclass 
class PRROC_ROCAUC_TOP50(PRROCScorer):
    def score(self, y_score:  np.ndarray[float], y_real:  np.ndarray[float], **kwargs) -> float:
        from rpy2.rinterface_lib import openrlib

        neg_mask = np.isclose(y_real, 0) # don't change unless rewriting HTSELEX ROCAUC TOP25
        negatives = y_score[neg_mask]
        quart, mod = divmod(negatives.shape[0], 2)
        if mod != 0:
            quart += 1
        negatives = negatives[-quart:] 

        positives = y_score[~neg_mask] 
        quart, mod = divmod(positives.shape[0], 2)
        if mod != 0:
            quart += 1
        positives = positives[-quart:] 

        with openrlib.rlock:
            pkg = import_PRROC()

            from rpy2.robjects.vectors import FloatVector
            auroc = pkg.roc_curve(scores_class0=FloatVector(positives), 
                                  scores_class1=FloatVector(negatives), 
                                  sorted=True)
            auroc = auroc[1][0]
        return ScorerResult(value=auroc)
    
@dataclass 
class PRROC_ROCAUC_AVERAGED(PRROCScorer):
    inner_scorers: dict[str, PRROCScorer] = field(init=False)

    def __post_init__(self):
        self.inner_scorers = {
            'full': PRROC_ROCAUC(self.name),
            'top50': PRROC_ROCAUC_TOP50(self.name),
            'top25': PRROC_ROCAUC_TOP25(self.name)
        }

    def score(self, y_score:  np.ndarray[float], y_real:  np.ndarray[float], **kwargs) -> float:

        inner_scores = {name: scorer.score(y_score=y_score, y_real=y_real) for\
                            name, scorer in self.inner_scorers.items()}
        score = sum(inner_scores.values) / len(inner_scores)
        
        return ScorerResult(value=score,
                            metainfo=inner_scores)

@dataclass
class ScorerInfo:
    name: str
    alias: str = ""
    params: dict = field(default_factory=dict)
    backgrounds: list[str] = field(default_factory=lambda: ['all']) # by default -- run for all benchmarks

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
        elif self.name == 'prroc_rocauc_top25':
            return PRROC_ROCAUC_TOP25(self.alias)
        elif self.name == 'prroc_rocauc_top50':
            return PRROC_ROCAUC_TOP50(self.alias)
        elif self.name == "prroc_averaged":
            return PRROC_ROCAUC_AVERAGED(self.alias)
        elif self.name == "prroc_prauc":
            tp = self.params.get("type")
            if tp is None:
                raise Exception("type must be specified for prauc scorer from PRROC package")
            tp = tp.lower()
            return PRROC_PRAUC(self.alias, tp)
        elif self.name == "prroc_prauc_top25":
            tp = self.params.get("type")
            if tp is None:
                raise Exception("type must be specified for prauc scorer from PRROC package")
            tp = tp.lower()
            return PRROC_PRAUC_TOP25(self.alias, tp)
        elif self.name == "prroc_prauc_top50":
            tp = self.params.get("type")
            if tp is None:
                raise Exception("type must be specified for prauc scorer from PRROC package")
            tp = tp.lower()
            return PRROC_PRAUC_TOP50(self.alias, tp)
        elif self.name == "prroc_prauc_averaged":
            tp = self.params.get("type")
            if tp is None:
                raise Exception("type must be specified for prauc scorer from PRROC package")
            tp = tp.lower()
            return PRROC_PRAUC_TOP50(self.alias, tp)
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
        return asdict(self)