from abc import ABCMeta, abstractmethod
from pathlib import Path
from tkinter import SEPARATOR
from typing import Dict, List
from attr import define
import numpy as np
from numpy.random import default_rng
from prediction import Prediction

from dataset import Dataset

class Model(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.is_trained = False

    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def score(self, X):
        pass

class RandomPredictor(Model):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.is_trained = True
        self.rng = default_rng(seed)

    def train(self, X, y):
        self.is_trained = True
    
    def score(self, X): 
        y_shape = len(X)
        score = self.rng.random(size=y_shape, dtype=np.float32)
        return score

class ClassBalancePredictor(Model):
    def __init__(self):
        super().__init__()
    
    def train(self, X, y):
        self.balance = np.mean(y)
    
    def score(self, X):
        y_shape = len(X)
        score = np.full(shape=y_shape, fill_value=self.balance, dtype=np.float32)
        return score

@define
class DictPredictor(Model):
    '''
    predict using entry-unique tag
    '''
    scores: Prediction

    @classmethod
    def load(cls, path: Path, sep="\t"):
        scores = Prediction.load(path)
        return cls(scores)

    def train(self, X, y):
        self.is_trained = True
    
    def score(self, X: Dataset) -> List[float]:
        predictions = []
        for entry in X.entries:
            tag = entry.tag
            score = self.scores[tag]
            predictions.append(score)
        return predictions