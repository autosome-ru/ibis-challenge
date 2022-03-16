from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tkinter import SEPARATOR
from typing import Dict, List
from attr import define
import numpy as np
from random import Random
from prediction import Prediction
from exceptions import ModelNotTrainedException

from dataset import Dataset

class Model(metaclass=ABCMeta):
    @abstractmethod
    def score(self, X: Dataset) -> Prediction:
        pass

class RandomPredictor(Model):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = seed

    def score(self, X: Dataset) -> Prediction:
        r = Random(self.seed) 
        scores = {e.tag: r.random() for e in X}
        return Prediction(scores)

class ClassBalancePredictor(Model):
    def __init__(self):
        super().__init__()
        self.is_trained = False
    
    def train(self, X, y):
        self.balance = np.mean(y)
        self.is_trained = True 

    def score(self, X: Dataset) -> Prediction:
        if not self.is_trained:
            raise ModelNotTrainedException("Model is not trained")
        dt = {e.tag: self.balance for e in X}
        return Prediction(dt)

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
    
    def score(self, X: Dataset) -> List[float]:
        predictions = []
        for entry in X.entries:
            tag = entry.tag
            score = self.scores[tag]
            predictions.append(score)
        return predictions