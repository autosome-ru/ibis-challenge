from abc import ABCMeta, abstractmethod
from ast import Sub
from dataclasses import dataclass
from pathlib import Path
from tkinter import SEPARATOR
from typing import Dict, List
from attr import define
import numpy as np
from random import Random
from prediction import Prediction, Submission
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
    scores: Submission

    @classmethod
    def load(cls, path: Path, sep="\t"):
        scores = Submission.load(path)
        return cls(scores)
    
    def score(self, X: Dataset) -> Prediction:
        pred = self.scores.get(X.tf_name)
        if pred is None:
            return Prediction({})
        predictions = {}
        for entry in X:
            tag = entry.tag
            score = pred.get(tag)
            if score is not None:
                predictions[tag] = score
        return Prediction(predictions)