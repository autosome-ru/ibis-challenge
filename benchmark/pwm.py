from attrs import define
from utils import auto_convert
from pathlib import Path
from typing import Sequence
from dataset import Dataset
from prediction import Prediction
import shlex
import subprocess

@define(field_transformer=auto_convert)
class PWMEvalPredictor:
    pwmeval_path: Path
    pwm_path: Path
    
    def score(self, Xs: Sequence[Dataset]) -> Prediction:
        all_dt = {}
        for X in Xs:
            dt = self.score_dataset(X)
            all_dt.update(dt)
        return Prediction(all_dt)

    def score_dataset(self, X: Dataset) -> dict:
        queries = []
        for entry in X:
            query = f">{entry.tag}\n{entry.sequence}"
            queries.append(query)
        total_query = "\n".join(queries)
        answer = self.process_query(total_query)
        prediction = self.process_answer(answer)
        return prediction
    
    def process_query(self, query: str) -> str:
        cmd = f"{self.pwmeval_path} -m {self.pwm_path}"
        cmd = shlex.split(cmd)
        p = subprocess.Popen(cmd, 
                       stdout=subprocess.PIPE,
                       stdin=subprocess.PIPE,
                       text=True)
        stdout, _ = p.communicate(query)
        return stdout
    
    def process_answer(self, answer: str) -> dict:
        dt = {}
        for line in answer.splitlines():
            tag, score = line.split()
            score = float(score)
            dt[tag] = score
        return dt