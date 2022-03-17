from attrs import define
from utils import auto_convert, TEMPORARY_FILES_DIR, temporary_file
from dataset import Dataset
from prediction import Prediction
import shlex
import subprocess
from model import Model
from pathlib import Path
from typing import Sequence, ClassVar, Optional, Union
from pwm import PFM

@define(field_transformer=auto_convert)
class PWMEvalPFMPredictor(Model):
    matrix_path: Path
    pwmeval_path: Path
    PWMEvalSeparator: ClassVar[str] = "\t"

    @classmethod
    def from_pfm(cls, 
                 pfm_path: Path, 
                 pwveval_path: Path):
        return cls(matrix_path=pfm_path, 
                   pwmeval_path=pwveval_path)

    def score(self, X: Dataset) -> Prediction:
        dt = self.score_dataset(X)
        return Prediction(dt)

    def score_batch(self, Xs: Sequence[Dataset]) -> Prediction:
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

    def get_cmd(self) -> str:
        return f"{self.pwmeval_path} -m {self.matrix_path}"
    
    def process_query(self, query: str) -> str:
        cmd = self.get_cmd()
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


@define(field_transformer=auto_convert)
class PWMEvalPWMPredictor(Model):
    matrix_path: Path
    pwmeval_path: Path
    PWMEvalSeparator: ClassVar[str] = "\t"

    @staticmethod
    def pfm2pwm(pfm_path: Path,
                pwm_path: Path):
        PFM.load(pfm_path).intpwm().write(pwm_path)
    
    @classmethod
    def from_pfm(cls, 
                 pfm_path: Path, 
                 pwmeval_path: Path,
                 pwm_path: Optional[Path] = None):
        if pwm_path is None:
            pwm_path = temporary_file()
        cls.pfm2pwm(pfm_path=pfm_path, 
                     pwm_path=pwm_path)
        return cls(matrix_path=pwm_path, pwmeval_path=pwmeval_path)
        

    def score(self, X: Dataset) -> Prediction:
        dt = self.score_dataset(X)
        return Prediction(dt)

    def score_batch(self, Xs: Sequence[Dataset]) -> Prediction:
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

    def get_cmd(self) -> str:
        return f"{self.pwmeval_path} -m {self.matrix_path} --best --pwm"
    
    def process_query(self, query: str) -> str:
        cmd = self.get_cmd()
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
            fields = line.rsplit(maxsplit=2, sep=self.PWMEvalSeparator)
            tag, score = fields[0], fields[-2]
            score = float(score)
            dt[tag] = score
        return dt