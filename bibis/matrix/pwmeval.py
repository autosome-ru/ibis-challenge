import shlex
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar, Iterable, Iterator, Optional, Union

from ..seq.seqentry import SeqEntry

@dataclass
class MatrixSumPredictor:
    matrix_path: Path
    pwmeval_path: Path
    PWMEvalSeparator: ClassVar[str] = "\t"

    @classmethod
    def from_pfm(cls, 
                 pfm_path: Union[Path, str], 
                 pwmeval_path: Union[Path, str]):
        if isinstance(pfm_path, str):
            pfm_path = Path(pfm_path)
        if isinstance(pwmeval_path, str):
            pwmeval_path = Path(pwmeval_path)
        return cls(matrix_path=pfm_path, 
                   pwmeval_path=pwmeval_path)
    
    def score(self, X: Iterator[SeqEntry]) -> dict[str, float]:
        dt = self.score_dataset(X)
        return dt

    def score_dataset(self, X: Iterator[SeqEntry]) -> dict[str, float]:
        queries = []
        for entry in X:
            query = f">{entry.tag}\n{entry.sequence}"
            queries.append(query)
        total_query = "\n".join(queries)
        answer = self.process_query(total_query)
        prediction = self.process_answer(answer)
        return prediction

    def get_cmd(self, path: str| Path | None) -> str:
        return f"{self.pwmeval_path} -m {self.matrix_path} {path}"
    
    def process_query(self, query: str) -> str:
        cmd = self.get_cmd(None)
        cmd = shlex.split(cmd)
        p = subprocess.Popen(cmd, 
                       stdout=subprocess.PIPE,
                       stdin=subprocess.PIPE,
                       text=True)
        stdout, _ = p.communicate(query)
        return stdout
    
    def process_answer(self, answer: str) -> dict[str, float]:
        dt = {}
        for line in answer.splitlines():
            tag, score = line.split()
            score = float(score)
            dt[tag] = score
        return dt
    
    def process_file(self, path: str | Path) -> str:
        cmd = self.get_cmd(path)
        cmd = shlex.split(cmd)
        p = subprocess.Popen(cmd, 
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       stdin=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if len(stderr) != 0 or p.returncode != 0:
            raise Exception(f"PWMEval exited with error {stderr}")
        try:
            stdout = stdout.decode()
        except UnicodeDecodeError:
            stdout = None
        if stdout is None:
            raise Exception("PWMEval returned non-unicode symbols")
        return stdout
    
    def score_file(self, path: str | Path) -> dict:       
        answer = self.process_file(path)
        prediction = self.process_answer(answer)
        return prediction


@dataclass
class MatrixMaxPredictor: #(Model):
    matrix_path: Path
    pwmeval_path: Path
    PWMEvalSeparator: ClassVar[str] = "\t"


    @classmethod
    def from_pfm(cls, 
                 pwm_path: Path, 
                 pwmeval_path: Path):
        return cls(matrix_path=pwm_path,
                   pwmeval_path=pwmeval_path)    

    def score(self, X: Iterable[SeqEntry]) -> dict[str, float]:
        dt = self.score_dataset(X)
        return dt  
    
    def process_query(self, query: str) -> str:
        cmd = self.get_cmd(None)
        cmd = shlex.split(cmd)
        p = subprocess.Popen(cmd, 
                       stdout=subprocess.PIPE,
                       stdin=subprocess.PIPE,
                       text=True)
        stdout, _ = p.communicate(query)
        return stdout

    def score_dataset(self, X: Iterable[SeqEntry]) -> dict:       
        queries = []
        for entry in X:
            query = f">{entry.tag}\n{entry.sequence}"
            queries.append(query)
        total_query = "\n".join(queries)
        answer = self.process_query(total_query)
        prediction = self.process_answer(answer)
        return prediction

    def get_cmd(self, path: str | Path | None) -> str:
        if path is None:
            return f"{self.pwmeval_path} -m {self.matrix_path} --best --pwm"
        else:
            return f"{self.pwmeval_path} -m {self.matrix_path} --best --pwm {path}"
    
    def process_file(self, path: str | Path) -> str:
        cmd = self.get_cmd(path)
        cmd = shlex.split(cmd)
        p = subprocess.Popen(cmd, 
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       stdin=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if len(stderr) != 0 or p.returncode != 0:
            raise Exception(f"PWMEval exited with error {stderr}")
        try:
            stdout = stdout.decode()
        except UnicodeDecodeError:
            stdout = None
        if stdout is None:
            raise Exception("PWMEval returned non-unicode symbols")
        return stdout
    
    def score_file(self, path: str | Path) -> dict:       
        answer = self.process_file(path)
        prediction = self.process_answer(answer)
        return prediction
    
    def process_answer(self, answer: str) -> dict:
        dt = {}
        for line in answer.splitlines():
            fields = line.split(sep=self.PWMEvalSeparator)
            tag, score = fields[0], fields[-2]
            score = float(score)
            dt[tag] = score
        return dt