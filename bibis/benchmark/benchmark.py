import tempfile
import tqdm
import pandas as pd
import numpy as np 

from pathlib import Path
from dataclasses import dataclass 
from typing import ClassVar, List
from pathlib import Path

from ..scoring.scorer import Scorer
from ..scoring.prediction import Prediction
from ..scoring.submission import Submission
from ..benchmark.dataset import Dataset
from ..matrix.pwmeval import MatrixSumPredictor, MatrixMaxPredictor
from ..matrix.pwm import PCM, PFM

from .dataset import Dataset

class Submit:
    name: str

@dataclass
class AAASubmit(Submit):
    name: str
    score_path: Path

@dataclass
class MatrixSumbit(Submit):
    name: str
    matrix_path: Path
    matrix_type: str
    scoring_type: str
    tf: str
    
    def _predict_pwm(self,
                     ds: Dataset,
                     pwm_path: Path,
                     pwmeval_path: Path) -> Prediction:
        model = MatrixMaxPredictor(pwm_path, 
                                    pwmeval_path=pwmeval_path)
        scores = Prediction(model.score_file(ds.path))
        return scores 
    
    def _predict_pfm(self, 
                     ds,
                     pfm_path: Path,
                     pwmeval_path: Path) -> Prediction:
        if self.scoring_type == "sumscore":
            model = MatrixSumPredictor(pfm_path, 
                                    pwmeval_path=pwmeval_path)
            scores = Prediction(model.score_file(ds.path))
        elif self.scoring_type == "best":
            pfm = PFM.load(self.matrix_path)
            int_pwm = pfm.pwm().intpwm()
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                pwm_path = tempdir / 'matrix.pwm'
                int_pwm.write(pwm_path)
                return self._predict_pwm(ds, 
                                         pwm_path=pwm_path, 
                                         pwmeval_path=pwmeval_path)
        else:
            raise Exception("Wrong scoring type")
        return scores 
    
    def predict(self, ds, pwmeval_path: Path) -> Prediction:
        if self.matrix_type == "pfm":
            return self._predict_pfm(ds, self.matrix_path, pwmeval_path=pwmeval_path)
        elif self.matrix_type == "pcm":
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                pfm_path = tempdir / 'matrix.pfm'
                pcm = PCM.load(self.matrix_path)
                pfm = pcm.pfm()
                pfm.write(pfm_path)
                return self._predict_pfm(ds, pfm_path, pwmeval_path=pwmeval_path)
        elif self.matrix_type == "pwm":
            if self.scoring_type == "sumscore":
                raise Exception("Sumscore mode is not available for the PWM matrix")
            else:
                return self._predict_pwm(ds, 
                                         pwm_path=self.matrix_path, 
                                         pwmeval_path=pwmeval_path)
        else:
            raise Exception("Wrong matrix type")
 

@dataclass
class Benchmark:
    name: str
    datasets: list[Dataset]
    scorers: list[Scorer]
    submits: list[Submit]
    
    results_dir: Path
    metainfo: dict
    pwmeval_path: Path
    n_procs: int

    SKIPVALUE: ClassVar[float] = np.nan
    REPR_SKIPVALUE: ClassVar[str] = "skipped"

    @property
    def skipped_prediction(self) -> dict[str, float]:
        ds_scores = {sc.name: self.SKIPVALUE for sc in self.scorers}
        return ds_scores
    
    def submit_aaa_model(self, name: str, score_path: str | Path):
        if isinstance(score_path, str):
            score_path = Path(score_path)
        aaa_sub = AAASubmit(name=name, 
                            score_path=score_path)
        self.submits.append(aaa_sub)
        
    def submit_matrix_model(self, 
                           name: str, 
                           tf: str,
                           matrix_path: str | Path, 
                           matrix_type: str, 
                           scoring_type: str):
        if isinstance(matrix_path, str):
            matrix_path = Path(matrix_path)
        mat_sub = MatrixSumbit(name=name,
                               matrix_path=matrix_path,
                               matrix_type=matrix_type,
                               scoring_type=scoring_type, 
                               tf=tf)
        self.submits.append(mat_sub)
        
    def score_prediction(self, ds: Dataset, prediction: Prediction) -> dict[str, float]:
        raise NotImplementedError()
    
    def score_aaa_submit(self, submit: AAASubmit) -> dict[str, dict[str, float]]:
        sub = Submission.load(submit.score_path)
        scores = {}
        for ds in self.datasets:
            if ds.name in sub:
                scores[ds] = self.score_prediction(ds, sub[ds.name])
        return scores 
    
    def score_matrix_submit(self, submit: MatrixSumbit) -> dict[str, dict[str, float]]:
        scores = {}
        for ds in self.datasets:
            if ds.tf == submit.tf:
                prediction = submit.predict(ds, self.pwmeval_path)
                scores[ds] = self.score_prediction(ds, prediction)
        return scores
                
    def get_results_file_path(self, name: str):
        return self.results_dir / f"{name}.tsv"
    
    def run(self):
        results = []
        with tqdm.tqdm(total=len(self.submits)) as pbar:
            for sub in self.submits:
                pbar.set_description(f"Processing {sub.name}")
                if isinstance(sub, AAASubmit):
                    scores = self.score_aaa_submit(sub)
                elif isinstance(sub, MatrixSumbit):
                    scores = self.score_matrix_submit(sub)
                else:
                    raise Exception("Wrong submit type")
                
                for ds, ds_dt in scores.items():
                    for sc, value in ds_dt.items():
                        results.append([sub.name, ds.name, sc, value])
                pbar.update(1)
        
        df = pd.DataFrame(results,
                          columns=["name",
                                   "dataset", 
                                   "score",
                                   "value"])
        return df