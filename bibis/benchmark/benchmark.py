import tempfile
import sys

import tqdm
import pandas as pd
import numpy as np 

from pathlib import Path
from dataclasses import dataclass 
from typing import ClassVar
from pathlib import Path

from ..scoring.scorer import SklearnROCAUC, SklearnPRAUC, PRROC_ROCAUC, PRROC_PRAUC, ConstantScorer
from .prediction import Prediction
from .score_submission import ScoreSubmission
from ..matrix.pwmeval import MatrixSumPredictor, MatrixMaxPredictor
from ..matrix.pwm import PCM, PFM
from ..seq.seqentry import read_fasta

from .dataset import DatasetInfo
from .benchmarkconfig import BenchmarkConfig
from .pwm_submission import PFMInfo, PWMSubmission

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
                     ds: DatasetInfo,
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
    kind: str
    datasets: list[DatasetInfo]
    scorers: list[SklearnROCAUC | SklearnPRAUC | PRROC_ROCAUC | PRROC_PRAUC | ConstantScorer]
    submits: list[Submit]
    
    results_dir: Path
    metainfo: dict
    pwmeval_path: Path

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
        
    def submit_pfm_info(self, pfm_info: PFMInfo):
        self.submit_matrix_model(name=pfm_info.tag,
                                 tf=pfm_info.tf,
                                 matrix_path=pfm_info.path,
                                 matrix_type="pfm",
                                 scoring_type="best")
        self.submit_matrix_model(name=pfm_info.tag,
                                 tf=pfm_info.tf,
                                 matrix_path=pfm_info.path,
                                 matrix_type="pfm",
                                 scoring_type="sumscore")
        
    def submit_pwm_submission(self, pwm_sub: PWMSubmission):
        for pfm_info in pwm_sub.split_into_pfms(self.results_dir):
            self.submit_pfm_info(pfm_info)
        
    def score_prediction(self, ds: DatasetInfo, prediction: Prediction) -> dict[str, float]:
        labelled_seqs = read_fasta(ds.path)
        if self.kind == "ChIPSeq":
            true_y = [int(s.label) for s in labelled_seqs]
            pred_y: list[float] = []
            for s in labelled_seqs:
                score = prediction.get(s.tag)
                if score is None:
                    print(f"Prediction doesn't contain information for sequence {s.tag}, skipping prediction", file=sys.stderr)
                    return self.skipped_prediction
                pred_y.append(score)
            
            scores: dict[str, float] = {}
            for sc in self.scorers:
                scores[sc.name] = sc.score(y_score=pred_y, y_real=true_y)
            return scores
        else:
            raise NotImplementedError()

    def score_aaa_submit(self, submit: AAASubmit) -> dict[str, dict[str, float]]:
        sub = ScoreSubmission.load(submit.score_path)
        scores = {}
        for ds in self.datasets:
            if ds.name in sub:
                scores[ds.name] = self.score_prediction(ds, sub[ds.name])
        return scores 
    
    def score_matrix_submit(self, submit: MatrixSumbit) -> dict[str, dict[str, float]]:
        scores = {}
        for ds in self.datasets:
            if ds.tf == submit.tf:
                prediction = submit.predict(ds, self.pwmeval_path)
                scores[ds.name] = self.score_prediction(ds, prediction)
        return scores
                
    def get_results_file_path(self, name: str):
        return self.results_dir / f"{name}.tsv"
    
    def run(self):
        
        ds_mapping = {ds.name: ds for ds in self.datasets}
        results = []
        with tqdm.tqdm(total=len(self.submits)) as pbar:
            for sub in self.submits:
                pbar.set_description(f"Processing {sub.name}")
                if isinstance(sub, AAASubmit):
                    scores = self.score_aaa_submit(sub)
                    scoring_type = "AAA"
                elif isinstance(sub, MatrixSumbit):
                    scores = self.score_matrix_submit(sub)
                    scoring_type = sub.scoring_type
                else:
                    raise Exception("Wrong submit type")
                
                for ds_name, ds_dt in scores.items():
                    for sc, value in ds_dt.items():
                        ds = ds_mapping[ds_name]
                        results.append([sub.name, scoring_type, ds.tf, ds.background, sc, value])
                pbar.update(1)
        
        df = pd.DataFrame(results,
                          columns=["name",
                                   "scoring_type",
                                   "tf", 
                                   "background",
                                   "score",
                                   "value"])
        return df
    
    @classmethod
    def from_cfg(cls, 
                 cfg: BenchmarkConfig, 
                 results_dir: str | Path) -> 'Benchmark':
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        
        scorers = [sc.make() for sc in cfg.scorers]
        
        return Benchmark(name=cfg.name,
                         kind=cfg.kind,
                         datasets=cfg.datasets,
                         scorers=scorers, 
                         submits=[],
                         results_dir=results_dir,
                         metainfo=cfg.metainfo,
                         pwmeval_path=cfg.pwmeval_path)
    