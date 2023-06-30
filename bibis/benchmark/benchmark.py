from re import sub
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
    scoring_type: str
    parent_name: str = ""

@dataclass
class AAASubmit(Submit):
    name: str
    scores: ScoreSubmission
    scoring_type: str = "AAA"
    parent_name: str = ""
    
    def __post_init__(self):
        self.parent_name = self.name 
    
    
@dataclass
class MatrixSumbit(Submit):
    name: str
    matrix_path: Path
    matrix_type: str
    scoring_type: str
    tf: str
    parent_name: str 
    
    SUM_SCORE_NAME: ClassVar[str] = "sumocc"
    BEST_SCORE_NAME: ClassVar[str] = "besthit"
    
    def _predict_pwm(self,
                     ds: DatasetInfo,
                     pwm_path: Path,
                     pwmeval_path: Path) -> Prediction:
        model = MatrixMaxPredictor(pwm_path, 
                                    pwmeval_path=pwmeval_path)
        scores = Prediction(model.score_file(ds.fasta_path))
        return scores 
    
    def _predict_pfm(self, 
                     ds: DatasetInfo,
                     pfm_path: Path,
                     pwmeval_path: Path) -> Prediction:
        if self.scoring_type == self.SUM_SCORE_NAME:
            model = MatrixSumPredictor(pfm_path, 
                                    pwmeval_path=pwmeval_path)
            scores = Prediction(model.score_file(ds.fasta_path))
        elif self.scoring_type == self.BEST_SCORE_NAME:
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
    
    def predict(self, ds: DatasetInfo, pwmeval_path: Path) -> Prediction:
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

    def submit_score_submission(self, sub: ScoreSubmission):
        aaa_sub = AAASubmit(name=sub.name,
                            scores=sub)
        self.submits.append(aaa_sub)
        
    def submit_matrix_model(self,
                            name: str, 
                            tf: str,
                            matrix_path: str | Path, 
                            matrix_type: str, 
                            scoring_type: str,
                            parent_name: str = ""):
        if isinstance(matrix_path, str):
            matrix_path = Path(matrix_path)
        mat_sub = MatrixSumbit(name=name,
                               matrix_path=matrix_path,
                               matrix_type=matrix_type,
                               scoring_type=scoring_type, 
                               tf=tf,
                               parent_name=parent_name)
        self.submits.append(mat_sub)
        
    def submit_pfm_info(self, 
                        pfm_info: PFMInfo,
                        parent_name: str = ""):
        self.submit_matrix_model(name=pfm_info.tag,
                                 tf=pfm_info.tf,
                                 matrix_path=pfm_info.path,
                                 matrix_type="pfm",
                                 scoring_type=MatrixSumbit.BEST_SCORE_NAME,
                                 parent_name=parent_name)
        self.submit_matrix_model(name=pfm_info.tag,
                                 tf=pfm_info.tf,
                                 matrix_path=pfm_info.path,
                                 matrix_type="pfm",
                                 scoring_type=MatrixSumbit.SUM_SCORE_NAME,
                                 parent_name=parent_name)
        
    def submit_pwm_submission(self, pwm_sub: PWMSubmission):
        sub_dir = self.results_dir / pwm_sub.name
        for pfm_info in pwm_sub.split_into_pfms(sub_dir):
            self.submit_pfm_info(pfm_info, parent_name=pwm_sub.name)
            
    def submit(self, submission: PWMSubmission | ScoreSubmission):
        if isinstance(submission, PWMSubmission):
            self.submit_pwm_submission(submission)
        elif isinstance(submission, ScoreSubmission):
            self.submit_score_submission(submission)
        else:
            raise Exception(f"Wrong submission class: {type(submission)}")
        
        
    def score_prediction(self, ds: DatasetInfo, prediction: Prediction) -> dict[str, float]:
        
        #labelled_seqs = read_fasta(ds.path)
        answer = ds.answer()
        if self.kind == "CHS" or self.kind == "GHTS":
            true_y = list(map(int, answer.values()))
            pred_y: list[float] = []
            for tag in answer.keys():
                score = prediction.get(tag)
                if score is None:
                    print(f"Prediction doesn't contain information for sequence {tag}, skipping prediction",
                          file=sys.stderr)
                    return self.skipped_prediction
                pred_y.append(score)
            
            scores: dict[str, float] = {}
            for sc in self.scorers:
                scores[sc.name] = sc.score(y_score=pred_y, y_real=true_y)
            return scores
        else:
            raise NotImplementedError()

    def score_aaa_submit(self, submit: AAASubmit) -> dict[str, dict[str, float]]:
        scores = {}
        for ds in self.datasets:
            if ds.tf in submit.scores:
                scores[ds.name] = self.score_prediction(ds, submit.scores[ds.tf])
            else:
                print(f"No predictions for {ds.name} are provided. Skipping", 
                      file=sys.stderr)
                scores[ds.name] = self.skipped_prediction
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
                pbar.set_description(f"Processing {sub.name}, {sub.scoring_type}")
                if isinstance(sub, AAASubmit):
                    scores = self.score_aaa_submit(sub)

                elif isinstance(sub, MatrixSumbit):
                    scores = self.score_matrix_submit(sub)
                else:
                    raise Exception("Wrong submit type")
                
                for ds_name, ds_dt in scores.items():
                    if ds_dt == self.skipped_prediction:
                        continue 
                    for sc, value in ds_dt.items():
                        ds = ds_mapping[ds_name]
                        results.append([sub.parent_name, 
                                        sub.name, 
                                        sub.scoring_type,
                                        ds.tf, 
                                        ds.background,
                                        sc,
                                        value])
                pbar.update(1)
        
        df = pd.DataFrame(results,
                          columns=["submission_name",
                                   "part_name",
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
    