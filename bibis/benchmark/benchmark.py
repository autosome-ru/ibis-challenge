import tempfile
import sys

import tqdm
import pandas as pd
import numpy as np 

from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass 
from typing import Any, ClassVar
from pathlib import Path

from ..scoring.scorer import Scorer, ScorerResult
from .prediction import Prediction
from .score_submission import ScoreSubmission
from ..matrix.pwmeval import MatrixSumPredictor, MatrixMaxPredictor
from ..matrix.pwm import PCM, PFM

from .dataset import DatasetInfo
from .benchmarkconfig import BenchmarkConfig
from .pwm_submission import PFMInfo, PWMSubmission
from ..logging import get_bibis_logger

logger = get_bibis_logger()

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
                                       pwmeval_path=pwmeval_path,
                                       pseudocount=PFM.EPS)
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
    scorers: dict[str, Scorer]
    submits: list[Submit]
    
    results_dir: Path
    metainfo: dict
    pwmeval_path: Path

    SKIPVALUE: ClassVar[float] = ScorerResult(np.nan)
    REPR_SKIPVALUE: ClassVar[str] = "skipped"
    
    def skipped_prediction(self, ds: DatasetInfo) -> dict[str, float]:
        ds_scores = {sc.name: self.SKIPVALUE for sc in self.scorers[ds.background]}
        return ds_scores
    
    def is_skipped_prediction(self, dt: dict[str, Any]):
        return all(v == self.SKIPVALUE for v in dt.values())

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
        if self.kind in ("CHS",  "GHTS", "PBM", "SMS", "HTS"): 
            true_y = list(map(int, answer['labels'].values()))
            if 'groups' in answer:
                y_group = np.array(list(answer['groups'].values()), 
                                   dtype=np.float32)
            else:
                y_group = None
            pred_y: list[float] = []
            for tag in answer['labels'].keys():
                score = prediction.get(tag)
                if score is None:
                    logger.warning(f"Prediction doesn't contain information for sequence {tag}, skipping prediction")
                    return self.skipped_prediction(ds)
                pred_y.append(score)

            pred_y = np.array(pred_y)
            true_y = np.array(true_y)
            ord = np.argsort(pred_y)
            pred_y = pred_y[ord]
            true_y = true_y[ord]
            if y_group is not None:
                y_group = y_group[ord]

            scores: dict[str, float] = {}
            for sc in self.scorers[ds.background]:
                scores[sc.name] = sc.score(y_score=pred_y, y_real=true_y, y_group=y_group)
            return scores
        else:
            raise NotImplementedError(f"Benchmark scoring is not implemented for {self.kind}")

    def score_aaa_submit(self, submit: AAASubmit) -> dict[str, dict[str, ScorerResult]]:
        scores = {}
        for ds in self.datasets:
            if ds.tf in submit.scores:
                scores[ds.name] = self.score_prediction(ds, submit.scores[ds.tf])
            else:
                logger.warning(f"No predictions for {ds.name} are provided. Skipping")
                scores[ds.name] = self.skipped_prediction(ds)
        return scores 
    
    def score_matrix_submit(self, submit: MatrixSumbit) -> dict[str, dict[str, ScorerResult]]:
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
        if len(ds_mapping) != len(self.datasets):
            raise Exception("Dataset names must be unique")
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
                    if self.is_skipped_prediction(ds_dt):
                        continue 
                    for sc, sc_result in ds_dt.items():
                        ds = ds_mapping[ds_name]
                        results.append([sub.parent_name, 
                                        sub.name, 
                                        sub.scoring_type,
                                        ds.tf, 
                                        ds.background,
                                        sc,
                                        sc_result.value,
                                        sc_result.metainfo])
                pbar.update(1)
        
        df = pd.DataFrame(results,
                          columns=["submission_name",
                                   "part_name",
                                   "scoring_type",
                                   "tf", 
                                   "background",
                                   "score",
                                   "value",
                                   "metainfo"])
        if self.kind == "PBM":
            back = [x.split("_")[0] for x in df['background'].values]
            exp = [x.split("_")[1] for x in df['background'].values]
            df['background'] = back
            df['experiment_id'] = exp
        
        return df
    
    @classmethod
    def from_cfg(cls, 
                 cfg: BenchmarkConfig, 
                 results_dir: str | Path) -> 'Benchmark':
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        
        all_backgrounds = set(ds.background for ds in cfg.datasets)

        back2scorers = defaultdict(list)
        for sc_cfg in cfg.scorers:
            backs_lst = []
            all_specified = False
            for back in sc_cfg.backgrounds:
                if back == 'all':
                    all_specified = True
                elif back not in all_backgrounds:
                    raise Exception(f"Wrong background specified: {back}")
                else:
                    backs_lst.append(back)
            if all_specified:
                if len(backs_lst) != 0:
                    logger.warning("'All' background specified, all background will be used, other specifications are ignored") 
                backs_lst = all_backgrounds
            
            scorer = sc_cfg.make() 
            for back in backs_lst:
                back2scorers[back].append(scorer)


        return Benchmark(name=cfg.name,
                         kind=cfg.kind,
                         datasets=cfg.datasets,
                         scorers=dict(back2scorers), 
                         submits=[],
                         results_dir=results_dir,
                         metainfo=cfg.metainfo,
                         pwmeval_path=cfg.pwmeval_path)