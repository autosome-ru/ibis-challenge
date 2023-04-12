from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


from .prediction import Prediction
from ..utils import END_LINE_CHARS

class ScoreSubmissionFormatException(Exception):
    pass


@dataclass
class ScoreSubmission:
    name: str 
    tags: list[str]
    _sub: dict[str, Prediction]
    tag_col_name: str = "tag" 
    MAX_PRECISION: ClassVar[int] = 5
    MAX_0_1_DIFF: ClassVar[float] = 0.000015

    FIELDSEP: ClassVar[str] = "\t"

    def __getitem__(self, ds_name: str) -> Prediction:
        return self._sub[ds_name]
    
    def __setitem__(self, ds_name: str, pred: Prediction):
        self._sub[ds_name] = pred

    def get(self, ds_name: str) -> Prediction | None:
        return self._sub.get(ds_name)

    def __contains__(self, ds_name: str) -> bool:
        return ds_name in self._sub
    
    @classmethod
    def validate_score(cls, score_str: str):
        if score_str == Prediction.REPR_SKIPVALUE:
            return
        number = None        
        try:
            number = float(score_str)
        except ValueError:
            pass
        
        if number is None:
            raise ScoreSubmissionFormatException(f"Submission score must { Prediction.REPR_SKIPVALUE} or number: {score_str}")
        
        if number > 1 + cls.MAX_0_1_DIFF:
            raise ScoreSubmissionFormatException(f"Submission score must be in [0, 1]: {score_str}")
        
        if number < -cls.MAX_0_1_DIFF:
            raise ScoreSubmissionFormatException(f"Submission score must be in [0, 1]: {score_str}")
        
        fields = score_str.split('.')
        
        if len(fields) == 1: # 0 or 1
            return 
        after_point = fields[1]
        
        if len(after_point) > cls.MAX_PRECISION:
            raise ScoreSubmissionFormatException(f"Submission score must contain at most {cls.MAX_PRECISION} after . : {score_str}")
        
    
    @classmethod
    def load(cls, path: Path | str, name: str | None = None):
        if isinstance(path, str):
            path = Path(path)
        if name is None:
            name = path.name.split('.')[0]
        with path.open('r') as infile:
            header = infile.readline().rstrip(END_LINE_CHARS)
            fields = header.split(cls.FIELDSEP)
            tag_col_name = fields[0]
           
            if len(fields) == 1:
                msg = f"Submission must contain at least one tf field"
                raise ScoreSubmissionFormatException(msg)
            tfs = fields[1:]
            _sub = {tf_name: Prediction() for tf_name in tfs}
            tags = []
            for line in infile:
                line = line.rstrip(END_LINE_CHARS)
                vals = line.split(cls.FIELDSEP)
                tag = vals[0]
                tags.append(tag)
                for tf_name, val in zip(tfs, vals[1:]):
                    cls.validate_score(val)
                    _sub[tf_name][tag] = Prediction.str2val(val)
        return cls(name=name, 
                   tags=tags, 
                   _sub=_sub,
                   tag_col_name=tag_col_name)

    def prepare_for_evaluation(self, tfs: list | set | None = None) -> 'ScoreSubmission':
        _sub = self._sub.copy()
        tags = self.tags
        
        if tfs is not None:
            for tf in tfs:
                if tf not in self._sub:
                     print(f"Warning: no prediction submitted for factor {tf}", file=sys.stderr)
            for tf in self._sub:
                if tf not in tfs:
                    msg = F"No such factor in benchmark: {tf}"
                    raise ScoreSubmissionFormatException(msg)
                

        for tf_name, pred in self._sub.items():
            for t in tags:
                if Prediction.is_skipvalue(pred[t]):
                    print(f"Column for factor {tf_name} contains skipped prediction for {t}." 
                            "Predictions for this factor won't be evaluated", 
                            file=sys.stderr)
                    _sub.pop(tf_name)
                    break
                
                
        if len(_sub) == 0:
            msg = f'Submission must contain full information for at least one factor'
            raise ScoreSubmissionFormatException(msg)

        cls = type(self)
        return cls(name=self.name,
                   tags=tags, 
                   _sub=_sub)

    @property
    def header(self) -> str:
        flds = [self.tag_col_name, *self._sub.keys()]
        return self.FIELDSEP.join(flds)
    
    @property
    def ds_names(self) -> list[str]:
        return list(self._sub.keys())
    
    def write(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        with path.open("w") as outp: 
            print(self.header, file=outp)
            for tag in self.tags:
                vals = [tag]
                for pred in self._sub.values():
                    v = pred[tag]
                    v = Prediction.val2str(v)
                    vals.append(v)
                print(*vals, sep=self.FIELDSEP, file=outp)

    @classmethod
    def template(cls, name: str, 
                 ds_names: list[str],
                 tags: list[str],
                 tag_col_name: str = "tag"):
        _sub = {tf: Prediction.template(tags) for tf in ds_names}
        return cls(name=name, 
                   tags=tags,
                   _sub=_sub,
                   tag_col_name=tag_col_name)