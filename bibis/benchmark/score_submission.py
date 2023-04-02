from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


from .prediction import Prediction
from ..utils import END_LINE_CHARS

@dataclass
class ScoreSubmission:
    name: str 
    tags: list[str]
    _sub: dict[str, Prediction]
    tag_col_name: str = "tag" 

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
            tfs = fields[1:]
            _sub = {tf_name: Prediction() for tf_name in tfs}
            tags = []
            for line in infile:
                line = line.rstrip(END_LINE_CHARS)
                vals = line.split(cls.FIELDSEP)
                tag = vals[0]
                tags.append(tag)
                for tf_name, val in zip(tfs, vals[1:]):
                    _sub[tf_name][tag] = Prediction.str2val(val)
        return cls(name=name, 
                   tags=tags, 
                   _sub=_sub,
                   tag_col_name=tag_col_name)

    def prepare_for_evaluation(self) -> 'ScoreSubmission':
        _sub = self._sub.copy()
        tags = self.tags

        for tf_name, pred in self._sub.items():
            for t in tags:
                if Prediction.is_skipvalue(pred[t]):
                    print(f"Column for factor {tf_name} contains skipped prediction for {t}." 
                            "Predictions for this factor won't be evaluated")
                    _sub.pop(tf_name)
                    break
        if len(_sub) == 0:
            msg = f'Submission must contain full information for at least one factor'
            raise Exception(msg)
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

    @classmethod
    def from_single_prediction(cls, 
                               name: str, 
                               tf_name: str,
                               pred: Prediction,
                               tag_col_name: str = "tag"):
        tags = pred.tags
        _sub = {tf_name: pred}
        return cls(name=name, 
                   tags=tags, 
                   _sub=_sub,
                   tag_col_name=tag_col_name)