from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


from .prediction import Prediction
from ..utils import END_LINE_CHARS

@dataclass
class ScoreSubmission:
    tags: list[str]
    _sub: dict[str, Prediction]

    TAGFIELDNAME: ClassVar[str] = "tag"
    FIELDSEP: ClassVar[str] = "\t"

    def __getitem__(self, tf: str) -> Prediction:
        return self._sub[tf]
    
    def __setitem__(self, tf: str, pred: Prediction):
        self._sub[tf] = pred

    def get(self, tf: str) -> Prediction | None:
        return self._sub.get(tf)

    def __contains__(self, tf: str) -> bool:
        return tf in self._sub
    
    @classmethod
    def load(cls, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        with path.open('r') as infile:
            header = infile.readline().rstrip(END_LINE_CHARS)
            fields = header.split(cls.FIELDSEP)
            if fields[0] != cls.TAGFIELDNAME:
                msg = f'First submission field must be tag field: {cls.TAGFIELDNAME}'
                raise Exception(msg)
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
        return cls(tags, _sub)

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
        return cls(tags, _sub)

    @property
    def header(self) -> str:
        flds = [self.TAGFIELDNAME, *self._sub.keys()]
        return self.FIELDSEP.join(flds)
    
    @property
    def tfs(self) -> list[str]:
        return list(self._sub.keys())
    
    def write(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        with path.open("w") as outp: 
            print(self.header, file=outp)
            for tag in self.tags:
                vals = [tag]
                for tf, pred in self._sub.items():
                    v = pred[tag]
                    v = Prediction.val2str(v)
                    vals.append(v)
                print(*vals, sep=self.FIELDSEP, file=outp)

    @classmethod
    def template(cls, tfs: list[str], tags: list[str]):
        _sub = {tf: Prediction.template(tags) for tf in tfs}
        return cls(tags, _sub)

    @classmethod
    def write_template(cls, tfs: list[str], tags: list[str], path: Path):
        cls.template(tfs, tags).write(path)

    @classmethod
    def from_single_prediction(cls, tf_name: str, pred: Prediction):
        tags = pred.tags
        _sub = {tf_name: pred}
        return cls(tags, _sub)

if __name__ == "__main__":
    ScoreSubmission.write_template(tfs=['TF1', 'TF2', 'TF3'], 
                              tags=['uniq1', 'uniq2', 'uniq3', 'uniq4'],
                              path=Path("sub_template.txt"))
    sub = ScoreSubmission.template(tfs=['TF1', 'TF2', 'TF3'], 
                              tags=['uniq1', 'uniq2', 'uniq3', 'uniq4'])
    sub['TF1']['uniq1'] = 5
    sub['TF2']['uniq2'] = 0.12
    sub.write("sub_template.txt")
    try:
        sub.prepare_for_evaluation()
    except Exception as exc:
        print(exc)
    for t in sub.tags:
        sub['TF3'][t] = 0.5
    sub = sub.prepare_for_evaluation()
    sub.write(Path("sub.txt"))
    sub = ScoreSubmission.load("sub.txt")
    print(sub)
    sub = ScoreSubmission.load("sub_template.txt")
    print(sub)