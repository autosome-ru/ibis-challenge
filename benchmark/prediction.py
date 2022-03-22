
from typing import Dict, Optional

from attr import define, field
from pathlib import Path
from typing import ClassVar
from exceptions import SubmissionFormatException

@define
class Prediction:
    _pred: Dict[str, float] = field(factory=dict)

    def __getitem__(self, tag: str) -> float:
        return self._pred[tag]

    def get(self, tag: str):
        return self._pred.get(tag, None)

    def __len__(self) -> int:
        return len(self._pred)

    def __setitem__(self, tag:str, score: float):
        self._pred[tag] = score

    @classmethod
    def load(cls, path: Path, sep="\t"):
        with path.open(mode="r") as infile:
            dt = {}
            for line in infile:
                tag, score = line.split(sep)
                score = float(score)
                dt[tag] = score
        return cls(dt)

@define 
class Submission:
    _sub: Dict[str, Prediction]

    TAGFIELDNAME: ClassVar[str] = "tag"
    SKIPFIELDNAME: ClassVar[str] = "nodata"

    def __getitem__(self, tf: str) -> Prediction:
        return self._sub[tf]
    
    def __setitem__(self, tf: str, pred: Prediction):
        self._sub[tf] = pred

    def get(self, tf: str) -> Optional[Prediction]:
        return self._sub.get(tf)

    def __contains__(self, tf: str) -> bool:
        return tf in self._sub
    
    @classmethod
    def load(cls, path: Path, sep="\t"):
        with path.open('r') as infile:
            header = infile.readline()
            fields = header.split(sep)
            if fields[0] != cls.TAGFIELDNAME:
                msg = f'First submission field must be tag field: {cls.TAGFIELDNAME}'
                raise SubmissionFormatException(msg)
            if len(fields) == 1:
                msg = f"Submission must contain at least one tf field"
            tfs = fields[1:]
            _sub = {tf_name: Prediction() for tf_name in tfs}
            has_nodata = {tf_name: False for tf_name in tfs}
            for line in infile:
                vals = line.split(sep)
                tag = vals[0]
                for tf_name, val in zip(tfs, vals[1:]):
                    if val == cls.SKIPFIELDNAME:
                        has_nodata[tf_name] = True
                        continue
                    val = float(val)
                    _sub[tf_name][tag] = val
        for tf_name, nodata in has_nodata.items():
            if nodata:
                print(f"Column for factor {tf_name} contains '{cls.SKIPFIELDNAME}' value." 
                       "Predictions for this factor won't be evaluated")
                _sub.pop(tf_name)
        return cls(_sub)