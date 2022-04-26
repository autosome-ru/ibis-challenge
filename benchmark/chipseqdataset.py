from typing import Union
from pathlib import Path
from benchmark.labels import BinaryLabel

from dataset import Dataset, DatasetType
from seqentry import SeqEntry
from sequence import Sequence

class ChIPSeqDataset(Dataset):
    def get_train_fields(self) -> list[str]:
        return [self.TAG_FIELD, self.SEQUENCE_FIELD,  self.LABEL_FIELD]
    
    def get_test_fields(self):
         return [self.TAG_FIELD, self.SEQUENCE_FIELD]
   
    def to_canonical_format(self, path):
        return self.to_tsv(path)

    @classmethod
    def load(cls, path: Union[Path, str], tp: DatasetType, fmt: str) -> 'ChIPSeqDataset':
        if isinstance(path, str):
            path = Path(path)
        # TODO: fix saving dataset (storing name, tf_name and metainfo in file)
        # TODO: add other fmts, maybe using class abstraction
        if fmt != ".tsv":
            raise NotImplementedError()
        
        self = cls(name="", tf_name="", entries=[], type=tp, metainfo={})
        fieldsname = self.get_fields()
        with path.open() as inp:
            _ = inp.readline() # header
            for line in inp:
                fields = line.split()
                dt = dict(zip(fieldsname, fields))
                sequence = dt.pop(cls.SEQUENCE_FIELD)
                tag = dt.pop(cls.TAG_FIELD)
                label = dt.pop(cls.LABEL_FIELD, None)
                if label is not None:
                    label = BinaryLabel(label)
                entry = SeqEntry(sequence=Sequence(sequence), tag=tag, label=label)  
                self.entries.append(entry)
        return self