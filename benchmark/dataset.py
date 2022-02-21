from typing import List
from seqentry import SeqEntry
from attrs import define 

@define
class Dataset:
    entries: List[SeqEntry]

    def to_tsv(self, path):
        raise NotImplementedError

    def to_json(self, path):
        raise NotImplementedError

    def to_canonical_format(self, path):
        return self.to_tsv(path)