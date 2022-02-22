from typing import List
from utils import undict
from seqentry import SeqEntry
from attrs import define 

@define
class Dataset:
    entries: List[SeqEntry]

    def infer_fields(self):
        fields = set()
        for en in self.entries:
            dt = undict(en.metainfo)
            fields.update(dt.keys())
        fields = list(fields)
        fields = ['seq', 'label'] + fields
        return fields
   
    def to_tsv(self, path, fields=None):
        if fields is None:
            fields = self.infer_fields()
        with open(path, "w") as out:
            header = "\t".join(fields)
            print(header, file=out)
            for en in self.entries:
                values = []
                for fld in fields:
                    if fld == "seq":
                        val = en.seq.seq
                    elif fld == "label":
                        val = en.label.name # type: ignore
                    else:
                        val = en.metainfo.get(fld, "")
                    values.append(str(val))
                print("\t".join(values), file=out)
                break
                    
    def to_json(self, path):
        raise NotImplementedError

    def to_canonical_format(self, path):
        return self.to_tsv(path)