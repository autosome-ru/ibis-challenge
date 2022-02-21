from __future__ import annotations

import numpy as np

from typing import List
from attrs import define, field
from labels import BinaryLabel
from math import nan
from utils import END_LINE_CHARS
from dataset import Dataset
from seqentry import SeqEntry
from pbmrecord import PBMRecord



@define
class PBMExperiment:
    records: List[PBMRecord] = field(repr=False)
    @staticmethod
    def parse_header(header):
        names = header\
                .lstrip("#")\
                .split('\t')
        return names

    @staticmethod
    def parse_record(names, line, sep):
        values = line.split(sep)
        record = PBMRecord.from_dict(
                    dict(zip(names,values))
                )
        return record

    @classmethod
    def read(cls, path, sep="\t"):
        records = []
        with open(path, "r") as inp:
            header = inp.readline().rstrip(END_LINE_CHARS)
            names = cls.parse_header(header)
            for line in inp:
                line = line.rstrip(END_LINE_CHARS)
                record = cls.parse_record(names, line, sep)
                records.append(record)
        return cls(records)

    def weirauch_threshold(self,
                           min_probs=50,
                           max_probs=1300):
        vals = [r.mean_signal_intensity for r in self.records]
        vals.sort()
        mean = np.mean(vals)
        std = np.std(vals)
        th1 = mean + 4 * std
        th2 = vals[-min_probs]
        th = min(th1, th2)
        th3 = vals[-max_probs]
        th = max(th, th3)
        return th

    def weirauch_protocol(self):
        threshold = self.weirauch_threshold()
        return self.one_threshold_protocol(threshold)
    
    def one_threshold_protocol(self,
                               threshold: float):
        entries = []
        for rec in self.records:
            if rec.mean_signal_intensity >= threshold:
                label = BinaryLabel.POSITIVE
            else:
                label = BinaryLabel.NEGATIVE
            entry = SeqEntry.from_record(rec, label)
            entries.append(entry)

        return Dataset(entries)

    