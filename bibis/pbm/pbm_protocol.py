import numpy as np

from typing import ClassVar, Sequence
from .pbm import PBMExperiment, PBMPreprocessing
from ..seq.seqentry import SeqEntry
from Bio.Seq import Seq



class IbisProtocol:
    ZScoreThreshold: ClassVar[float] = 4.0

    @staticmethod
    def std_threshold(exp: PBMExperiment, 
                           min_probs=50,
                           max_probs=1300):
        vals = [r.mean_signal_intensity for r in exp.records]
        vals.sort()
        mean = np.mean(vals)
        std = np.std(vals)
        th1 = mean + 4 * std
        th2 = vals[-min_probs]
        th = min(th1, th2) # type: ignore
        th3 = vals[-max_probs]
        th = max(th, th3)
        return th 


    def process_pbm(self, 
                    exp: PBMExperiment, 
                    preprocessing: str) -> tuple[list[SeqEntry], list[SeqEntry]]:
        if preprocessing == "SD":
            threshold = self.std_threshold(exp)
        elif preprocessing  == "QNZS":
            threshold = 4
        else:
            raise Exception(
                     f"Protocol {type(self).__name__} is not implemented for {preprocessing}")
        
        pos_entries = []
        neg_entries = []
        
        for rec in exp.records:
            if rec.mean_signal_intensity >= threshold:
                label = 1
            else:

                label = 0
            entry = rec.to_seqentry(label=label)
        
            if label == 1:
                pos_entries.append(entry)
            else:
                neg_entries.append(entry)
        return pos_entries, neg_entries

