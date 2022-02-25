from atexit import register
from enum import Enum
import numpy as np
from dataset import PBMDataset
from labels import BinaryLabel
from typing import List
from pbm import PBMExperiment
from experiment import Experiment, ExperimentType
from utils import register_enum, singledispatchmethod
from seqentry import SeqEntry

@register_enum
class ProtocolType(Enum):
    WEIRAUCH = "weirauch"

class Protocol:
    def process(self, exp: Experiment):
        raise NotImplementedError(
            f"processing is not defined for experiment {type(exp)} in {type(self)}")

class WeirauchProtocol(Protocol):
    @singledispatchmethod
    def process(self, exp: Experiment):
        return super().process(exp)

    @staticmethod
    def pbm_threshold(exp: PBMExperiment,
                           min_probs=50,
                           max_probs=1300):
        vals = [r.mean_signal_intensity for r in exp.records]
        vals.sort()
        mean = np.mean(vals)
        std = np.std(vals)
        th1 = mean + 4 * std
        th2 = vals[-min_probs]
        th = min(th1, th2)
        th3 = vals[-max_probs]
        th = max(th, th3)
        return th

    @process.register
    def process_pbm(self, 
                    exp: PBMExperiment) -> PBMDataset:
        threshold = self.pbm_threshold(exp)
        entries = []
        for rec in exp.records:
            if rec.mean_signal_intensity >= threshold:
                label = BinaryLabel.POSITIVE
            else:
                label = BinaryLabel.NEGATIVE
            entry = SeqEntry.from_record(rec, label)
            entries.append(entry)
        
        metainfo = exp.metainfo.copy()
        metainfo['protocol'] = ProtocolType.WEIRAUCH
        metainfo['exp_type'] = ExperimentType.PBM
        metainfo['pbm_type'] = exp.pbm_type
        metainfo['preprocessing'] = exp.preprocessing

        return PBMDataset(exp.name, exp.motif, entries, metainfo)