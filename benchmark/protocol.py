from atexit import register
from enum import Enum
import numpy as np
from dataset import Dataset, PBMDataset, DatasetType
from labels import BinaryLabel
from typing import ClassVar, Sequence
from pbm import PBMExperiment, PBMPreprocessing
from experiment import Experiment, ExperimentType
from utils import register_enum, singledispatchmethod
from seqentry import SeqEntry
from exceptions import ProtocolException

@register_enum
class ProtocolType(Enum):
    WEIRAUCH = "weirauch"
    IRIS = "iris"


class Protocol:
    def process(self, exp: Experiment, ds_type: DatasetType) -> Sequence[Dataset]:
        raise NotImplementedError(
            f"processing is not defined for experiment {type(exp)} in {type(self)}")

class WeirauchProtocol(Protocol):
    @singledispatchmethod
    def process(self, 
                exp: Experiment, 
                ds_type: DatasetType) -> Sequence[Dataset]:
        return super().process(exp, ds_type)

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
                    exp: PBMExperiment, 
                    ds_type: DatasetType) -> Sequence[Dataset]:
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

        return (PBMDataset(exp.name,
                          ds_type,
                          exp.tf,
                          entries, 
                          metainfo),)

class IbisProtocol(Protocol):
    ZScoreThreshold: ClassVar[float] = 4.0

    @singledispatchmethod
    def process(self, 
                exp: Experiment, 
                ds_type: DatasetType) -> Sequence[Dataset]:
        return super().process(exp, ds_type)

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
        th = min(th1, th2)
        th3 = vals[-max_probs]
        th = max(th, th3)
        return th 

    @process.register
    def process_pbm(self, 
                    exp: PBMExperiment, 
                    ds_type: DatasetType) -> Sequence[Dataset]:
        if exp.preprocessing is PBMPreprocessing.SD:
            threshold = self.std_threshold(exp)
        elif exp.preprocessing is PBMPreprocessing.QNZS:
            threshold = 4
        else:
            raise ProtocolException(
                     f"Protocol {type(self).__name__} is not implemented for {exp.preprocessing.value}")
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

        return (PBMDataset(exp.name,
                          ds_type,
                          exp.tf,
                          entries, 
                          metainfo),)

class ProtocolFactory:
    @staticmethod
    def make(tp: ProtocolType):
        if tp is ProtocolType.WEIRAUCH:
            return WeirauchProtocol()
        if tp is ProtocolType.IRIS:
            return WeirauchProtocol()
        else:
            raise ProtocolException("Protocol {tp.value} isn't implemented")