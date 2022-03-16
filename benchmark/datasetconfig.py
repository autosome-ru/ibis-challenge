from protocol import  ProtocolType, WeirauchProtocol
from exceptions import WrongExperimentTypeException
from utils import  auto_convert
from attrs import define, fields
from pathlib import Path
from pbm import PBMExperiment
from experiment import ExperimentType, CurationStatus
from dataset import DatasetType, Dataset
from typing import Sequence

@define(field_transformer=auto_convert)
class DatasetConfig:
    name: str
    exp_type: ExperimentType
    motif: str
    ds_type: DatasetType
    path: Path
    curation_status: CurationStatus
    protocol: ProtocolType
    metainfo: dict
    
    @classmethod
    def from_dict(cls, dt: dict):
        names = {f.name for f in fields(cls)}
        args = {}
        metainfo = {}
        for key, value in dt.items():
            if key in names:
                args[key] = value
            else:
                metainfo[key] = value
        if "metainfo" not in args:
            args['metainfo'] = metainfo
        else:
            args['metainfo'].update(metainfo)
        return cls(**args)

    def get_exp_metainfo(self):
        metainfo = self.metainfo.copy()
        metainfo['curation_status'] = self.curation_status
        return metainfo

    def infer_experiment_cls(self):
        if self.exp_type is ExperimentType.PBM:
            cls = PBMExperiment
        elif self.exp_type is ExperimentType.ChIPSeq:
            raise NotImplementedError()
        else:
            raise WrongExperimentTypeException(f"Wrong experiment type: {self.exp_type}")
        return cls

    def infer_protocol_cls(self):
        if self.protocol is ProtocolType.WEIRAUCH:
            return WeirauchProtocol
        else:
            raise WrongExperimentTypeException(f"Wrong protocol type: {self.protocol}")

    def make(self) -> Sequence[Dataset]:
        exp_cls = self.infer_experiment_cls()
        metainfo = self.get_exp_metainfo()
        experiment = exp_cls.read(self.name,
                                  self.path,
                                  self.motif,
                                  metainfo)
        prc_cls = self.infer_protocol_cls()
        protocol = prc_cls()
        return protocol.process(experiment, self.ds_type)