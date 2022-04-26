from dataset import Dataset
from experiment import ExperimentType
from abc import abstractproperty, abstractmethod, ABCMeta

from typing import List, Union

from datasetconfig import DatasetConfig

class SubProtocol(metaclass=ABCMeta):

    @abstractproperty
    def data_type(self) -> ExperimentType:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, cfgs: List[DatasetConfig]):
        '''
        prepare data given cgfs files of all datasets of specified datatype
        '''
        raise NotImplementedError()

    @abstractmethod
    def process(self, cfg: Union[List[DatasetConfig], DatasetConfig]) -> 'Dataset':
        '''
        create dataset from cfg file using it's data and data aquired during
        preprocess stage. Data from other cfg should not be used 
        '''
        raise NotImplementedError()