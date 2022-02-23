from dataset import DatasetMode
from examples import PBM_EXAMPLE_PATH
from pbm import PBMExperiment

exp = PBMExperiment.read(PBM_EXAMPLE_PATH)
dataset = exp.weirauch_protocol()
dataset.to_canonical_format("as_test.txt")
dataset.to_canonical_format("as_train.txt", mode=DatasetMode.TRAIN)
