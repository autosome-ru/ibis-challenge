from examples import PBM_EXAMPLE_PATH
from pbm import PBMExperiment

exp = PBMExperiment.read(PBM_EXAMPLE_PATH)
rec = exp.records[0]
print(rec)

dataset = exp.weirauch_protocol()
print(dataset.entries[0])
