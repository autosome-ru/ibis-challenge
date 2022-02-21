import pandas as pd
from sklearn import datasets 
from examples import PBM_EXAMPLE_PATH
from pbm import PBMExperiment, PBMDataset
from attrs import fields



exp = PBMExperiment.read(PBM_EXAMPLE_PATH)
rec = exp.records[0]
print(rec)

#datasets = PBMDataset.weirauch_protocol(exp)

#print(exp)
