import json 

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from .dataset import  HTSRawDataset

@dataclass
class HTSRawConfig:
    tf_name: str
    tf_id: int
    stage: str
    stage_id: int
    splits: dict[str, dict[int, list[HTSRawDataset]]] 

    def save(self, path: str | Path):
        dt = asdict(self)
        with open(path, "w") as out:
            json.dump(obj=dt,
                      fp=out,
                      indent=4)
            
    @classmethod
    def load(cls, path: str | Path):
        with open(path, "r") as inp:
            dt = json.load(inp)
        restored_splits = {}
        for split, split_info in dt['splits'].items():
            res_split = {}
            for rep, rep_info in split_info.items():
                rep_dt = {}
                for cycle, ds in rep_info.items():
                    rep_dt[cycle] = HTSRawDataset(**ds)
                res_split[rep] = rep_dt
            restored_splits[split] = res_split
        dt['splits'] = restored_splits
        return cls(**dt)

    
def split_traintest(datasets: list[HTSRawDataset]):
    groups = defaultdict(lambda: defaultdict(dict))

    for ds in datasets:
        groups[ds.exp_tp][ds.rep][ds.cycle] = ds

    train_datasets = {}
    test_datasets = {}

    cur = test_datasets
    nxt = train_datasets
    for _, rep_dt in groups.items():
        for rep, ds_cycles in rep_dt.items():
            cur[rep] = ds_cycles
            cur, nxt = nxt, cur
    return train_datasets, test_datasets

def split_by_rep(datasets: list[HTSRawDataset]):
    spl = defaultdict(dict)
    for ds in datasets:
        spl[ds.rep][ds.cycle] = ds
    return dict(spl)

def split_datasets(datasets: list[HTSRawDataset], split: str) -> dict[str, list[HTSRawDataset]]:
    if split == "Train":
        ds_split = {'train': split_by_rep(datasets)}
    elif split == "Train/Test":
        train_datasets, test_datasets = split_traintest(datasets)
        ds_split = {'train': train_datasets,
                    'test': test_datasets}
    elif split == "Test":
        ds_split = {'test': split_by_rep(datasets)}
    else:
        raise Exception(f"Wrong dataset split: {split}")
    return ds_split 