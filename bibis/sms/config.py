import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .dataset import SMSRawDataset


@dataclass
class SMSRawConfig:
    tf_name: str
    splits: dict[str, list[SMSRawDataset]]

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
        for split, ds_list in dt['splits'].items():
            restored_splits[split] = [SMSRawDataset(**d) for d in ds_list]
        dt['splits'] = restored_splits

        return cls(**dt)

def split_datasets(datasets: list[SMSRawDataset], split: str) -> dict[str, list[SMSRawDataset]]:
    if split == "Train":
        ds_split = {'train': datasets}
    elif split == "Train/Test":
        train_datasets = []
        assign_datasets = []
        for ds in datasets:
            if Path(ds.path).name.startswith("SRR"):
                train_datasets.append(ds)
            else:
                assign_datasets.append(ds)

        test_ind = max( enumerate(assign_datasets), key=lambda x: x[1].size)[0]
        test_datasets = [assign_datasets.pop(test_ind)]
        train_datasets.extend(assign_datasets) 
        
        ds_split = {'train': train_datasets,
                    'test': test_datasets}
    elif split == "Test":
        ds_split = {'test': datasets}
    else:
        raise Exception(f"Wrong dataset split: {split}")
    return ds_split 