
import argparse
from pickletools import read_floatnl
from turtle import right
import pandas as pd
from pathlib import Path
import sys 
import shutil
from dataclasses import dataclass, asdict
import json

@dataclass
class SMSRawDataset:
    path: str
    left_flank: str
    right_flank: str

@dataclass
class SMSConfig:
    tf_name: str
    train_datasets: list[SMSRawDataset] # path to tsvs with train chips
    test_datasets: list[SMSRawDataset] # path to tsvs with test chips

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
        dt['train_datasets'] = [SMSConfig(*d) for d in  dt['train_datasets'] ]
        dt['test_datasets'] = [SMSConfig(*d) for d in  dt['test_datasets'] ]
        return cls(**dt)


sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")
from bibis.pbm.config import PBMConfig

LEADERBOARD_EXCEL = "/home_local/dpenzar/IBIS TF Selection - Nov 2022 _ Feb 2023.xlsx"
SPLIT_SHEET_NAME = "v3 TrainTest marked (2023)"
UNPUBLISHED_SMS_DIR = Path("/home_local/vorontsovie/greco-data/release_8d.2022-07-31/full/SMS")
PUBLISHED_SMS_DIR = Path("/home_local/vorontsovie/greco-data/release_8d.2022-07-31/full/SMS.published")
STAGES = ('Final', 'Leaderboard')

OUT_DIR = Path("/home_local/dpenzar/BENCH_FULL_DATA/SMS/RAW/")


parser = argparse.ArgumentParser()
parser.add_argument("--neg2pos_ratio",
                    type=int, 
                    default=10)
args = parser.parse_args()

ibis_table = pd.read_excel(LEADERBOARD_EXCEL, sheet_name=SPLIT_SHEET_NAME)
ibis_table = ibis_table[['Transcription factor', 'SMS', 'SMiLE-Seq', 'Stage']]
ibis_table.columns = ['tf', 'replics', 'split', 'stage']
ibis_table['replics'] = ibis_table['replics'].str.split(',')

print(ibis_table.head())

def get_flanks_from_name(path: str):
    name = path.split("@")[2]
    _, left_flank, right_flank = name.split(sep=".")
    return left_flank, right_flank


from collections import defaultdict


if not ibis_table['stage'].isin(STAGES).all():
    raise Exception(f"Some tfs has invalid stage: {ibis_table[~(ibis_table['stage'].isin(STAGES))]}")

test_left_flanks = []
for stage in STAGES:
    copy_paths = defaultdict(dict)
    stage_ibis_table =  ibis_table[ibis_table['stage'] == stage]
    

    
    for ind, tf, reps, ibis_split in stage_ibis_table[['tf', 'replics', 'split'] ].itertuples():
        if ibis_split == "-":
            #print(f"Skipping factor tf: {tf}. No split provided")
            continue
        if isinstance(reps, float): # skip tf without pbms
            #print(f"Skipping factor tf: {tf}. Its split ({ibis_split}) is not none but there is no experiments for that factor", file=sys.stderr)
            continue
        #print(ind, tf, reps, ibis_split)
        copy_paths[tf][rep].append(path)

        tf_dir = OUT_DIR / tf 
        for rep in reps:
            rep_dir = tf_dir / rep
            if rep.startswith("SRR"):
                search_dir = PUBLISHED_SMS_DIR
            else:
                search_dir = UNPUBLISHED_SMS_DIR
            path_cands = list(search_dir.glob(f"*/*@{rep}.*"))
            assert (len(path_cands) == 2)
            l_flank_1, r_flank_1 = get_flanks_from_name(str(path_cands[0]))
            l_flank_2, r_flank_2 = get_flanks_from_name(str(path_cands[1]))
            assert l_flank_1 == l_flank_2
            assert r_flank_1 == r_flank_2
            left_flank, right_flank = l_flank_1, r_flank_1
            if "Test" in ibis_split:
                print(left_flank)
                test_left_flanks.append(left_flank)
            

possible_pref_end = len(test_left_flanks[0])
for i in range(1, len(test_left_flanks)):
    print(possible_pref_end)
    for j in range(0, possible_pref_end):
        if test_left_flanks[i][j] != test_left_flanks[i-1][j]:
            break
    else:
        j = len(test_left_flanks[i-1])
    
    possible_pref_end = min(possible_pref_end, j)
print(possible_pref_end)
for flank in test_left_flanks:
    print(flank, flank[:possible_pref_end] + 'N' * (len(flank) - possible_pref_end))


'''
        for rep in reps:
            for norm in ["QNZS", "SD"]:
                norm_dir = PBM_DIR / f"PBM.{norm}" 
                for exp_type, split in EXPTP2SPLIT.items():
                    split_dir = norm_dir / f"{split}_intensities"

                    for path in split_dir.glob(f"*{exp_type}@{rep}*"):
                            if split == "Train":
                                real_split = "train"
                            elif split == "Val":
                                real_split = "test"
                            else:
                                raise Exception(f"Wrong split {split}")
                            
                            if real_split == "train" and ibis_split == "Test":
                                print(f"Skipping file {path} for {tf} as it is from wrong split", file=sys.stderr)
                                continue
                            if real_split == "test" and ibis_split == "Train":
                                print(f"Skipping file {path} for {tf} as it is from wrong split", file=sys.stderr)
                                continue
                            name = path.name
                            exp_type = exp_type.replace("PBM.", "")
                            copy_paths[tf][real_split][norm].append(path)
        for sp in ibis_split.split("/"):
            if sp == "Train":
                if len(copy_paths[tf]['train']) == 0:
                    raise Exception(f"No train experiments for tf {tf} although train split is specified")
            elif sp == "Test":
                if len(copy_paths[tf]['test']) == 0:
                    raise Exception(f"No train experiments for tf {tf} although test split is specified")
            else:    
                raise Exception(f"Wrong split {sp}")

    configs_dir = OUT_DIR / "configs" / stage
    configs_dir.mkdir(exist_ok=True, parents=True)
    
    for tf, split_dt in copy_paths.items():
        cfg = PBMConfig(tf, 
                        train_paths=[],
                        test_paths=[],
                        protocol="ibis", 
                        neg2pos_ratio=args.neg2pos_ratio)
        for split, norm_dt in split_dt.items():
            split = split.lower()
            for norm, in_paths in norm_dt.items():            
                for ind, in_p in enumerate(in_paths, 1):
                    pbm_id = in_p.name.split("@", maxsplit=3)[2]
                    pbm_id = pbm_id.split(".")[0]
                    out_dir = OUT_DIR / tf / split / norm / exp_type
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{pbm_id}.tsv"
                    shutil.copy(in_p, out_path)
                    if split == "train":
                        cfg.train_paths.append(str(out_path))
                    elif split == "test":
                        cfg.test_paths.append(str(out_path))
                    else:
                        raise Exception(f"Wrong split: {split}")
        cfg_path = configs_dir /  f"{tf}.json"
        cfg.save(cfg_path)
'''