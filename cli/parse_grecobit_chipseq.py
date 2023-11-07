
import argparse
import pandas as pd
import glob 

from pathlib import Path
import sys 


sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

from bibis.chipseq.peakfile import ChIPPeakList

from bibis.chipseq.config import ChipSeqConfig, ChipSeqSplit, ForeignConfig, GenomeSampleConfig, ShadesConfig
from bibis.ibis_utils import ibis_default_name_parser

def extract_files(dir, row, parser=ibis_default_name_parser()):
    files = glob.glob(str(dir / f"{row.tf}.*"))
    if len(files) == 0:
        raise Exception(f"No files found for tf: {row.tf}")
    info = {}
    for fl in files:
        if res := parser.parse(Path(fl).name):
            name = res["name"] #type: ignore
            if name not in row.replics:
                print(f"Skipping replic {name} for {row.tf} as it wasn't specified in the table", file=sys.stderr)
                continue
            info[name] = fl
        else:
            raise Exception("Wrong naming former")
    if len(info) != len(row.replics):
        raise Exception(f"No files found for replics: {set(row.replics) - set(info)}")
    return info 

def merge_files(train_files, valid_files, dest_dir):
    merged_info = {}
    for name, train_fl in train_files.items():
        val_fl = valid_files[name]
          
        dest_path = dest_dir / f"{name}.peaks"
        with open(dest_path, "w") as out:
            with open(train_fl, 'r') as inp:
                for line in inp:
                    print(line, file=out, end="")
            with open(val_fl, "r") as inp:
                for line in inp:
                    print(line, file=out, end="")
        merged_info[name] = str(dest_path)
    return merged_info

def process_row(row, train_dir, valid_dir, out_dir):
    train_files = extract_files(train_dir, row)
    valid_files = extract_files(valid_dir, row)
    dest_dir = out_dir / row.stage / row.tf
    dest_dir.mkdir(exist_ok=True, parents=True)
    replics_info = merge_files(train_files=train_files,
                valid_files=valid_files,
                dest_dir=dest_dir) 
    
    return replics_info
    

LEADERBOARD_EXCEL = "/home_local/dpenzar/IBIS TF Selection - Nov 2022 _ Feb 2023.xlsx"
SPLIT_SHEET_NAME = "v3 TrainTest marked (2023)"
ILYA_DIR = Path("/home_local/vorontsovie/greco-data/release_8d.2022-07-31/full/CHS/")
TRAIN_INT = ILYA_DIR / "Train_intervals"
VAL_INT = ILYA_DIR / "Val_intervals"
OUT_DIR = Path("/home_local/dpenzar/BENCH_FULL_DATA/CHS")


parser = argparse.ArgumentParser()
parser.add_argument("--genome",
                    help="Genome fasta",
                    type=str,
                    required=True)
parser.add_argument("--black_list_regions", 
                    required=False, 
                    default=None,
                    type=str)
parser.add_argument("--valid_hide_regions", 
                    required=True, 
                    type=str)
parser.add_argument('--train_chroms',
                    help="Chromosoms, for training", 
                    nargs="+", 
                    type=str,
                    default=[f"chr{ind*2+1}" for ind in range(11)])
parser.add_argument('--valid_chroms',
                    help="Chromosoms, for validation", 
                    nargs="+", 
                    type=str,
                    default=[f"chr{ind*2 + 2}" for ind in range(11)])
parser.add_argument("--seqsize", 
                    type=int,
                    default=301)
parser.add_argument("--shades_balance", 
                    type=int, 
                    default=1)
parser.add_argument("--foreign_balance", 
                    type=int,
                    default=2)
parser.add_argument("--genome_balance",
                    type=int, 
                    default=2)
parser.add_argument("--genome_max_overlap", 
                    type=int,
                    default=0)
parser.add_argument("--shades_max_dist", 
                    type=int,
                    default=300)
parser.add_argument("--exact_genome", 
                    action="store_true")
parser.add_argument("--seed", 
                    type=int, 
                    default=777)
parser.add_argument("--n_procs",
                    type=int,
                    default=1)


args = parser.parse_args()
print(args)
    

ibis_table = pd.read_excel(LEADERBOARD_EXCEL, sheet_name=SPLIT_SHEET_NAME)
ibis_table = ibis_table[['Transcription factor', 'CHS', 'ChIP-Seq', 'Stage']]
ibis_table.columns = ['tf', 'replics', 'split', 'stage']
ibis_table['replics'] = ibis_table['replics'].str.split(',')


for stage in ('Final', 'Leaderboard'):
    stage_table = ibis_table[ibis_table.stage == stage]
    stage_info = {}
    for ind, row in stage_table.iterrows():
        tf_info = process_row(row, 
                    train_dir=TRAIN_INT,
                    valid_dir=VAL_INT, 
                    out_dir=OUT_DIR / 'data')
        stage_info[row.tf] = tf_info
    
    
        
    configs_dir = OUT_DIR / "configs" / stage
    configs_dir.mkdir(exist_ok=True, parents=True)
    
    
    test_files = {}
    to_save: dict[str,  tuple[Path, ChipSeqConfig]] = {}
    
    assert len(stage_table) == len(set(row.tf for _, row in stage_table.iterrows()))
    
    for ind, row in stage_table.iterrows():
        tf = row.tf
        replics = stage_info[tf]        
        paths = list(replics.values())
        
        tf_peaks = [ChIPPeakList.read(t) for t in paths]
        tf_beds = [f.to_beddata() for f in tf_peaks]
            
        if row.split == "Train":
            splits = {"train": ChipSeqSplit(paths=paths,
                                            chroms=args.train_chroms,
                                            hide_regions=None)}
        elif row.split == "Test":
            ind, _ = max(enumerate(tf_beds), key=lambda x: len(x[1]))
            test_peak_path = paths.pop(ind)
            test_files[tf] = test_peak_path
            splits = {"test": ChipSeqSplit(paths=[test_peak_path],
                                           chroms=args.valid_chroms,
                                           hide_regions=args.valid_hide_regions)}
           
            
        elif row.split == "Train/Test":
            ind, _ = max(enumerate(tf_beds), key=lambda x: len(x[1]))
            test_peak_path = paths.pop(ind)
            train_peaks_paths = paths
            test_files[tf] = test_peak_path
            splits = {"train": ChipSeqSplit(paths=train_peaks_paths,
                                            chroms=args.train_chroms,
                                            hide_regions=None),
                      "test": ChipSeqSplit(paths=[test_peak_path],
                                           chroms=args.valid_chroms,
                                           hide_regions=args.valid_hide_regions)}
        else:
            raise Exception("Wrong split: {row.split}")
        
        config = ChipSeqConfig(tf_name=tf,
                               splits=splits,
                               black_list_path=args.black_list_regions,
                               friends_path=[],
                               window_size=args.seqsize,
                               genome_path=args.genome,
                               seed=args.seed,
                               shades_cfg=ShadesConfig(balance=args.shades_balance,
                                                            max_dist=args.shades_max_dist),
                               foreign_cfg=ForeignConfig(balance=args.foreign_balance,
                                                            foreigns_path=[]), # For now we can't set foreigns without data leakage
                               genome_sample_cfg=GenomeSampleConfig(balance=args.genome_balance,
                                                                        max_overlap=args.genome_max_overlap,
                                                                        n_procs=args.n_procs,
                                                                        exact=args.exact_genome,
                                                                        precalc_profile=False))
        path = configs_dir / f"{tf}.json"
        to_save[tf] = (path, config)
        
    for tf, (path, config) in to_save.items():        
        config.foreign_cfg.foreigns_path = [path for other_tf, path in test_files.items() if other_tf != tf]
        config.save(path)
