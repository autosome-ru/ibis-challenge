
import argparse
import pandas as pd
import glob 

from pathlib import Path
import sys 
import parse

def ibis_default_name_parser():
    return parse.compile("{tf_name}.{fl}@{exp_type}@{name}@Peaks.{unique_tag}.{dt_type}.peaks")


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",
                    type=str,
                    required=True)
parser.add_argument("--genome",
                    help="Genome fasta",
                    type=str,
                    required=True)
parser.add_argument("--black_list_regions", 
                    required=False, 
                    default=None,
                    type=str)
parser.add_argument("--valid_hide_regions", 
                    required=False,
                    default=None,
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
parser.add_argument("--foreign_balance", 
                    type=int,
                    default=2)
parser.add_argument("--foreign_min_dist", 
                    type=int,
                    default=300)
parser.add_argument("--genome_balance",
                    type=int, 
                    default=2)
parser.add_argument("--genome_min_dist", 
                    type=int,
                    default=300)
parser.add_argument("--genome_max_overlap", 
                    type=int,
                    default=0)
parser.add_argument("--shades_balance", 
                    type=int, 
                    default=1)
parser.add_argument("--shades_min_dist", 
                    type=int,
                    default=300)
parser.add_argument("--shades_max_dist", 
                    type=int,
                    default=600)
parser.add_argument("--exact_genome", 
                    action="store_true")
parser.add_argument("--seed", 
                    type=int, 
                    default=777)
parser.add_argument("--n_procs",
                    type=int,
                    default=1)
parser.add_argument("--log_path",
                    default='log.txt')
parser.add_argument("--logger_name",
                    default="parse_chipseq")

args = parser.parse_args()

from bibis.peaks.peakfile import PeakList

from bibis.peaks.config import PeakSeqConfig, PeakSeqSplit, ForeignConfig, GenomeSampleConfig, ShadesConfig

from bibis.logging import get_logger, BIBIS_LOGGER_CFG
BIBIS_LOGGER_CFG.set_path(path=args.log_path)
logger = get_logger(name=args.logger_name, path=args.log_path)


def log_splits(cfg: PeakSeqConfig, splits: list[str]=None):
    if splits is None:
        splits = ['train', 'test']

    for split in splits:
        split_inst = cfg.splits.get(split)
        if split_inst is None:
            logger.info(f"For factor {cfg.tf_name} no replics are going to {split}")
        else:
            reps = ", ".join(split_inst.reps.keys())
            logger.info(f"For factor {cfg.tf_name} the following replics are going to {split}: {reps}")    


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
OUT_DIR = Path(args.out_dir)

logger.info("Reading ibis metainfo for ChIPSeq")
ibis_table = pd.read_excel(LEADERBOARD_EXCEL, sheet_name=SPLIT_SHEET_NAME)
ibis_table = ibis_table[['Transcription factor', 'CHS', 'ChIP-Seq', 'Stage']]
ibis_table.columns = ['tf', 'replics', 'split', 'stage']
ibis_table['replics'] = ibis_table['replics'].str.split(',')

for stage in ('Leaderboard', 'Final'):
    logger.info(f"Spliting {stage} datasets and writing configs")
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
    to_save: dict[str,  tuple[Path, PeakSeqConfig]] = {}
    
    assert len(stage_table) == len(set(row.tf for _, row in stage_table.iterrows()))
    
    for ind, row in stage_table.iterrows():
        tf = row.tf
        replics = stage_info[tf]
        rep_names = list(replics.keys())                
        tf_peaks = [PeakList.read(t) for t in replics.values()]
        tf_beds = [f.to_beddata() for f in tf_peaks]
            
        if row.split == "Train":
            splits = {"train": PeakSeqSplit(reps=replics,
                                            chroms=args.train_chroms,
                                            hide_regions=None)}
        elif row.split == "Test":
            ind, _ = max(enumerate(tf_beds), key=lambda x: len(x[1]))
            test_rep = rep_names.pop(ind)
            test_files[tf] = replics[test_rep]
            splits = {"test": PeakSeqSplit(reps={test_rep: replics[test_rep]},
                                           chroms=args.valid_chroms,
                                           hide_regions=args.valid_hide_regions)}
        elif row.split == "Train/Test":
            ind, _ = max(enumerate(tf_beds), key=lambda x: len(x[1]))
            test_rep = rep_names.pop(ind)
            test_files[tf] = replics[test_rep]
            train_replics = {rep: replics[rep] for rep in rep_names}
            splits = {"train": PeakSeqSplit(reps=train_replics,
                                            chroms=args.train_chroms,
                                            hide_regions=None),
                      "test": PeakSeqSplit(reps={test_rep: replics[test_rep]},
                                           chroms=args.valid_chroms,
                                           hide_regions=args.valid_hide_regions)}
        else:
            raise Exception("Wrong split: {row.split}")
        
        config = PeakSeqConfig(tf_name=tf,
                               splits=splits,
                               black_list_path=args.black_list_regions,
                               friends_path=[],
                               window_size=args.seqsize,
                               genome_path=args.genome,
                               seed=args.seed,
                               shades_cfg=ShadesConfig(balance=args.shades_balance,
                                                       min_dist=args.shades_min_dist,
                                                       max_dist=args.shades_max_dist),
                               foreign_cfg=ForeignConfig(balance=args.foreign_balance,
                                                         min_dist=args.foreign_min_dist,
                                                         foreigns_path=[]), # For now we can't set foreigns without data leakage
                               genome_sample_cfg=GenomeSampleConfig(balance=args.genome_balance,
                                                                    min_dist=args.genome_min_dist,
                                                                    max_overlap=args.genome_max_overlap,
                                                                    n_procs=args.n_procs,
                                                                    exact=args.exact_genome,
                                                                    precalc_profile=False))
        log_splits(config)
        path = configs_dir / f"{tf}.json"
        to_save[tf] = (path, config)
        
    for tf, (path, config) in to_save.items():        
        config.foreign_cfg.foreigns_path = [path for other_tf, path in test_files.items() if other_tf != tf]
        config.save(path)
