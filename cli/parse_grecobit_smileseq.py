
import argparse
import pandas as pd
import sys 
import tqdm 
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
parser.add_argument("--log_path",
                    default='log.txt')
parser.add_argument("--logger_name",
                    default="parse_sms")

args = parser.parse_args()
sys.path.append(args.bibis_root)

from bibis.sms.config import RAW_SMSConfig, SMSRawConfig, split_datasets
from bibis.sms.dataset import SMSRawDataset
from bibis.utils import merge_fastqgz
from bibis.logging import get_logger, BIBIS_LOGGER_CFG

BIBIS_LOGGER_CFG.set_path(path=args.log_path)
logger = get_logger(name=args.logger_name, path=args.log_path)

LEADERBOARD_EXCEL = "/home_local/dpenzar/IBIS TF Selection - Nov 2022 _ Feb 2023.xlsx"
SPLIT_SHEET_NAME = "v3 TrainTest marked (2023)"
UNPUBLISHED_SMS_DIR = Path("/home_local/vorontsovie/greco-data/release_8d.2022-07-31/full/SMS")
PUBLISHED_SMS_DIR = Path("/home_local/vorontsovie/greco-data/release_8d.2022-07-31/full/SMS.published")
STAGES = ('Final', 'Leaderboard')

OUT_DIR = Path("/home_local/dpenzar/BENCH_FULL_DATA/SMS/RAW/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Reading ibis metainfo for SMS data")
ibis_table = pd.read_excel(LEADERBOARD_EXCEL, sheet_name=SPLIT_SHEET_NAME)
ibis_table = ibis_table[['Transcription factor', 'SMS', 'SMiLE-Seq', 'Stage']]
ibis_table.columns = ['tf', 'replics', 'split', 'stage']
ibis_table['replics'] = ibis_table['replics'].str.split(',')


def get_flanks_from_name(path: str):
    name = path.split("@")[2]
    _, left_flank, right_flank = name.split(sep=".")
    return left_flank, right_flank

if not ibis_table['stage'].isin(STAGES).all():
    raise Exception(f"Some tfs has invalid stage: {ibis_table[~(ibis_table['stage'].isin(STAGES))]}")


test_left_flanks = []
for stage in STAGES:
    logger.info(f"Preparing {stage} sms datasets for furher processing")
    copy_paths = defaultdict(list)
    stage_ibis_table =  ibis_table[ibis_table['stage'] == stage]
    tf2split = {}
    for ind, tf, reps, ibis_split in stage_ibis_table[['tf', 'replics', 'split'] ].itertuples():
        if ibis_split == "-":
            #print(f"Skipping factor tf: {tf}. No split provided")
            continue
        if isinstance(reps, float): # skip tf without pbms
            #print(f"Skipping factor tf: {tf}. Its split ({ibis_split}) is not none but there is no experiments for that factor", file=sys.stderr)
            continue
        tf2split[tf] = ibis_split
        tf_dir = OUT_DIR / tf 
        tf_dir.mkdir(parents=True, exist_ok=True)
        for rep in reps:
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
            out_path = tf_dir / f"{rep}.fastq.gz"
            copy_paths[tf].append( (rep, path_cands, out_path, left_flank, right_flank))

    configs: list[SMSRawConfig] = []
    for tf, exps in tqdm.tqdm(copy_paths.items()):
        ibis_split = tf2split[tf]
        datasets = []
        for rep, path_cands, out_path, left_flank, right_flank in exps:
            ds_size = merge_fastqgz(in_paths=path_cands,
                          out_path=out_path)
            ds = SMSRawDataset(path=str(out_path),
                               size=ds_size,
                               left_flank=left_flank, 
                               right_flank=right_flank,
                               rep=rep)
            datasets.append(ds)

        cfg = SMSRawConfig(tf_name=tf,
                            splits=split_datasets(datasets=datasets, 
                                                  split=ibis_split))
        cfg_path = OUT_DIR / "{tf}.cfg" 
        configs.append(cfg)

    configs_dir = OUT_DIR / "configs" / stage
    configs_dir.mkdir(exist_ok=True, parents=True)
    for cfg in configs:
        cfg_path = configs_dir /  f"{cfg.tf_name}.json"
        cfg.save(cfg_path)