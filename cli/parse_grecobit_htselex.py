import argparse
import pandas as pd
from pathlib import Path
import sys
import tqdm 
import random
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--recalc", action="store_true")
parser.add_argument("--calc_dir", help="Dir for calculation mapreduce operations", default="mapreduce_calc")
parser.add_argument("--n_procs", type=int, default=20)
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--bibis_path", default='/home_local/dpenzar/bibis_git/ibis-challenge')
parser.add_argument("--log_path",
                    default='log.txt')
parser.add_argument("--logger_name",
                    default="parse_hts")

args = parser.parse_args()

sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

from bibis.utils import END_LINE_CHARS
from bibis.counting.fastqcounter import FastqGzReadsCounter
from bibis.hts.config import HTSRawConfig, split_datasets
from bibis.hts.dataset import HTSRawDataset
from bibis.hts.seqentry import SeqAssignEntry
from bibis.seq.utils import gc
from bibis.sampling.reservoir import PredefinedIndicesSelector
from bibis.logging import get_logger, BIBIS_LOGGER_CFG

BIBIS_LOGGER_CFG.set_path(path=args.log_path)
logger = get_logger(name=args.logger_name, path=args.log_path)
    

def get_cycle_flanks_from_name(path: str):
    name = path.split("@")[2]
    _, cycle, left_flank, right_flank = name.split(sep=".")
    return cycle, left_flank, right_flank

def parse_ibis_table(path, sheet_name):
    ibis_table = pd.read_excel(path, sheet_name)
    ibis_table = ibis_table[['Transcription factor', 'HTS-GFPIVT', 'HTS-IVT', 'HTS-LYS', 'HT-SELEX', 'Stage']]
    
    tps = ('HTS-GFPIVT', 'HTS-IVT', 'HTS-LYS', 'HT-SELEX')
    ibis_table.columns = ['tf', 'replics1', 'replics2',  'replics3', 'split', 'stage']
    rep2typemap = {}
    replics = []
    for _, rep1, rep2, rep3 in ibis_table[[ 'replics1', 'replics2',  'replics3']].itertuples():
        rep = []
        for reps, tp in zip((rep1, rep2, rep3), tps):
            if not isinstance(reps, float):
                cur_rs = reps.split(",")
                for r in cur_rs:
                    rep2typemap[r] = tp
                rep.extend(cur_rs)
        replics.append(rep)
    ibis_table['replics'] = replics
    ibis_table  = ibis_table[['tf', 'replics', 'split', 'stage']]
    return ibis_table, rep2typemap

def extract_info(ibis_table, stages):
    all_nonzero_paths = [] 
    stage_wise_copy_paths = {}
    stage_wise_tf2split = {}
    for stage in stages:
        copy_paths = defaultdict(list)
        stage_ibis_table =  ibis_table[ibis_table['stage'] == stage]
        tf2split = {}
        for ind, tf, reps, ibis_split in stage_ibis_table[['tf', 'replics', 'split'] ].itertuples():
            if ibis_split == "-":
                logger.warning(f"Skipping factor tf: {tf}. No split provided")
                continue
            if isinstance(reps, float): # skip tf without pbms
                logger.warning(f"Skipping factor tf: {tf}. Its split ({ibis_split}) is not none but there is no experiments for that factor", file=sys.stderr)
                continue
            tf2split[tf] = ibis_split
            for rep in reps:
                for cycle in ('C1', 'C2', 'C3', 'C4'):
                    path_cands = list(HTS_DIR.glob(f"*/*@{rep}.{cycle}.*"))
                    if len(path_cands) == 0:
                        logger.info(f'No {cycle} found for replic {rep} for tf {tf}')
                        continue
                    elif len(path_cands) != 2:
                        raise Exception('Unexpected data')
                    all_nonzero_paths.extend(path_cands)
                    cycle1, l_flank_1, r_flank_1 = get_cycle_flanks_from_name(str(path_cands[0]))
                    cycle2, l_flank_2, r_flank_2 = get_cycle_flanks_from_name(str(path_cands[1]))
                    assert cycle == cycle1 and cycle1 == cycle2
                    assert l_flank_1 == l_flank_2
                    assert r_flank_1 == r_flank_2
                    left_flank, right_flank = l_flank_1, r_flank_1
                    copy_paths[tf].append( (rep, cycle, path_cands, left_flank, right_flank))
        stage_wise_copy_paths[stage] = copy_paths
        stage_wise_tf2split[stage] = tf2split

    all_nonzero_paths = sorted([str(x) for x in all_nonzero_paths])
    return stage_wise_copy_paths, stage_wise_tf2split, all_nonzero_paths

def count_seqs(counter_dir: Path,
                counter_config: Path,
                zeros_paths, 
                nonzero_paths,
                counts_path: Path,
                recalc: bool,
                indices_sep: str):
    
    if counter_config.exists() and not args.recalc:
        counter = FastqGzReadsCounter.load(counter_config)
    else:
        counter = FastqGzReadsCounter.create(mapdir=counter_dir, n_jobs=20)

    counter.add(zeros_paths+nonzero_paths)
    counter.dump(counter_config)
    zero_set = set(s for s in zeros_paths)
   
    assert indices_sep != counter.FIELD_SEP

    if not htselex_counts_path.exists() or recalc:
        def reduce_store(grp):
            exist = set()
            for e in grp:
                file_ind = e.file_ind
                name = counter.index[file_ind]
                if name in zero_set:
                    exist.add(-1)
                else:
                    exist.add(file_ind)
            s = indices_sep.join(map(str, exist))
            return s 

        counter.reduce(counts_path, reduce_fn=reduce_store)
    else:
        logger.info("Skipping sequence counting as file exists and recalc flag is not specified")
    return counter 

def restore_int_keys(dt):
    int_dt = {}
    for key, value in dt.items():
        key = int(key)
        if isinstance(value, dict):
            value = restore_int_keys(value)
        int_dt[key] = value
    return int_dt

def assign_seqs(counter,
                counts_path,
                stage_wise_copy_paths, 
                assign_path, 
                meta_info_path: str | Path,
                recalc: bool, 
                indices_sep=" "):
    
  
    if not recalc and Path(meta_info_path).exists():
        logger.info("Skipping sequences assigning as all files already exist and recalc is not specified")

        with open(meta_info_path) as inp:
            meta = json.load(inp)
        tf2id = meta['tf2id']
        rep2id = meta['rep2id']
        stage2id = meta['stage2id']
        ds_sizes = restore_int_keys(meta['ds_sizes'])
        
        zero_size = meta['zero_size']
        return tf2id, rep2id, stage2id, ds_sizes, zero_size    

    reverse_index = {p: x for x, p in enumerate(counter.index)}
    tf2id = {}
    rep2id = {}
    stage2id = {}

    stage_mapping = {}
    tf_mapping = {}
    rep_mapping = {}
    cycle_mapping = {}

    for stage, copy_paths in stage_wise_copy_paths.items():
        if stage not in stage2id:
            stage2id[stage] = len(stage2id)
        for tf, exps in tqdm.tqdm(copy_paths.items()):
            if tf not in tf2id:
                tf2id[tf] = len(tf2id)
              
            for rep, cycle, path_cands, _, _ in exps:
                if rep not in rep2id:
                    rep2id[rep] = len(rep2id)
                for path in path_cands:
                    ix = reverse_index[str(path)]
                    stage_mapping[ix] = stage2id[stage]
                    tf_mapping[ix] = tf2id[tf]
                    rep_mapping[ix] = rep2id[rep]
                    cycle_mapping[ix] = int(cycle.replace('C', ''))

    ds_sizes = defaultdict(lambda: defaultdict(int))

    zero_size = 0
    with open(counts_path, 'r') as inp, open(assign_path, "w") as out:
        for line in tqdm.tqdm(inp):
            seq, occs = line.strip(END_LINE_CHARS).split(counter.FIELD_SEP)
            occs = set(map(int, occs.split(indices_sep)))
            if len(occs) > 1: # non-unique read
                continue
            ind = occs.pop()
            if ind != -1: # read from non-zero cycle 
                rep_ind = rep_mapping[ind]
                tf_ind = tf_mapping[ind]
                stage_ind = stage_mapping[ind]
                cycle = cycle_mapping[ind]
                ds_sizes[rep_ind][cycle] += 1
            else: # read from zero cycle
                rep_ind = -1 
                tf_ind = -1
                cycle = 0
                stage_ind = -1 # unfinished file, stage for zero cycles is not assigned 
                zero_size += 1
            entry = SeqAssignEntry(seq=seq,
                                   cycle=cycle,
                                   rep_ind=rep_ind,
                                   tf_ind=tf_ind,
                                   stage_ind=stage_ind,
                                   gc_content=gc(seq))
            print(entry.to_line(),
                  file=out)
    meta = {}
    meta['tf2id'] = tf2id
    meta['rep2id'] = rep2id
    meta['stage2id'] = stage2id
    meta['ds_sizes'] = ds_sizes
    meta['zero_size'] = zero_size
    with open(meta_info_path, "w") as out:
        json.dump(meta, out)

    return tf2id, rep2id, stage2id, ds_sizes, zero_size

def write_flanks(datasets: list[HTSRawDataset], flanks_path: str):
    dsid2flank = defaultdict(dict)
    for ds in datasets:
        dsid2flank[ds.rep_id][ds.cycle] = [ds.left_flank, ds.right_flank]
    dsid2flank = dict(dsid2flank)
    with open(flanks_path, "w") as out:
        json.dump(dsid2flank, out, indent=4)
    return dsid2flank

def log_splits(cfg: HTSRawConfig, splits: list[str]=None):
    if splits is None:
        splits = ['train', 'test']

    for split in splits:
        datasets = cfg.splits.get(split)
        if datasets is None:
            logger.info(f"For factor {cfg.tf_name} no replics are going to {split}")
        else:
            reps = ", ".join(datasets.keys())
            logger.info(f"For factor {cfg.tf_name} the following replics are going to {split}: {reps}")    
    

def split_data(stage_wise_copy_paths, 
                  stage_wise_tf2split,
                  tf2id,
                  rep2id,
                  stage2id,
                  ds_sizes,
                  rep2typemap):
    all_configs = {}
    for stage, copy_paths in stage_wise_copy_paths.items():
        configs = []
        tf2split = stage_wise_tf2split[stage]
    

        for tf, exps in tqdm.tqdm(copy_paths.items()):
            ibis_split = tf2split[tf]
            datasets = []
            for rep, cycle, path_cands, left_flank, right_flank in exps:
                exp_tp = rep2typemap[rep]
                rep_id = rep2id[rep]
                cycle_id = int(cycle.replace('C', ''))
                ds = HTSRawDataset(
                                rep_id=rep_id,
                                size=ds_sizes[rep_id][cycle_id],
                                cycle=cycle_id,
                                rep=rep,
                                exp_tp=exp_tp,
                                left_flank=left_flank,
                                right_flank=right_flank,
                                raw_paths=[str(p) for p in path_cands])
                datasets.append(ds)

            splits = split_datasets(datasets=datasets, 
                                    split=ibis_split)
            cfg = HTSRawConfig(tf_name=tf,
                               tf_id=tf2id[tf],
                               stage=stage, 
                               stage_id=stage2id[stage],
                               splits=splits,
                               flanks="", # unfinished, will be asssigned only to test datasets
                               assign_path="") # unfinished, will be asssigned only to test datasets
            configs.append(cfg)
        all_configs[stage] = configs
    
    return all_configs

def get_zero_selectors(stage_id_sizes: dict[int, int], 
                 zero_size: int,
                 seed: int = 777):
    total_size = sum(stage_id_sizes.values())
    stage_zero_sizes = {stage: int((size / total_size) * zero_size) for stage, 
                         size in stage_id_sizes.items()}
    rest = total_size - sum(stage_zero_sizes)
    first_stage = next(iter(stage_zero_sizes.keys()))
    stage_zero_sizes[first_stage] += rest
    stage_zero_assigns = []
    for stage, size in stage_zero_sizes.items():
        for _ in range(size):
            stage_zero_assigns.append(stage)
    rng = random.Random(seed)
    rng.shuffle(stage_zero_assigns)

    zero_selectors = {}
    for stage in stage_id_sizes.keys():
        zero_stage_inds = [ind for ind, sid in enumerate(stage_zero_assigns) if sid == stage]
        zero_selectors[stage] = PredefinedIndicesSelector(zero_stage_inds)

    return zero_selectors

def write_filtered_assign(in_path: str, 
                          stage2assign: dict[int, Path],
                          stage2datasets: dict[int, list[HTSRawDataset]],
                          stage_id_sizes: dict[int,int],
                          zero_size: int,
                          seed: int,
                          recalc: bool):
    if not recalc and all(path.exists() for path in stage2assign.values()):
        logger.info("Skipping stage-specific assign files writing as they are already exist and no recalc option specified")
        return
    
    logger.info("Creating zero seqs selectors for stage-specific assign files")
    stage2zero_selector = get_zero_selectors(stage_id_sizes=stage_id_sizes,
                                zero_size=zero_size,
                                seed=seed)
    
    logger.info("Writing stage-specific assign files with train datasets omitted")
    stage2repids = {}
    for stage, datasets in stage2datasets.items():
        stage2repids[stage] = set(ds.rep_id for ds in datasets)

    try:
        stage_fds = {stage: open(path, "w") for stage, path in stage2assign.items()}
        with open(in_path, 'r') as inp:
            for line in tqdm.tqdm(inp):
                entry = SeqAssignEntry.from_line(line)
                if entry.cycle == 0: # zero seqs
                    for stage_id, selector in stage2zero_selector.items():
                        to_take = selector.add(entry)
                        if to_take:
                            entry.stage_ind = stage_id
                            print(entry.to_line(), file=stage_fds[stage_id])
                else:
                    stage_id = entry.stage_ind
                    repids = stage2repids[stage_id]
                    if entry.rep_ind in repids:
                        print(line, file=stage_fds[stage_id], end="")
    finally:           
        for fd in stage_fds.values():
            fd.close()

def write_data(stage_configs: dict[str, list[HTSRawConfig]],
                     stage2id: dict[str, int],
                     zero_size: int, 
                     assign_path: str | Path, 
                     out_dir: str, 
                     recalc: bool,
                     seed=777):
    configs_main_dir = out_dir / 'configs'
    assign_main_dir = out_dir / 'assign'

    stage_sizes = {st: 0 for st in stage_configs.keys()}
    
    for stage, configs in stage_configs.items():
        for cfg in configs:
            test_datasets = cfg.splits.get('test')
            if test_datasets is None:
                continue
            else:
                for _, rep_info in test_datasets.items():
                    for _, ds in rep_info.items():
                        stage_sizes[stage] += ds.size
                      
    stage_id_sizes = {stage2id[stage]: size for stage, size in stage_sizes.items()}
    
    
    stage2test_datasets = {stage2id[st]: [] for st in stage_configs.keys()}
    stage2assign = {}

    for stage, configs in stage_configs.items():
        stage_id = stage2id[stage]
       
        assign_stage_dir = assign_main_dir / stage
        assign_stage_dir.mkdir(exist_ok=True, parents=True) 
        assign_stage_seqs_path = assign_stage_dir / 'assign.seqs2'        
        
        configs_stage_dir = configs_main_dir / stage
        configs_stage_dir.mkdir(exist_ok=True, parents=True)
        
        assign_flanks_path = assign_stage_dir / 'flanks.json'
        for cfg in configs:
            cfg.assign_path = str(assign_stage_seqs_path)
            cfg.flanks = str(assign_flanks_path)
            test_datasets = cfg.splits.get('test')
            if test_datasets is not None:
                for _, rep_info in test_datasets.items():
                    for _, ds in rep_info.items():
                        stage2test_datasets[stage_id].append(ds)

            cfg_path = configs_stage_dir /  f"{cfg.tf_name}.json"
            cfg.save(cfg_path)

        write_flanks(datasets=stage2test_datasets[stage_id],
                     flanks_path=assign_flanks_path)
        
        stage2assign[stage_id] = assign_stage_seqs_path
        
    write_filtered_assign(in_path=assign_path,
                          stage2assign=stage2assign,
                          stage2datasets=stage2test_datasets,
                          stage_id_sizes=stage_id_sizes,
                          zero_size=zero_size,
                          seed=seed,
                          recalc=recalc)


LEADERBOARD_EXCEL = "/home_local/dpenzar/IBIS TF Selection - Nov 2022 _ Feb 2023.xlsx"
SPLIT_SHEET_NAME = "v3 TrainTest marked (2023)"
HTS_DIR = Path("/home_local/vorontsovie/greco-data/release_8d.2022-07-31/full/HTS")
STAGES = ('Final', 'Leaderboard')
ZEROS_CYCLE_DIR = Path("/mnt/space/hughes/GHT-SELEXFeb2023/")
OUT_DIR = Path("/home_local/dpenzar/BENCH_FULL_DATA/HTS/")
OUT_RAW_DIR = OUT_DIR / "RAW"
OUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
INDICES_SEP = " "

logger.info("Reading ibis meta information for htselex")
ibis_table, rep2typemap = parse_ibis_table(path=LEADERBOARD_EXCEL, 
                              sheet_name=SPLIT_SHEET_NAME)

zeros_paths = list(ZEROS_CYCLE_DIR.glob("*Cycle0_R1.fastq.gz"))
zeros_paths = sorted([str(x) for x in zeros_paths])

if not ibis_table['stage'].isin(STAGES).all():
    raise Exception(f"Some tfs has invalid stage: {ibis_table[~(ibis_table['stage'].isin(STAGES))]}")

stage_wise_copy_paths, stage_wise_tf2split, all_nonzero_paths = extract_info(ibis_table=ibis_table,
                                                                             stages=STAGES)

logger.info("Calculating file with sequence counts")
counter_dir = OUT_RAW_DIR / "counter"
counter_config = OUT_RAW_DIR / 'counter.json'
htselex_counts_path = OUT_RAW_DIR / "htselex_occs.txt"
seq_assign_path = OUT_RAW_DIR / "seq_assign.txt"
meta_info_path = OUT_RAW_DIR / "meta.json"

counter = count_seqs(counter_dir=counter_dir,
           counter_config=counter_config,
           zeros_paths=zeros_paths,
           nonzero_paths=all_nonzero_paths,
           counts_path=htselex_counts_path,
           indices_sep=INDICES_SEP,
           recalc=args.recalc)

logger.info("Filtering non-unique seqs and calculating info for next steps")
tf2id, rep2id, stage2id, ds_sizes, zero_size = assign_seqs(counter=counter,
            counts_path=htselex_counts_path,
            stage_wise_copy_paths=stage_wise_copy_paths,
            assign_path=seq_assign_path,
            meta_info_path=meta_info_path,
            recalc=args.recalc,
            indices_sep=INDICES_SEP)

logger.info("Splitting data into train-test and writing configs")
stage_configs = split_data(stage_wise_copy_paths=stage_wise_copy_paths,
                               stage_wise_tf2split=stage_wise_tf2split,
                               tf2id=tf2id,
                               rep2id=rep2id,
                               stage2id=stage2id,
                               ds_sizes=ds_sizes,
                               rep2typemap=rep2typemap)

logger.info("Writing raw datasets")
write_data(stage_configs=stage_configs,
                 stage2id=stage2id,
                 zero_size=zero_size,
                 assign_path=seq_assign_path,
                 out_dir=OUT_DIR,
                 recalc=args.recalc,
                 seed=args.seed)