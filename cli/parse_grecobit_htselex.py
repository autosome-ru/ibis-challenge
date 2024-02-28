
import argparse
import pandas as pd
from pathlib import Path
import sys 
import tqdm 
import random

import argparse


sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

from collections import defaultdict

from bibis.utils import END_LINE_CHARS
from bibis.counting.fastqcounter import FastqGzReadsCounter
from bibis.hts.config import HTSRawConfig
from bibis.hts.dataset import HTSRawDataset
from bibis.hts.seqentry import SeqAssignEntry
from bibis.seq.utils import gc
    
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
                #print(f"Skipping factor tf: {tf}. No split provided")
                continue
            if isinstance(reps, float): # skip tf without pbms
                #print(f"Skipping factor tf: {tf}. Its split ({ibis_split}) is not none but there is no experiments for that factor", file=sys.stderr)
                continue
            tf2split[tf] = ibis_split
            tf_dir = OUT_DIR / tf 
            tf_dir.mkdir(parents=True, exist_ok=True)
            for rep in reps:
                for cycle in ('C1', 'C2', 'C3', 'C4'):
                    path_cands = list(HTS_DIR.glob(f"*/*@{rep}.{cycle}.*"))
                    if len(path_cands) == 0:
                        print(f'No {cycle} found for replic {rep} for tf {tf}')
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
    return counter 



def assign_seqs(counter,
                counts_path,
                stage_wise_copy_paths, 
                assign_path, 
                indices_sep=" "):
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

    stage_ids_opts = list(stage2id.values())

    ds_sizes = defaultdict(lambda: defaultdict(int))
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
                stage_ind = random.choice(stage_ids_opts) # randomly assign zero read to stage

            entry = SeqAssignEntry(seq=seq,
                                   cycle=cycle,
                                   rep_ind=rep_ind,
                                   tf_ind=tf_ind,
                                   stage_ind=stage_ind,
                                   gc_content=gc(seq))
            print(entry.to_line(),
                  file=out)

    return tf2id, rep2id, stage2id, ds_sizes


def write_configs(assign_path,
                  stage_wise_copy_paths, 
                  stage_wise_tf2split,
                  tf2id,
                  rep2id,
                  stage2id,
                  ds_sizes,
                  rep2typemap,
                  configs_main_dir):
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
                                path=str(assign_path), 
                                cycle=cycle_id,
                                rep=rep,
                                exp_tp=exp_tp,
                                left_flank=left_flank, 
                                right_flank=right_flank,
                                raw_paths=[str(p) for p in path_cands])
                datasets.append(ds)
            cfg = HTSRawConfig(tf_name=tf,
                               tf_id=tf2id[tf],
                               stage=stage, 
                               stage_id=stage2id[stage],
                               split=ibis_split,
                               datasets=datasets)
            configs.append(cfg)
        configs_stage_dir = configs_main_dir / stage
        configs_stage_dir.mkdir(exist_ok=True, parents=True)
        for cfg in configs:
            cfg_path = configs_stage_dir /  f"{cfg.tf_name}.json"
            cfg.save(cfg_path)

from bibis.utils import END_LINE_CHARS
LEADERBOARD_EXCEL = "/home_local/dpenzar/IBIS TF Selection - Nov 2022 _ Feb 2023.xlsx"
SPLIT_SHEET_NAME = "v3 TrainTest marked (2023)"
HTS_DIR = Path("/home_local/vorontsovie/greco-data/release_8d.2022-07-31/full/HTS")
STAGES = ('Final', 'Leaderboard')
STAGES_IDS = {'Final': 1, 'Leaderboard': 2}
ZEROS_CYCLE_DIR = Path("/mnt/space/hughes/GHT-SELEXFeb2023/")
OUT_DIR = Path("/home_local/dpenzar/BENCH_FULL_DATA/HTS/RAW3/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--recalc", action="store_true")
parser.add_argument("--calc_dir", help="Dir for calculation mapreduce operations", default="mapreduce_calc")
parser.add_argument("--n_procs", type=int, default=20)

args = parser.parse_args()

ibis_table, rep2typemap = parse_ibis_table(path=LEADERBOARD_EXCEL, 
                              sheet_name=SPLIT_SHEET_NAME)
print(ibis_table.head())
print(rep2typemap)

zeros_paths = list(ZEROS_CYCLE_DIR.glob("*Cycle0_R1.fastq.gz"))
zeros_paths = sorted([str(x) for x in zeros_paths])

if not ibis_table['stage'].isin(STAGES).all():
    raise Exception(f"Some tfs has invalid stage: {ibis_table[~(ibis_table['stage'].isin(STAGES))]}")

stage_wise_copy_paths, stage_wise_tf2split, all_nonzero_paths = extract_info(ibis_table=ibis_table,
                                                                             stages=STAGES)

counter_dir = OUT_DIR / "counter"
counter_config = OUT_DIR / 'counter.json'
htselex_counts_path = OUT_DIR / "htselex_occs.txt"
seq_assign_path = OUT_DIR / "seq_assign.txt"

INDICES_SEP = " "

counter = count_seqs(counter_dir=counter_dir,
           counter_config=counter_config,
           zeros_paths=zeros_paths,
           nonzero_paths=all_nonzero_paths,
           counts_path=htselex_counts_path,
           indices_sep=INDICES_SEP,
           recalc=args.recalc)

tf2id, rep2id, stage2id, ds_sizes = assign_seqs(counter=counter,
            counts_path=htselex_counts_path,
            stage_wise_copy_paths=stage_wise_copy_paths,
            assign_path=seq_assign_path)
print(ds_sizes)
configs_dir = OUT_DIR / 'configs'
write_configs(stage_wise_copy_paths=stage_wise_copy_paths,
              stage_wise_tf2split=stage_wise_tf2split,
              tf2id=tf2id,
              rep2id=rep2id,
              stage2id=stage2id,
              ds_sizes=ds_sizes,
              rep2typemap=rep2typemap,
              configs_main_dir=configs_dir,
              assign_path=seq_assign_path)