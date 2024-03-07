import sys 
from pathlib import Path
import random 
import argparse
import sys

def filter_chrom(peak_file: str | Path,
                 out_file: str | Path,
                 chroms: set[str] | list[str]):
    with open(peak_file, 'r') as inp, open(out_file, "w") as out:
        for line in inp:
            if line.startswith("#"):
                print(line, end="", file=out)
            else:
                ch, _ = line.split("\t", maxsplit=1)
                if ch in chroms:
                    print(line, end="", file=out)  

parser = argparse.ArgumentParser()

parser.add_argument("--benchmark_out_dir", 
                    required=True, 
                    type=str)
parser.add_argument("--tagdb_cfg",
                    required=True,
                    type=str)
parser.add_argument("--config_file", 
                    required=True, 
                    type=str)
parser.add_argument("--type", 
                    choices=['Leaderboard', 'Final'], 
                    required=True, type=str)
parser.add_argument("--n_procs", 
                    default=1,
                    type=int)
parser.add_argument("--bedtools", 
                    default="/home_local/dpenzar/bedtools2/bin",
                    type=str)
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
parser.add_argument("--log_path",
                    default='log.txt')
parser.add_argument("--logger_name",
                    default="peaks_split")
parser.add_argument("--recalc",
                    action="store_true")

args = parser.parse_args()

sys.path.append(args.bibis_root) # temporary solution while package is in development

from bibis.seqdb.config import DBConfig 
from bibis.peaks.config import PeakSeqConfig, PeakSeqDatasetConfig
from bibis.peaks.peakfile import PeakList
from bibis.bedtools.beddata import BedData, join_bed
from bibis.peaks.samplers import (PeakForeignSampler, 
                                  PeakShadesSampler,
                                  PeakGenomeSampler,
                                  cut_to_window)
from bibis.seq.genome import Genome
from bibis.seq.seqentry import SeqEntry, write as seq_write
from bibis.bedtools.bedtoolsexecutor import BedtoolsExecutor
from bibis.logging import get_logger, BIBIS_LOGGER_CFG
from bibis.scoring.label import NO_LABEL

BIBIS_LOGGER_CFG.set_path(path=args.log_path)
logger = get_logger(name=args.logger_name, path=args.log_path)

BedtoolsExecutor.set_defaull_executor(args.bedtools)

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

BENCH_SEQDB_CFG = Path(args.tagdb_cfg)

PEAK_BENCH_DIR = Path(args.benchmark_out_dir)
PEAK_BENCH_DIR.mkdir(parents=True, exist_ok=True)

cfg = PeakSeqConfig.load(args.config_file)

log_splits(cfg)


if "train" in cfg.splits:
    logger.info("Writing train datasets")
    split = cfg.splits["train"]
    train_dir = PEAK_BENCH_DIR / "train" / cfg.tf_name
    train_dir.mkdir(exist_ok=True, parents=True)
    
    for ind, peak_path in enumerate(split.reps.values(), 1):
        fl_name = Path(peak_path).name
        replic_path = train_dir / fl_name
        filter_chrom(peak_file=peak_path, 
                     out_file=replic_path,
                     chroms=split.chroms)

if "test" not in cfg.splits:
    exit(0)

logger.info("Creating test datasets")
valid_dir = PEAK_BENCH_DIR / "valid" / cfg.tf_name
if not args.recalc and valid_dir.exists():
    logger.info("Skipping dataset writing as datasets dir already exist and recalc flag is not specified")
    exit(0)
valid_dir.mkdir(exist_ok=True, parents=True)

split = cfg.splits["test"]

if len(split.reps) != 1:
    raise Exception("Only one replic can be used as validation")

validation_replic_file = next(iter(split.reps.values()))

foreign_peaks = [PeakList.read(t) for t in cfg.foreign_cfg.foreigns_path]
foreign_beds = [f.to_beddata() for f in foreign_peaks]

valid_bed = PeakList.read(validation_replic_file).to_beddata()
valid_bed = valid_bed.filter(lambda e: e.chr in split.chroms) # type: ignore

if "train" in cfg.splits:
    train_peaks = [PeakList.read(t) for t in cfg.splits["train"].reps.values()]
    train_beds = [f.to_beddata() for f in train_peaks]
else:
    train_beds = []

train_beds = [bed.filter(lambda e: e.chr in split.chroms) for bed in train_beds]
foreign_beds = [bed.filter(lambda e: e.chr in split.chroms) for bed in foreign_beds]

if cfg.black_list_path is not None:
    black_list = BedData.from_file(cfg.black_list_path)
    assert split.hide_regions is not None
    logger.info(f"Hiding regions {split.hide_regions} from test")
    hide_regions = BedData.from_file(split.hide_regions)
    black_list = join_bed([black_list, hide_regions]).merge()
else:
    black_list = None

if black_list is not None:
    valid_black_list = black_list.filter(lambda e: e.chr in split.chroms)
else:
    valid_black_list = None

friends_peaks = list(train_beds)
logger.info("Downloading genome")
gpath = Path(cfg.genome_path)
if gpath.is_dir():
    genome = Genome.from_dir(gpath)
else:
    genome = Genome.from_fasta(gpath)

valid_bed = valid_bed.to_min_width(width=cfg.window_size, 
                                   genome=genome)

positives_bed = cut_to_window(bed=valid_bed, 
                              window_size=cfg.window_size,
                              genome=genome)
before_filter_pos_cnt = len(positives_bed)
positives_bed = positives_bed.subtract(valid_black_list, 
                       full=True)
removed_cnt = before_filter_pos_cnt - len(positives_bed)
if removed_cnt != 0:
    logger.warning(f"Removed {removed_cnt} positive seqs as containing N or sequences from prohibited regions")

samples: dict[str, BedData] = {"positives": positives_bed}

logger.info("Creating shades")    
shades_sampler = PeakShadesSampler.make(window_size=cfg.window_size,
                                        genome=genome,
                                        tf_peaks=[valid_bed], # type: ignore
                                        friend_peaks=friends_peaks,
                                        black_list_regions=valid_black_list,
                                        min_dist=cfg.shades_cfg.min_dist, 
                                        max_dist=cfg.shades_cfg.max_dist,
                                        sample_per_object=cfg.shades_cfg.balance,
                                        seed=cfg.seed)
samples['shades'] = shades_sampler.sample_bed()
friends_peaks.append(samples['shades'])

logger.info("Creating aliens")    
foreign_sampler = PeakForeignSampler.make(window_size=cfg.window_size,
                                        genome=genome,
                                        tf_peaks=[valid_bed], # type: ignore
                                        real_peaks=foreign_beds,
                                        friend_peaks=friends_peaks,
                                        black_list_regions=valid_black_list,
                                        min_dist=cfg.foreign_cfg.min_dist,
                                        sample_per_object=cfg.foreign_cfg.balance,
                                        seed=cfg.seed)

samples['aliens'] = foreign_sampler.sample_bed()
friends_peaks.append(samples['aliens'])

logger.info("Creating random genome samples")
genome_sampler = PeakGenomeSampler.make(window_size=cfg.window_size,
                                genome=genome,
                                tf_peaks=[valid_bed], # type: ignore
                                friend_peaks=friends_peaks,
                                black_list_regions=valid_black_list,
                                min_dist=cfg.foreign_cfg.min_dist,
                                sample_per_object=cfg.genome_sample_cfg.balance,
                                exact=cfg.genome_sample_cfg.exact,
                                max_overlap=cfg.genome_sample_cfg.max_overlap,
                                n_procs=args.n_procs,
                                seed=cfg.seed)

samples['random'] = genome_sampler.sample_bed()
friends_peaks.append(samples['random'])

db = DBConfig.load(BENCH_SEQDB_CFG).build()

# benchmark part files
logger.info("Writing datasets files")
parts_dir = valid_dir / 'parts'
parts_dir.mkdir(exist_ok=True)

user_known_samples: list[SeqEntry] = []
for name, bed in samples.items():
    fas = genome.cut(bed)
    fas = db.taggify_entries(fas)
    user_known_samples.extend(fas)
    table_path = parts_dir / f"{name}.tsv"
    with open(table_path, "w") as out:
        for entry, seq in zip(bed, fas):
            print(entry.chr, 
                    entry.start,
                    entry.end, 
                    seq.tag, 
                    sep="\t",
                    file=out)

    seq_path = parts_dir / f"{name}.fasta"        
    seq_write(fas, seq_path)
    
ds_config_file = valid_dir / f"config.json"
ds = PeakSeqDatasetConfig(tf_name=cfg.tf_name, 
                            tf_path=str(valid_dir))
ds.save(ds_config_file)

logger.info("Writing answer file")
# answer files for benchmark
answer_dir = valid_dir / 'answer'
answer_dir.mkdir(exist_ok=True)
for background in ('shades', 'aliens', 'random'):
    path_pref = str(answer_dir / f"{cfg.tf_name}_{background}")
    ds_dir = answer_dir / background
    ds_dir.mkdir(exist_ok=True)
    ds_info = ds.make_ds(background=background,
                         path_pref=str(ds_dir / "data"),
                         hide_labels=False)
    
    filepath = answer_dir / background / f"config.json"
    ds_info.save(filepath)

logger.info(f"Writing participants sequence file")
participants_valid_dir = valid_dir / "participants"
participants_valid_dir.mkdir(exist_ok=True)
participants_fasta_path = participants_valid_dir / "submission.fasta"
logger.info(f"Total user sequences: {len(user_known_samples)}")
random.shuffle(user_known_samples)
for entry in user_known_samples:
    entry.label = NO_LABEL

seq_write(user_known_samples, participants_fasta_path)
