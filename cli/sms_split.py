import sys 
import json
from pathlib import Path
import shutil
import argparse
import sys
import gzip 
from Bio import SeqIO
from Bio.Seq import Seq
import random

parser = argparse.ArgumentParser()

def sample_count_conv(arg):
    if arg == "all":
        return arg
    else:
        return int(arg)
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
parser.add_argument("--sample_count",
                    default="all",
                    type=sample_count_conv)
parser.add_argument("--zero_seqs_path", 
                    required=True, 
                    type=str)
parser.add_argument("--unique_seqs_path", 
                    required=True, 
                    type=str)
parser.add_argument("--foreign_neg2pos_ratio",
                    default=5,
                    type=int)
parser.add_argument("--zero_neg2pos_ratio",
                    default=5,
                    type=int)
parser.add_argument("--seed",
                    action='store_true',
                    default=777)
parser.add_argument("--log_path",
                    default='log.txt')
parser.add_argument("--logger_name",
                    default="sms_split")
parser.add_argument("--recalc",
                    action="store_true")

args = parser.parse_args()

from bibis.sms.config import SMSRawConfig
from bibis.sampling.gc import SetGCSampler
from bibis.benchmark.dataset import DatasetInfo
from bibis.seq.seqentry import SeqEntry, write as seq_write
from bibis.scoring.label import NO_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL
from bibis.seqdb.config import DBConfig 
from bibis.logging import get_logger, BIBIS_LOGGER_CFG

BIBIS_LOGGER_CFG.set_path(path=args.log_path)
logger = get_logger(name=args.logger_name, path=args.log_path)

def log_splits(cfg: SMSRawConfig, splits: list[str]=None):
    if splits is None:
        splits = ['train', 'test']

    for split in splits:
        datasets = cfg.splits.get(split)
        if datasets is None:
            logger.info(f"For factor {cfg.tf_name} no replics are going to {split}")
        else:
            reps = ", ".join([ds.rep for ds in datasets])
            logger.info(f"For factor {cfg.tf_name} the following replics are going to {split}: {reps}")    

BENCH_SEQDB_CFG = Path(args.tagdb_cfg)

SMS_BENCH_DIR = Path(args.benchmark_out_dir)
SMS_BENCH_DIR.mkdir(parents=True, exist_ok=True)

cfg = SMSRawConfig.load(args.config_file)
log_splits(cfg)

with open(args.unique_seqs_path, "r") as inp:
    unique_seqs2flanks: dict[str, tuple[str, str]] = json.load(inp)

friend_seqs = set()

train_datasets = cfg.splits.get('train')
test_datasets = cfg.splits.get('test')

if train_datasets is not None:
    train_dir = SMS_BENCH_DIR / "train" / cfg.tf_name  
    train_dir.mkdir(parents=True, exist_ok=True)
    for ds in train_datasets:
        path = Path(ds.path)
        shutil.copy(path, train_dir / path.name)

if test_datasets is None:
    exit(0)

left_flank=test_datasets[0].left_flank
right_flank=test_datasets[0].right_flank
ZERO_FLANKS = (left_flank, right_flank)

valid_dir = SMS_BENCH_DIR / "valid" / cfg.tf_name  
if not args.recalc and valid_dir.exists():
    logger.info("Skipping dataset writing as datasets dir already exist and recalc flag is not specified")
    exit(0)
valid_dir.mkdir(exist_ok=True, parents=True)

answer_valid_dir = valid_dir / "answer"
answer_valid_dir.mkdir(exist_ok=True)
participants_valid_dir = valid_dir / "participants"
participants_valid_dir.mkdir(exist_ok=True)

logger.info('Deducing aliens pool')
test_sequences = []
for ds in test_datasets:
    with gzip.open(ds.path, "rt") as inp:
        for rec in SeqIO.parse(inp, format="fastq"):
            test_sequences.append(str(rec.seq).upper())

unique_seqs = set(unique_seqs2flanks.keys())
alien_seqs = list(unique_seqs - friend_seqs - set(test_sequences))
alien_seqs = [SeqEntry(sequence=Seq(s),
                         label=NEGATIVE_LABEL) for s in alien_seqs]
# foreign seqs now contain only sequences for other tfs

db = DBConfig.load(BENCH_SEQDB_CFG).build()
    
# benchmark part files

if args.sample_count != "all":
    logger.info("Sampling positives")
    num_samples = args.sample_count
    if num_samples > len(test_sequences):
        num_samples = len(test_sequences)
        print(f"Cant sample more than {num_samples} for {cfg.tf_name}")
        pos_samples = test_sequences
    else:
        pos_samples = random.sample(test_sequences, k=num_samples)
else:
    logger.info("Using all positives seqs")
    pos_samples = test_sequences

pos_samples = [SeqEntry(sequence=Seq(s),
                        label=POSITIVE_LABEL) for s in pos_samples]
pos_samples = db.taggify_entries(pos_samples)

user_known_samples: list[SeqEntry] = []
user_known_samples.extend(pos_samples)

seq_datasets: dict[str, list[SeqEntry]] = {}

logger.info("Generating aliens dataset")
alien_sampler = SetGCSampler.make(negatives=alien_seqs,
                                  sample_per_object=args.foreign_neg2pos_ratio,
                                  seed=args.seed)

alien_samples = alien_sampler.sample(positive=pos_samples, return_loss=False)
alien_samples = db.taggify_entries(alien_samples)
user_known_samples.extend(alien_samples)
seq_datasets['aliens'] =  pos_samples + alien_samples    

# zero seqs 
logger.info("Generating input dataset")
with open(args.zero_seqs_path) as inp:
    zero_seqs = json.load(inp)
# zero seqs already contain no sequences from this tf

zero_seqs = [SeqEntry(sequence=Seq(s),
                     label=NEGATIVE_LABEL) for s in zero_seqs]
zero_sampler = SetGCSampler.make(negatives=zero_seqs,
                                     sample_per_object=args.zero_neg2pos_ratio,
                                     seed=args.seed)
zero_samples = zero_sampler.sample(positive=pos_samples, return_loss=False)
zero_samples = db.taggify_entries(zero_samples)
user_known_samples.extend(zero_samples)

seq_datasets['input'] = pos_samples + zero_samples    

for dataset_name, samples in seq_datasets.items():
    logger.info(f"Writing {dataset_name} dataset")
    ds_dir = answer_valid_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = ds_dir / 'data.fasta'

    flanked_samples = []
    for entry in samples:
        seq = str(entry.sequence)
        if dataset_name == "input":
            lf, rf = unique_seqs2flanks.get(seq, ZERO_FLANKS) 
        else:
            lf, rf = unique_seqs2flanks[seq]

        flanked_seq = lf[1:] + seq + rf[1:] 
        flanked_entry = SeqEntry(sequence=Seq(flanked_seq),
                                 tag=entry.tag,
                                 label=entry.label)
        flanked_samples.append(flanked_entry)
    seq_write(flanked_samples, fasta_path)

    answer = {'labels': {pe.tag: pe.label for pe in samples}}

    answer_path = ds_dir   / "data_answer.json"
    with open(answer_path, "w") as out:
        json.dump(answer, fp=out, indent=4)

    config_path = ds_dir / "config.json"
    ds_info = DatasetInfo(name = f"{cfg.tf_name}_{dataset_name}", 
                          tf = cfg.tf_name,
                          background=dataset_name,
                          fasta_path=str(fasta_path),
                          answer_path=str(answer_path))
    ds_info.save(config_path)

# write sequences for user
logger.info(f"Writing participants sequence file")
participants_fasta_path = participants_valid_dir / "submission.fasta"
random.shuffle(user_known_samples)
for entry in user_known_samples:
    entry.label = NO_LABEL
    entry.metainfo = {}
seq_write(user_known_samples, participants_fasta_path)
