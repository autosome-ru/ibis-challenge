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
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
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

args = parser.parse_args()


sys.path.append(args.bibis_root) # temporary solution while package is in development

from bibis.sms.config import RAW_SMSConfig

from bibis.sampling.gc import SetGCSampler
from bibis.benchmark.dataset import DatasetInfo
from bibis.seq.seqentry import SeqEntry, write as seq_write
from bibis.scoring.label import NO_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL
from bibis.seqdb.config import DBConfig 

BENCH_SEQDB_CFG = Path(args.tagdb_cfg)

SMS_BENCH_DIR = Path(args.benchmark_out_dir)
SMS_BENCH_DIR.mkdir(parents=True, exist_ok=True)

cfg = RAW_SMSConfig.load(args.config_file)

with open(args.unique_seqs_path, "r") as inp:
    unique_seqs2flanks = json.load(inp)

friend_seqs = set()

train_datasets = cfg.splits.get('train')
test_datasets = cfg.splits.get('test')

if train_datasets is not None:
    print(f"For factor {cfg.tf_name} the following replics are going to train:")
    train_dir = SMS_BENCH_DIR / "train" / cfg.tf_name  
    train_dir.mkdir(parents=True, exist_ok=True)
    for ds in train_datasets:
        print(f"\t{ds.rep}")
        path = Path(ds.path)
        shutil.copy(path, train_dir / path.name)
else:
    print(f"For factor {cfg.tf_name} no replics are going to train")

if test_datasets is None:
    print(f"For factor {cfg.tf_name} no replics are going to test")
    exit(0)
else:
    print(f"For factor {cfg.tf_name} the following replics are going to test:")
    for ds in test_datasets:
        print(f"\t{ds.rep}")

left_flank=test_datasets[0].left_flank
#left_flank = left_flank[:args.keep_left_cnt] + "N" * (len(left_flank) - args.keep_left_cnt)
right_flank=test_datasets[0].right_flank
ZERO_FLANKS = (left_flank, right_flank)

valid_dir = SMS_BENCH_DIR / "valid" / cfg.tf_name  
valid_dir.mkdir(exist_ok=True, parents=True)
answer_valid_dir = valid_dir / "answer"
answer_valid_dir.mkdir(exist_ok=True)
participants_valid_dir = valid_dir / "participants"
participants_valid_dir.mkdir(exist_ok=True)

test_sequences = []
for ds in test_datasets:
    with gzip.open(ds.path, "rt") as inp:
        for rec in SeqIO.parse(inp, format="fastq"):
            test_sequences.append(str(rec.seq).upper())

unique_seqs = set(unique_seqs2flanks.keys())
foreign_seqs = list(unique_seqs - friend_seqs - set(test_sequences))
foreign_seqs = [SeqEntry(sequence=Seq(s),
                         label=NEGATIVE_LABEL) for s in foreign_seqs]
# foreign seqs now contain only sequences for other tfs

db = DBConfig.load(BENCH_SEQDB_CFG).build()
    
# benchmark part files

if args.sample_count != "all":
    num_samples = args.sample_count
    if num_samples > len(test_sequences):
        num_samples = len(test_sequences)
        print(f"Cant sample more than {num_samples} for {cfg.tf_name}")
        pos_samples = test_sequences
    else:
        pos_samples = random.sample(test_sequences, k=num_samples)
else:
    pos_samples = test_sequences

pos_samples = [SeqEntry(sequence=Seq(s),
                        label=POSITIVE_LABEL) for s in pos_samples]
pos_samples = db.taggify_entries(pos_samples)

user_known_samples: list[SeqEntry] = []
user_known_samples.extend(pos_samples)


# foreign 
print("Generating foreign dataset")
negative_sampler = SetGCSampler.make(negatives=foreign_seqs,
                                     sample_per_object=args.foreign_neg2pos_ratio,
                                     seed=args.seed)
neg_samples = negative_sampler.sample(positive=pos_samples, return_loss=False)
neg_samples = db.taggify_entries(neg_samples)
user_known_samples.extend(neg_samples)

foreign_ds_dir = answer_valid_dir /  "foreign"
foreign_ds_dir.mkdir(parents=True, exist_ok=True)
samples: list[SeqEntry] = pos_samples + neg_samples    

fasta_path = foreign_ds_dir  / "data.fasta"
flanked_samples = []
for entry in samples:
    seq = str(entry.sequence)
    left_flank, right_flank = unique_seqs2flanks[seq]
    flanked_seq = left_flank[1:] + seq + right_flank[1:]
    flanked_entry = SeqEntry(sequence=Seq(flanked_seq),
                             tag=entry.tag,
                             label=entry.label)
    flanked_samples.append(flanked_entry)

# for answer, we write flanked sequences
seq_write(flanked_samples, fasta_path)

answer = {pe.tag: pe.label for pe in samples}
answer_path = foreign_ds_dir   / "data_answer.json"
with open(answer_path, "w") as out:
    json.dump(answer, fp=out, indent=4)

config_path = foreign_ds_dir / "config.json"
ds_info = DatasetInfo(name = f"{cfg.tf_name}_foreign", 
                      tf = cfg.tf_name,
                      background="foreign",
                      fasta_path=str(fasta_path),
                      answer_path=str(answer_path),
                      left_flank=left_flank,
                      right_flank=right_flank)
ds_info.save(config_path)

# zero seqs 
print("Generating input dataset")
with open(args.zero_seqs_path) as inp:
    zero_seqs = json.load(inp)
# zero seqs already contains no sequences from this tf

zero_seqs = [SeqEntry(sequence=Seq(s),
                     label=NEGATIVE_LABEL) for s in zero_seqs]
negative_sampler = SetGCSampler.make(negatives=zero_seqs,
                                     sample_per_object=args.zero_neg2pos_ratio,
                                     seed=args.seed)
neg_samples = negative_sampler.sample(positive=pos_samples, return_loss=False)
neg_samples = db.taggify_entries(neg_samples)
user_known_samples.extend(neg_samples)

zeros_ds_dir =  answer_valid_dir / "input"
zeros_ds_dir.mkdir(parents=True, exist_ok=True)
samples: list[SeqEntry] = pos_samples + neg_samples    

fasta_path = zeros_ds_dir  / "data.fasta"
flanked_samples = []

for entry in samples:
    seq = str(entry.sequence)
    left_flank, right_flank = unique_seqs2flanks.get(seq, ZERO_FLANKS)
    flanked_seq = left_flank[1:] + seq + right_flank[1:]
    flanked_entry = SeqEntry(sequence=Seq(flanked_seq),
                             tag=entry.tag,
                             label=entry.label)
    flanked_samples.append(flanked_entry)
seq_write(flanked_samples, fasta_path)
        
answer = {pe.tag: pe.label for pe in samples}
answer_path = zeros_ds_dir   / "data_answer.json"
with open(answer_path, "w") as out:
    json.dump(answer, fp=out, indent=4)

config_path = zeros_ds_dir/ "config.json"
ds_info = DatasetInfo(name = f"{cfg.tf_name}_input", 
                      tf = cfg.tf_name,
                      background="input",
                      fasta_path=str(fasta_path),
                      answer_path=str(answer_path),
                      left_flank=left_flank,
                      right_flank=right_flank)
ds_info.save(config_path)

# write sequences for user
# we write them without flanks as they are masked to be identical for each datasets
# and will be provided separately 
participants_fasta_path = participants_valid_dir / "submission.fasta"
random.shuffle(user_known_samples)
for entry in user_known_samples:
    entry.label = NO_LABEL
    entry.metainfo = {}
seq_write(user_known_samples, participants_fasta_path)
