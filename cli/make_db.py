import sys 
import argparse


from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--tagdb",
                    required=True,
                    type=str)
parser.add_argument("--tagdb_cfg",
                    required=True,
                    type=str)
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)

args = parser.parse_args()

sys.path.append(args.bibis_root)

from bibis.seqdb.config import DBConfig

DB_PATH = Path(args.tagdb)
DB_CONFIG_PATH =  Path(args.tagdb_cfg)

DB_PATH.parent.mkdir(exist_ok=True, parents=True)
DB_CONFIG_PATH.parent.mkdir(exist_ok=True, parents=True)


NAME_PARTS = ["adj", "adj", "nat", "animal"]
PART_PATHS = {
    "adj": "/home_local/dpenzar/ibis-challenge/benchmark/data/adjectives.txt",
    "animal": "/home_local/dpenzar/ibis-challenge/benchmark/data/animals.txt",
    "nat": "/home_local/dpenzar/ibis-challenge/benchmark/data/nations.txt"
}

cfg = DBConfig.make(db_path=DB_PATH,
                    parts=NAME_PARTS, 
                    parts_path=PART_PATHS)
cfg.build()
cfg.save(DB_CONFIG_PATH)