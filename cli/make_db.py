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

args = parser.parse_args()


from bibis.seqdb.config import DBConfig

DB_PATH = Path(args.tagdb)
DB_CONFIG_PATH =  Path(args.tagdb_cfg)

DB_PATH.parent.mkdir(exist_ok=True, parents=True)
DB_CONFIG_PATH.parent.mkdir(exist_ok=True, parents=True)



cfg = DBConfig.make(db_path=DB_PATH,
                    tagger_type="alpha",
                    tag_length=7)
cfg.build()
cfg.save(DB_CONFIG_PATH)