import sys 

sys.path.append("/home_local/dpenzar/bibis_git/ibis-challenge")

from pathlib import Path

from bibis.seqdb.config import DBConfig

BENCH_PROCESSED_DIR = Path("/home_local/dpenzar/BENCHMARK_PROCESSED")
BENCH_PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
BENCH_SEQDB_CFG = BENCH_PROCESSED_DIR / "tag.json"

DB_PATH = BENCH_PROCESSED_DIR/"tag.db"
DB_CONFIG_PATH =  BENCH_PROCESSED_DIR / "tag.json"
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