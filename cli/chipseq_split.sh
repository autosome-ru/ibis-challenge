
type="Leaderboard"
BECNHMARK_PROCESSED="/home_local/dpenzar/BENCHMARK_PROCESSED"
TAG_DB=${BECNHMARK_PROCESSED}/tag.db
TAG_DB_CFG=${BECNHMARK_PROCESSED}/tag.json

python make_db.py --tagdb ${TAG_DB} --tagdb_cfg ${TAG_DB_CFG}
for config_path in /home_local/dpenzar/BENCH_FULL_DATA/configs/${type}/*.json; do
    echo ${config_path}
    python chipseq_split.py --benchmark_out_dir ${BECNHMARK_PROCESSED}\
                            --config_file ${config_path}\
                            --tagdb_cfg ${TAG_DB_CFG}\
                            --type $type
done