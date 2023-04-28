
type="Leaderboard"
BECNHMARK_PROCESSED="/home_local/dpenzar/BENCHMARK_PROCESSED"
TAG_DB=${BECNHMARK_PROCESSED}/tag.db
TAG_DB_CFG=${BECNHMARK_PROCESSED}/tag.json
DATA_TYPE="AFS"

python make_db.py --tagdb ${TAG_DB} --tagdb_cfg ${TAG_DB_CFG}
for config_path in /home_local/dpenzar/BENCH_FULL_DATA/${DATA_TYPE}/configs/${type}/*.json; do
    echo ${config_path}
    out_dir="${BECNHMARK_PROCESSED}/${DATA_TYPE}/${type}"
    python peaks_split.py --benchmark_out_dir ${out_dir}\
                            --config_file ${config_path}\
                            --tagdb_cfg ${TAG_DB_CFG}\
                            --type $type
done
