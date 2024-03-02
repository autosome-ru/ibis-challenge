

BECNHMARK_PROCESSED="/home_local/dpenzar/BENCHMARK_PROCESSED"
TAG_DB=${BECNHMARK_PROCESSED}/tag.db
TAG_DB_CFG=${BECNHMARK_PROCESSED}/tag.json

DATA_TYPE="SMS"
LOGS_PATH="${DATA_TYPE}_logs.txt"
echo $DATA_TYPE
python make_db.py --tagdb ${TAG_DB} --tagdb_cfg ${TAG_DB_CFG}
for type in "Leaderboard" "Final"; do
    echo $type
    for config_path in /home_local/dpenzar/BENCH_FULL_DATA/${DATA_TYPE}/configs/${type}/*.json; do
        echo ${config_path}
        out_dir="${BECNHMARK_PROCESSED}/${DATA_TYPE}/${type}"
        python sms_split.py  --benchmark_out_dir ${out_dir}\
                             --config_file ${config_path}\
                             --tagdb_cfg ${TAG_DB_CFG}\
                             --type $type\
                             --zero_seqs_path ~/BENCH_FULL_DATA/SMS/data/${type}/zeros.json\
                             --unique_seqs_path ~/BENCH_FULL_DATA/SMS/data/${type}/unique_with_flanks.json >> $LOGS_PATH  
    done
done
