

BECNHMARK_PROCESSED="/home_local/dpenzar/BENCHMARK_PROCESSED"
DATA_TYPE="PBM"

for type in "Leaderboard" "Final"; do
    echo $type
    for config_path in /home_local/dpenzar/BENCH_FULL_DATA/${DATA_TYPE}/configs/${type}/*.json; do
        echo ${config_path}
        out_dir="${BECNHMARK_PROCESSED}/${DATA_TYPE}/${type}"
        python pbm_split.py --benchmark_out_dir ${out_dir}\
                              --config_file ${config_path}\
                              --type $type\
                              --remove_grey_zone
    done
done
