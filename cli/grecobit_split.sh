BENCH_DATA=/home_local/dpenzar/BENCHMARK_APRIL_FINAL/BENCHMARK_DATA
BECNHMARK_PROCESSED="/home_local/dpenzar/BENCHMARK_APRIL_FINAL/BENCHMARK_PROCESSED"
TAG_DB=${BECNHMARK_PROCESSED}/tag.db
TAG_DB_CFG=${BECNHMARK_PROCESSED}/tag.json
python make_db.py --tagdb ${TAG_DB} --tagdb_cfg ${TAG_DB_CFG}
LOGFILE="split.log"
DATA_TYPE="PBM"
for type in "Final"; do
    for config_path in ${BENCH_DATA}/${DATA_TYPE}/configs/${type}/*.json; do
        echo ${config_path}
        out_dir="${BECNHMARK_PROCESSED}/${DATA_TYPE}/${type}"
        python pbm_split.py --benchmark_out_dir ${out_dir}\
                            --config_file ${config_path}\
                            --type $type\
                            --remove_grey_zone\
                            --log_path $LOGFILE
    done
done

for type in "Final"; do
    for DATA_TYPE in "CHS" "GHTS"; do 
        small_tp=`echo "$type" | awk '{print tolower($0)}'`
        small_data_tp=`echo "$DATA_TYPE" | awk '{print tolower($0)}'`
        for config_path in ${BENCH_DATA}/${DATA_TYPE}/configs/${type}/*.json; do
            echo ${config_path}
            out_dir="${BECNHMARK_PROCESSED}/${DATA_TYPE}/${type}"
            python peaks_split.py --benchmark_out_dir ${out_dir}\
                                    --config_file ${config_path}\
                                    --tagdb_cfg ${TAG_DB_CFG}\
                                    --type $type\
                                    --logger_name ${small_data_tp}_${small_tp}_split\
                                    --log_path $LOGFILE
        done
    done
done

DATA_TYPE="SMS"
for type in "Final"; do
    for config_path in ${BENCH_DATA}/${DATA_TYPE}/configs/${type}/*.json; do
        echo ${config_path}
        out_dir="${BECNHMARK_PROCESSED}/${DATA_TYPE}/${type}"
        python sms_split.py  --benchmark_out_dir ${out_dir}\
                             --config_file ${config_path}\
                             --tagdb_cfg ${TAG_DB_CFG}\
                             --type $type\
                             --zero_seqs_path  ${BENCH_DATA}/${DATA_TYPE}/data/${type}/zeros.json\
                             --unique_seqs_path  ${BENCH_DATA}/${DATA_TYPE}/data/${type}/unique_with_flanks.json\
                             --log_path $LOGFILE 
    done
done

DATA_TYPE="HTS"
python make_db.py --tagdb ${TAG_DB} --tagdb_cfg ${TAG_DB_CFG}
for type in "Final"; do
    for config_path in ${BENCH_DATA}/${DATA_TYPE}/configs/${type}/*.json; do
        echo ${config_path}
        out_dir="${BECNHMARK_PROCESSED}/${DATA_TYPE}/${type}"
        python hts_split.py  --benchmark_out_dir ${out_dir}\
                             --config_file ${config_path}\
                             --tagdb_cfg ${TAG_DB_CFG}\
                             --type $type\
                             --log_path $LOGFILE
    done
done