
BECNHMARK_PROCESSED="/home_local/dpenzar/BENCHMARK_APRIL_FINAL/BENCHMARK_PROCESSED"
BECNHMARK_CONFIGS="/home_local/dpenzar/BENCHMARK_APRIL_FINAL/BENCHMARK_CONFIGS"
DATA_DIR="../data"
SCORER_DIR=${DATA_DIR}/scorers/
LOGFILE="collect.log"

BENCHMARK_KIND="PBM"
for TYPE in "Final"; do
    python collect_benchmark.py --benchmark_root  ${BECNHMARK_PROCESSED}/${BENCHMARK_KIND}/${TYPE}/\
            --out_dir ${BECNHMARK_CONFIGS}/${BENCHMARK_KIND}/${TYPE}/\
            --benchmark_name ${BENCHMARK_KIND}_${TYPE}\
           --benchmark_kind ${BENCHMARK_KIND}\
           --scorers ${SCORER_DIR}/pbm_scorers.json\
           --log_path $LOGFILE
done

for TYPE in "Final"; do
    for BENCHMARK_KIND in "GHTS" "CHS"; do
        echo ${BENCHMARK_KIND}_${TYPE}
        python collect_benchmark.py --benchmark_root ${BECNHMARK_PROCESSED}/${BENCHMARK_KIND}/${TYPE}/\
            --out_dir ${BECNHMARK_CONFIGS}/${BENCHMARK_KIND}/${TYPE}/\
            --benchmark_name ${BENCHMARK_KIND}_${TYPE}\
            --benchmark_kind ${BENCHMARK_KIND}\
            --scorers ${SCORER_DIR}/peaks_scorers.json\
            --log_path $LOGFILE
    done
done

BENCHMARK_KIND="SMS"
for TYPE in "Final"; do
    python collect_benchmark.py --benchmark_root ${BECNHMARK_PROCESSED}/${BENCHMARK_KIND}/${TYPE}/\
            --out_dir ${BECNHMARK_CONFIGS}/${BENCHMARK_KIND}/${TYPE}/\
            --benchmark_name ${BENCHMARK_KIND}_${TYPE}\
            --benchmark_kind ${BENCHMARK_KIND}\
            --scorers ${SCORER_DIR}/sms_scorers.json\
            --log_path $LOGFILE
done

BENCHMARK_KIND="HTS"
for TYPE in "Final"; do
   python collect_benchmark.py --benchmark_root ${BECNHMARK_PROCESSED}/${BENCHMARK_KIND}/${TYPE}/\
           --out_dir ${BECNHMARK_CONFIGS}/${BENCHMARK_KIND}/${TYPE}/\
           --benchmark_name ${BENCHMARK_KIND}_${TYPE}\
           --benchmark_kind ${BENCHMARK_KIND}\
           --scorers ${SCORER_DIR}/hts_scorers.json\
           --log_path $LOGFILE
done

python  generate_stagewise_pwmtemplates.py --benchmark_root ${BECNHMARK_PROCESSED}\
    --templates_dir ${BECNHMARK_CONFIGS}\
    --log_path $LOGFILE