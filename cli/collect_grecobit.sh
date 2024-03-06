
BECNHMARK_PROCESSED="/home_local/dpenzar/BENCHMARK/BENCHMARK_PROCESSED"
BECNHMARK_CONFIGS="home_local/dpenzar/BENCHMARK/BENCHMARK_CONFIGS"

BENCHMARK_KIND="PBM"
for TYPE in "Leaderboard" "Final"; do
    python collect_benchmark.py --benchmark_root  ${BECNHMARK_PROCESSED}/${BENCHMARK_KIND}/${TYPE}/\
            --out_dir ${BECNHMARK_CONFIGS}/${BENCHMARK_KIND}/${TYPE}/\
            --benchmark_name ${BENCHMARK_KIND}_${TYPE}\
            --benchmark_kind ${BENCHMARK_KIND}\
            --scorers ../data/pbm_scorers.json
done

for BENCHMARK_KIND in "GHTS" "CHS"; do
    for TYPE in "Leaderboard" "Final"; do
        python collect_benchmark.py --benchmark_root ${BECNHMARK_PROCESSED}/${BENCHMARK_KIND}/${TYPE}/\
            --out_dir ${BECNHMARK_CONFIGS}/${BENCHMARK_KIND}/${TYPE}/\
            --benchmark_name ${BENCHMARK_KIND}_${TYPE}\
            --benchmark_kind ${BENCHMARK_KIND}\
           --scorers ../data/peaks_scorers.json
    done
done

BENCHMARK_KIND="SMS"
for TYPE in "Leaderboard" "Final"; do
    echo $TYPE
    python collect_benchmark.py --benchmark_root ${BECNHMARK_PROCESSED}/${BENCHMARK_KIND}/${TYPE}/\
            --out_dir ${BECNHMARK_CONFIGS}/${BENCHMARK_KIND}/${TYPE}/\
            --benchmark_name ${BENCHMARK_KIND}_${TYPE}\
            --benchmark_kind ${BENCHMARK_KIND}\
            --scorers ../data/sms_scorers.json
   break
done

BENCHMARK_KIND="HTS"
for TYPE in "Leaderboard" "Final"; do
    python collect_benchmark.py --benchmark_root ${BECNHMARK_PROCESSED}/${BENCHMARK_KIND}/${TYPE}/\
            --out_dir ${BECNHMARK_CONFIGS}/${BENCHMARK_KIND}/${TYPE}/\
            --benchmark_name ${BENCHMARK_KIND}_${TYPE}\
            --benchmark_kind ${BENCHMARK_KIND}\
            --scorers ../data/hts_scorers.json
done

python  generate_stagewise_pwmtemplates.py --benchmark_root ${BECNHMARK_PROCESSED} --templates_dir ${BECNHMARK_CONFIGS}