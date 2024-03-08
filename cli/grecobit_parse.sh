OUT_DIR="/home_local/dpenzar/BENCHMARK/BENCHMARK_DATA"
python parse_grecobit_pbm.py --neg2pos_ratio 10\
       --out_dir ${OUT_DIR}/PBM
python parse_grecobit_chipseq.py\
      --genome /home_local/dpenzar/final_genome/hg38/\
       --black_list_regions /home_local/dpenzar/bibis_git/ibis-challenge/data/Ns_ENCODE_blacklist.bed\
       --out_dir ${OUT_DIR}/CHS/
python parse_grecobit_affiseq.py\
       --genome /home_local/dpenzar/final_genome/hg38/\
       --black_list_regions  /home_local/dpenzar/bibis_git/ibis-challenge/data/Ns_ENCODE_blacklist.bed\
       --out_dir ${OUT_DIR}/GHTS
python parse_grecobit_smileseq.py --out_dir ${OUT_DIR}/SMS/RAW
python preprocess_smileseq.py --in_dir ${OUT_DIR}/SMS/RAW\
    --out_dir ${OUT_DIR}/SMS
python parse_grecobit_htselex.py --out_dir ${OUT_DIR}/HTS

