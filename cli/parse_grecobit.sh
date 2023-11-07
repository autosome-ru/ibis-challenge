#python parse_grecobit_chipseq.py\
#       --genome /home_local/dpenzar/hg38\
#       --black_list_regions /home_local/dpenzar/ibis-challenge/benchmark/data/blacklist.bed\
#       --valid_hide_regions /home_local/dpenzar/centromers_split/chs_hide.bed
#python parse_grecobit_affiseq.py\
#       --genome /home_local/dpenzar/hg38\
#       --black_list_regions /home_local/dpenzar/ibis-challenge/benchmark/data/blacklist.bed\
#       --valid_hide_regions /home_local/dpenzar/centromers_split/ghts_hide.bed
python parse_grecobit_pbm.py --neg2pos_ratio 10
