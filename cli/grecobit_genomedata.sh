DATA_DIR="../data"
GENOME_PATH=https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.chromFa.tar.gz
GENOME_GZ_OUT_PATH=${DATA_DIR}/hg38.tar.gz
GENOME_OUT_PATH=${DATA_DIR}/hg38
Ns_PATH=${DATA_DIR}/hg38_Ns.bed
BLACK_REGIONS_PATH=${DATA_DIR}/hg38-blacklist.v2.bed
BAD_REGIONS_PATH=${DATA_DIR}/Ns_ENCODE_blacklist.bed
CENTROMERS_PATH=${DATA_DIR}/hg38_centromers.bed
CENTROMERS_SPLIT_DIR=${DATA_DIR}/centromers_split

mkdir -p $DATA_DIR

if [ ! -d $GENOME_OUT_PATH ] ; then
    echo "downloading genome" 
    wget $GENOME_PATH -O $GENOME_GZ_OUT_PATH
    tar -xzvf $GENOME_GZ_OUT_PATH -C $DATA_DIR .
    rm $GENOME_GZ_OUT_PATH
    mv ${DATA_DIR}/chroms $GENOME_OUT_PATH
    #keep autosomes
    find $GENOME_OUT_PATH -regextype posix-egrep \! -regex ".*chr[0-9]+\.fa" -regex '.*.fa' -delete
fi

if [ ! -f $Ns_PATH ] ; then
    echo "Creating Nmask for genome"
    python create_Nmask.py --genome_dir $GENOME_OUT_PATH --Nmask $Ns_PATH
fi 

if [ ! -f $BAD_REGIONS_PATH ] ; then
    echo "Writing final bad regions file"
    python create_bad_regions.py --Nmask ${Ns_PATH}\
        --encode_blacklist ${BLACK_REGIONS_PATH}\
        --bad_regions ${BAD_REGIONS_PATH}
fi 

if [ ! -d $CENTROMERS_SPLIT_DIR ] ; then
    echo "Creating by-centromere split"
    python create_by_centromere_split.py --centromers_file $CENTROMERS_PATH\
        --genome $GENOME_OUT_PATH\
        --out_path ${CENTROMERS_SPLIT_DIR}
fi