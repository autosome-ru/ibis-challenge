if [ -z "$1" ]; then
    echo "Benchmark root dir is not provided"
    exit 1
else
    BENCHMARK_ROOT=`realpath -m $1`
fi

if [ -z "$2" ]; then
    PWMEVAL_PATH=__PWM_EVAL_PWM_SCORING_PATH__
else
    PWMEVAL_PATH=`realpath -m $2`
fi

if [ -z "$3" ]; then
    ROOT_LABEL="__ROOTDIR__" 
else 
    ROOT_LABEL=$3
fi

for path in $BENCHMARK_ROOT/BENCHMARK_CONFIGS/*/*/benchmark.json; do
    python format_bench.py --in_benchmark_path $path\
         --root_dir $BENCHMARK_ROOT\
         --root_label $ROOT_LABEL\
         --pwmeval_path $PWMEVAL_PATH\
         --out_benchmark_path ${path}
done