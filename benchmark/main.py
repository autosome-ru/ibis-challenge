
from pathlib import Path
from _benchmark import BenchmarkConfig

BENCHMARK_CONFIG = Path("/home_local/dpenzar/ibis-challenge/benchmark/benchmark.json")

if __name__ == '__main__':
    benchmark = BenchmarkConfig\
                .from_json(BENCHMARK_CONFIG)\
                .make_benchmark()
    print(benchmark)