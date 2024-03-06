import logging
import tqdm
import argparse 
import sys 

parser = argparse.ArgumentParser()
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
parser.add_argument("--genome_dir",
                     type=str,
                       required=True)
parser.add_argument("--Nmask", 
                    type=str,
                    required=True)
parser.add_argument("--log_path", 
                    type=str,
                    default="log.txt")
parser.add_argument("--log_name",
                    type=str,
                    default="create_nmask")
args = parser.parse_args()

sys.path.append(args.bibis_root)

from bibis.seq.genome import Genome
from bibis.bedtools.beddata import BedData
from bibis.logging import get_logger, BIBIS_LOGGER_CFG
BIBIS_LOGGER_CFG.set_path(path=args.log_path)

logger = get_logger(args.log_name, args.log_path)

logger.info("Reading genome")
genome = Genome.from_dir(args.genome_dir)

logging.info("Writing Nmask")
with open(args.Nmask, "w") as out:
    for ch, seq in tqdm.tqdm(genome.chroms.items()):
        start_N = -1
        for i in range(len(seq)):
            c = seq[i]
            if c == "N" or c == "n":
                if start_N == -1:
                    start_N = i
                # else nothing to do
            else:
                if start_N != -1:
                    end_N = i
                    print(ch, start_N, end_N, -1, file=out, sep="\t")
                    start_N = -1
        if start_N != -1:
            end_N = len(seq)
            print(ch, start_N, end_N, -1, file=out, sep="\t")
            start_N = -1

bed = BedData.from_file(args.Nmask)
bed.sort()
bed.write(args.Nmask, write_peak=False)