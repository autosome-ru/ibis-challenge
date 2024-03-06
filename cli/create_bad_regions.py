import logging
import argparse 
import sys 

parser = argparse.ArgumentParser()
parser.add_argument("--bibis_root",
                    default="/home_local/dpenzar/bibis_git/ibis-challenge",
                    type=str)
parser.add_argument("--Nmask", 
                    type=str,
                    required=True)
parser.add_argument("--encode_blacklist", 
                    type=str,
                    required=True)
parser.add_argument("--bad_regions", 
                    type=str,
                    required=True)
parser.add_argument("--log_path", 
                    type=str,
                    default="log.txt")
parser.add_argument("--log_name",
                    type=str,
                    default="create_nmask")
parser.add_argument("--bedtools", 
                    default="/home_local/dpenzar/bedtools2/bin",
                    type=str)
args = parser.parse_args()

sys.path.append(args.bibis_root)

from bibis.bedtools.beddata import BedData, join_bed
from bibis.logging import get_logger, BIBIS_LOGGER_CFG
BIBIS_LOGGER_CFG.set_path(path=args.log_path)
from bibis.bedtools.bedtoolsexecutor import BedtoolsExecutor
BedtoolsExecutor.set_defaull_executor(args.bedtools)

logger = get_logger(args.log_name, args.log_path)

logger.info("Reading Ns mask file")
nmask = BedData.from_file(args.Nmask)

logger.info("Reading ENCODE blacklist regions")
black_list = BedData.from_file(args.encode_blacklist)

logger.info("Joining bad regions")
bad_list = join_bed([black_list, nmask]).merge()
bad_list.sort()

logging.info("Writing bad regions")
bad_list.write(args.bad_regions, write_peak=False)