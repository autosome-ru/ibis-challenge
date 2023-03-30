import parse
from pathlib import Path
from ..datasetconfig import DatasetConfig

def ibis_default_name_parser():
    return parse.compile("{tf_name}.{fl}@{exp_type}@{name}@Peaks.{unique_tag}.{dt_type}.peaks")

def chip_ibisname2config(path: str | Path, 
                    name_parser=ibis_default_name_parser()) -> DatasetConfig:
    if isinstance(path, str):
        path = Path(path)

    path = path.absolute()
    data = name_parser.parse(path.name)
    if data is None or isinstance(data, parse.Match):
            raise Exception("Wrong peakfile name format")
      
    dt ={
            "name": data["name"],
            "exp_type": "ChIPSeq",
            "tf": data['tf_name'],
            "ds_type": data['dt_type'],
            "path": path,
            "curation_status": "accepted",
            "protocol": "iris",
            "metainfo": {}
        }
    
    return DatasetConfig.from_dict(dt)