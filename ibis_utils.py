import parse
from pathlib import Path

def ibis_default_name_parser():
    return parse.compile("{tf_name}.{fl}@{exp_type}@{name}@Peaks.{unique_tag}.{dt_type}.peaks")

def ibispath2info(path: str | Path, 
                    name_parser=ibis_default_name_parser()) -> dict[str, str]:
    if isinstance(path, str):
        path = Path(path)

    data = name_parser.parse(path.name)
    if data is None or isinstance(data, parse.Match):
            raise Exception("Wrong peakfile name format")
      
    dt ={
            "name": data["name"],
            "tf": data['tf_name'],
            "ds_type": data['dt_type'],
            "path": path.absolute()
        }
    
    return dt
