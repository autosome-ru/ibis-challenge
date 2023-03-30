import json
from pathlib import Path
from dataclasses import dataclass, asdict
from ..utils import replace_path2str

@dataclass
class ShadesConfig:
    balance: int
    max_dist: int
    
@dataclass
class ForeignConfig:
    balance: int
    foreigns_path: list[Path]
    
@dataclass
class GenomeSampleConfig:
    balance: int
    max_overlap: int | None 
    n_procs: int
    exact: bool
    precalc_profile: bool

@dataclass
class ChipSeqDSConfig:
    tf_name: str
    tf_path: list[Path]
    black_list_path: list[Path]
    friends_path: list[Path]
    window_size: int
    genome_path: Path
    seed: int 
    shades_cfg: ShadesConfig
    foreign_cfg: ForeignConfig
    genome_sample_cfg: GenomeSampleConfig
    
    @staticmethod
    def _convert_path(path: list[str | Path] | list[Path] | list[str] | Path | str) -> list[Path]:
        if not isinstance(path, list):
            if isinstance(path, str):
                path = Path(path)
            path = [path]
        path_lst = []
        for p in path:
            if isinstance(p, str):
                p = Path(p)
            p = p.absolute()
            if p.is_dir():
                path_lst.extend(p.iterdir())
            else:
                path_lst.append(p)
        return path_lst
    
    def save(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        if path.is_dir():
            path = path / f"{self.tf_name}.json"
        
        with open(path, "w") as out:
            json.dump(obj=replace_path2str(asdict(self)),
                      fp=out,
                      indent=4)
            
    @classmethod
    def load(cls, path: str | Path):
        with open(path, "r") as inp:
            dt = json.load(inp)
        dt["shades_cfg"] = ShadesConfig(**dt["shades_cfg"])
        dt["foreign_cfg"]['foreigns_path'] = [Path(x) for x in  dt["foreign_cfg"]['foreigns_path'] ]
        dt["foreign_cfg"] = ForeignConfig(**dt["foreign_cfg"])
        dt["genome_sample_cfg"] = GenomeSampleConfig(**dt["genome_sample_cfg"])
        
        dt["tf_path"] = [Path(x) for x in dt["tf_path"]]
        dt["black_list_path"] = [Path(x) for x in dt["black_list_path"]]
        dt["friends_path"] =      [Path(x) for x in dt["friends_path"]]
        dt["genome_path"] = Path(dt["genome_path"])
        return cls(**dt)
        
    @classmethod
    def make(cls,
             tf_name: str,
             tf_path: list[str | Path] | list[Path] | list[str] | Path | str,
             foreign_path: list[str | Path] | list[Path] | list[str] | Path | str,
             black_list_path: list[str | Path] | list[Path] | list[str] | Path | str,
             window_size: int,
             genome_path: str | Path,
             shades_balance: int,
             shades_max_dist: int,
             foreign_balance: int,
             genome_random_balance: int,
             precalc_profile: bool = False,
             friends_path: list[str | Path] | list[Path] | list[str] | Path | str | None = None,
             genome_n_procs: int = 1,
             genome_exact: bool = True,
             genome_max_overlap: int | None = True,
             seed: int = 777):
        if friends_path is None:
            friends_path = []
        if isinstance(genome_path, str):
            genome_path = Path(genome_path)
        tf_path = cls._convert_path(tf_path)
        foreign_path = cls._convert_path(foreign_path)
        friends_path = cls._convert_path(friends_path)
        black_list_path = cls._convert_path(black_list_path)
        
        shades_cfg = ShadesConfig(balance=shades_balance,
                                  max_dist=shades_max_dist)
        foreign_cfg = ForeignConfig(balance=foreign_balance,
                                    foreigns_path=foreign_path)
        genome_sample_cfg = GenomeSampleConfig(balance=genome_random_balance,
                                               max_overlap=genome_max_overlap,
                                               n_procs=genome_n_procs,
                                               exact=genome_exact,
                                               precalc_profile=precalc_profile)
        
        return cls(tf_name=tf_name,
                   tf_path=tf_path,
                   black_list_path=black_list_path,
                   friends_path=friends_path,
                   window_size=window_size,
                   genome_path=genome_path,
                   seed=seed,
                   shades_cfg=shades_cfg,
                   foreign_cfg=foreign_cfg,
                   genome_sample_cfg=genome_sample_cfg)