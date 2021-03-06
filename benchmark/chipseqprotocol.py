import json

from abc import abstractmethod, ABCMeta
from attr import define, field
from collections import defaultdict
from pathlib import Path 
from typing import ClassVar, Union, List
from itertools import groupby

from datasetconfig import DatasetConfig
from beddata import BedData
from subprotocol import SubProtocol
from genome import Genome
from seqdb import SeqDB 
from dataset import Dataset
from experiment import ExperimentType
from beddata import join_bed
from bedentry import BedEntry
from beddata import BedData
from seqentry import SeqEntry
from labels import BinaryLabel
from chipseqdataset import ChIPSeqDataset


class NegativesSampler(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, prot: 'ChIPSeqIrisProtocol', cfg: DatasetConfig) -> BedData:
        raise NotImplementedError

# TODO: move sampling functionality and args directly to samplers 
class ForeignPeakSampler(NegativesSampler):
    def sample(self, prot: 'ChIPSeqIrisProtocol', cfg: DatasetConfig) -> BedData:
        return prot.get_foreign_peaks(cfg)


class ShadesSampler(NegativesSampler):
    def sample(self, prot: 'ChIPSeqIrisProtocol', cfg: DatasetConfig) -> BedData:
        return prot.get_shades(cfg)


@define
class ChIPSeqIrisProtocol(SubProtocol):   
    mx_dist: int
    peak_size: int
    shades_per_peak: int
    negative_tf_ratio: int 
    genome: Genome = field(repr=False)
    root: Path
    db: SeqDB = field(factory=SeqDB)
    
    TF_MERGE_DIR_NAME: ClassVar[str] = "TF_MERGE"
    TF_FOREIGN_DIR_NAME: ClassVar[str] = "TF_FOREIGN"
    REAL_PEAKS_FILE_NAME: ClassVar[str] = "real.bed"
    ALL_PEAKS_FILE_NAME: ClassVar[str] = "all.bed"

    MX_DIST_FIELD: ClassVar[str] = "mx_dist"
    PEAK_SIZE_FIELD: ClassVar[str] = "peak_size"
    SHADES_PER_PEAK_FIELD: ClassVar[str] = "shades_per_peak"
    NEGATIVE_TF_RATIO_FIELD: ClassVar[str] = "negative_ratio"
    ROOT_FIELD: ClassVar[str] = "root_dir"
    GENOME_FIELD: ClassVar[str] = "genome"

    DEFAULT_ROOT = Path(".ChIPSeq")

    @property
    def data_type(self) -> ExperimentType:
        return ExperimentType.ChIPSeq

    @property
    def tf_merge_dir(self) -> Path:
        return self.root / self.TF_MERGE_DIR_NAME

    @property
    def tf_foreign_dir(self) -> Path:
        return self.root / self.TF_FOREIGN_DIR_NAME

    @property
    def real_peaks_path(self) -> Path:
        return self.root / self.REAL_PEAKS_FILE_NAME
   
    def mkdirs(self):
        self.root.mkdir(exist_ok=True, parents=True)
        self.tf_merge_dir.mkdir(exist_ok=True, parents=True)
        self.tf_foreign_dir.mkdir(exist_ok=True, parents=True)
    
    def __attrs_post_init__(self):
        if self.peak_size % 2 != 1:
            raise Exception(f"Peak size must be odd: {self.peak_size}")
        self.mkdirs()

    def tf_merge_path(self, tf_name:str) -> Path:
        return self.tf_merge_dir / f"{tf_name}.bed"

    def tf_foreign_path(self, tf_name:str) -> Path:
        return self.tf_foreign_dir / f"{tf_name}.bed"

    def merge_by_tf(self, cfgs: list[DatasetConfig], real_peaks: BedData):
        cfgs.sort(key=lambda x : x.tf)
        for tf_name, tf_records in groupby(cfgs, key=lambda x : x.tf):
            beds = [BedData.from_file(x.path, header=True) for x in tf_records]
            jnd = join_bed(beds)
            mrg = jnd.merge()
            mrg_path = self.tf_merge_path(tf_name)
            mrg.write(mrg_path, write_peak=False)

            foreign = real_peaks.subtract(mrg, full=True)
            foreign_path = self.tf_foreign_path(tf_name)
            foreign.write(foreign_path, write_peak=True)

    @staticmethod
    def map_peaks(bed: BedData) -> dict[int, BedData]:
        dt = defaultdict(BedData)
        for e in bed:
            dt[e.peak].append(BedEntry(e.chr, e.start, e.end))
        return dict(dt)

    def sample_shades(self, ds: BedData, tf: BedData) -> BedData:
        flanks = ds.flank(self.genome, self.mx_dist + self.peak_size // 2)
        sbstr = flanks.subtract(tf, full=False)
        dt = self.map_peaks(sbstr)
        for peak, sgmnts in dt.items():
            sgmnts = sgmnts.apply(lambda s: s.truncate(self.peak_size // 2)).filter(lambda s: len(s) > 0)
            dt[peak] = sgmnts
        
        entries = []
        for peak, sgmnts in dt.items():
            smpl = sgmnts.sample_shades(seg_size=self.peak_size, k=self.shades_per_peak)
            entries.extend(smpl)
        return BedData(entries)

    def get_shades(self, cfg: DatasetConfig) -> BedData:
        ds = BedData.from_file(cfg.path, header=True)
        tf_file = self.tf_merge_path(cfg.tf)
        tf = BedData.from_file(tf_file, header=True)
        return self.sample_shades(ds, tf)
    
    def get_foreign_peaks(self, cfg: DatasetConfig) -> BedData:
        ds = BedData.from_file(cfg.path, header=True)
        foreign_path = self.tf_foreign_path(cfg.tf)
        foreign = BedData.from_file(foreign_path)
        size = len(ds) * self.negative_tf_ratio
        peaks = foreign.sample(size)
        cut_peaks = []
        for p in peaks:
            if p.peak is None:
                cnt = (p.end + p.start) // 2
            else:
                cnt = p.peak
            entry = BedEntry.from_center(p.chr, cnt, self.peak_size // 2)
            cut_peaks.append(entry)
        return BedData(cut_peaks, sorted=False)
    
    def make_real_peaks(self, cfgs: List[DatasetConfig]) -> BedData:
        beds = [BedData.from_file(x.path, header=True) for x in cfgs]
        real_bed = join_bed(beds)
        real_bed.write(self.real_peaks_path)
        return real_bed

    def preprocess(self, cfgs: List[DatasetConfig]) -> None:
        real_bed = self.make_real_peaks(cfgs)
        self.merge_by_tf(cfgs, real_bed)
        
    @classmethod
    def from_dt(cls, dt: dict) -> 'ChIPSeqIrisProtocol':
        mx_dist=int(dt[cls.MX_DIST_FIELD])
        peak_size = int(dt[cls.PEAK_SIZE_FIELD])
        shades_per_peak = int(dt[cls.SHADES_PER_PEAK_FIELD])
        negative_tf_ratio = int(dt[cls.NEGATIVE_TF_RATIO_FIELD])
        genome = Genome.from_dir(dt[cls.GENOME_FIELD])

        root  = dt.get(cls.ROOT_FIELD, cls.DEFAULT_ROOT)
        root = Path(root)
    
        return cls(mx_dist=mx_dist,
                   peak_size=peak_size,
                   shades_per_peak=shades_per_peak,
                   negative_tf_ratio=negative_tf_ratio,
                   genome=genome,
                   root=root)

    @classmethod
    def from_json(cls, path: Union[Path, str]) -> 'ChIPSeqIrisProtocol':
        if isinstance(path, str):
            path = Path(path)
        with path.open() as inp:
            dt = json.load(inp)
        lower_dt = {key.lower() : value for key, value in dt.items() }
        return cls.from_dt(lower_dt)

    def get_positives(self, cfg: DatasetConfig) -> list[SeqEntry]:
        peaks = BedData.from_file(cfg.path, header=True)
        cut_peaks = []
        for p in peaks:
            if p.peak is None:
                cnt = (p.end + p.start) // 2
            else:
                cnt = p.peak
            entry = BedEntry.from_center(p.chr, cnt, self.peak_size // 2)
            cut_peaks.append(entry)
        seqs = self.genome.cut(cut_peaks)
        entries = [SeqEntry(sequence=s, 
                            tag=self.db.add(s),
                            label=BinaryLabel.POSITIVE) for s in seqs]
        return entries

    def get_negatives(self, cfg: DatasetConfig, sampler: NegativesSampler) -> list[SeqEntry]:
        bed = sampler.sample(self, cfg)
        seqs = self.genome.cut(bed)
        entries = [SeqEntry(sequence=s, 
                        tag=self.db.add(s), 
                        label=BinaryLabel.NEGATIVE) for s in seqs]
        return entries

    def prepare_dataset(self, cfg: DatasetConfig, sampler: NegativesSampler, name: str):
        pos = self.get_positives(cfg)
        neg = self.get_negatives(cfg, sampler)
        return ChIPSeqDataset(name=name, 
                            tf_name=cfg.tf, 
                            type=cfg.ds_type, 
                            entries=pos+neg, 
                            metainfo=cfg.metainfo)

    @staticmethod
    def get_shades_ds_name(cfg: DatasetConfig) -> str:
        return f"{cfg.name}_shades"

    @staticmethod
    def get_foreign_ds_name(cfg: DatasetConfig) -> str:
        return f"{cfg.name}_foreign"

    def process(self, cfg: DatasetConfig) -> list[Dataset]:
        shades_ds = self.prepare_dataset(cfg,
                                         sampler=ShadesSampler(),
                                         name=self.get_shades_ds_name(cfg))
        foreign_ds = self.prepare_dataset(cfg,
                                          sampler=ForeignPeakSampler(),
                                          name=self.get_foreign_ds_name(cfg))
        return [shades_ds, foreign_ds]