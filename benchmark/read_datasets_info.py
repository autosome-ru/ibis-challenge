from collections import Counter
import json
from pathlib import Path
import pandas as pd
if __name__ == "__main__":
    df = pd.read_table("/home_local/dpenzar/outer_benchmark.tsv")
    tfs = set(df['TF'])
    pbms = set()
    for p in df['pbm']:
        if p == "__fail" or p == "__nodata":
            continue
        pbms.update(p.split(",")) 
    MOTIFS_DIR = Path('/home_local/vorontsovie/greco-motifs/release_7d_motifs_2021-12-21/')
    datasets = []
    with open('/home_local/dpenzar/previous_metrics.json') as inp:
        dt = json.load(inp)
    ds_types = []
    motif_models = []
    added = set()
    for tf_name, tf_dt in dt.items():
        if tf_name not in tfs:
            continue
        for motif, motif_dt in tf_dt.items():
            if not motif.endswith(".ppm"):
                continue
            motif_info = {"tf": tf_name,
                          "name": motif,
                          "path": str(MOTIFS_DIR / motif)}
            motif_models.append(motif_info)
            for exp_type, exp_tp_dt in motif_dt.items():
                if exp_type != "PBM":
                    continue
               
                for exp_id, exp_dt in exp_tp_dt.items():
                    if exp_id not in pbms:
                        continue
                    for ds_name, ds_dt in exp_dt.items():
                        for pr_type, pr_lst in ds_dt.items():
                            if pr_type != "QNZS":
                                continue
                            tag = (exp_id, pr_type)
                            if tag in added:
                                continue
                            added.add(tag)
                            fields = ds_name.split(".")
                            fields = ds_name.split("@")
                            pbm_type = fields[1].replace("PBM.", '')
                            ds_types.append(pbm_type)
                            path = f'/home_local/vorontsovie/greco-data/release_7a.2021-10-14/full/PBM.{pr_type}/Val_intensities/{ds_name}'
                            dataset_entry = {"name": exp_id,
                                            "exp_type": exp_type,
                                            "tf": tf_name,
                                            "ds_type": "test",
                                            "path":  path,
                                            "curation_status": "NOT_CURATED",
                                            "protocol": "iris",
                                            "pbm_type": pbm_type,
                                            "preprocessing": pr_type,
                                            "metainfo":{
                                            }}
                            datasets.append(dataset_entry)

    print(set(ds['preprocessing'] for ds in datasets))
    print(len(pbms), len(datasets))
    for ind, ds in enumerate(datasets):
        if ds['name'] == "PBM13944":
            print('Dataset ind', ind)

    print(len(motif_models))
    with open("/home_local/dpenzar/models.json", "w") as outp:
        json.dump(motif_models, outp, indent=4)
    benchmark_config = {}
    benchmark_config['name'] = "consistency_test"
    benchmark_config['pwmeval'] = "/home_local/dpenzar/PWMEval/pwm_scoring"
    benchmark_config['scorers'] = [
        {
            "name": "scikit_rocauc",
            "alias": "rocauc",
            "params": {}
        },
        {
            "name": "scikit_prauc",
            "alias": "prauc",
            "params": {}
        },
        {
            "name": "prroc_prauc",
            "alias": "dg_prauc",
            "params": {"type": "davisgoadrich"}
        },
        {
            "name": "constant_scorer",
            "alias": "cons50",
            "params": {
                "cons": 0.5
            }
        },
        {
            "name": "constant_scorer",
            "alias": "cons90",
            "params": {
                "cons": 0.9
            } 
        }
    ]
    print(len(datasets))
    benchmark_config["results_dir"] = "/home_local/dpenzar/test_results_parallel"
    benchmark_config['datasets'] = datasets
    with open("/home_local/dpenzar/test_config.json", "w") as outp:
        json.dump(benchmark_config, outp, indent=4)
