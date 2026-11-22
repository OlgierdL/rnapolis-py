import concurrent.futures
import csv
import math
import os
import optuna
import sys
sys.path.insert(0, os.path.abspath("src"))
from rnapolis.scorer import compute_score, evaluate_similarity
from scipy.stats import spearmanr
from tqdm import tqdm
import json


def eval_one(model_filename, dir_path, target_path):
    model_path = os.path.join(dir_path, model_filename)
    return model_filename, evaluate_similarity(target_path, model_path)


def normalise(weights):
    total = sum(weights)
    return [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)


def load_dataset(dir_name, measure_to_correlate="RMSD"):
    dir_path = os.path.join('values/casp-unified', dir_name)
    cache_path = os.path.join(dir_path, "metrics_cache.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            metrics_by_file = json.load(f)
    else:
        pdb_files = [f for f in os.listdir(dir_path) if f.endswith("pdb")]
        target_filename = next((f for f in pdb_files if "-" in f), None)

        target_path = os.path.join(dir_path, target_filename)
        metrics_by_file = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            futures = {
            executor.submit(eval_one, f, dir_path, target_path): f
            for f in pdb_files
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(pdb_files),
                desc=f"Precomputing {dir_path}",
            ):
                fname, metrics = future.result()
                metrics_by_file[fname] = metrics

        with open(cache_path, "w") as f:
            json.dump(metrics_by_file, f, indent=2, default=float)
        print(f"Saved metrics cache to {cache_path}")

    pairs = []
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            with open(os.path.join(dir_path, file), "r", newline="") as f:
                for row in csv.DictReader(f):
                    row_id = row["ID"] + ".pdb"
                    if row_id not in metrics_by_file:
                        continue
                    try:
                        val = float(row[measure_to_correlate])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if not math.isnan(val):
                        pairs.append((val, metrics_by_file[row_id]))

    return pairs


def make_objective(pairs):
    def objective(trial):
        wf = normalise([trial.suggest_float(f"wf_{i}", 0.0, 1.0) for i in range(2)])
        wc = normalise([trial.suggest_float(f"wc_{i}", 0.0, 1.0) for i in range(4)])
        wu = normalise([trial.suggest_float(f"wu_{i}", 0.0, 1.0) for i in range(6)])
        ws = normalise([trial.suggest_float(f"ws_{i}", 0.0, 1.0) for i in range(3)])

        x_vals = [val for val, _ in pairs]
        y_vals = [compute_score(m, wf, wc, wu, ws) for _, m in pairs]

        corr = spearmanr(x_vals, y_vals).statistic
        return float(corr) if not math.isnan(corr) else 0.0
    return objective


if __name__ == "__main__":
    measure = "RMSD"
    direction = "minimize"
    dir = "R1107"

    param_sets = [
        {
            "measure" : "RMSD",
            "direction" : "minimize",
            "dir" : "R1107"
        },
        {
            "measure" : "RMSD",
            "direction" : "minimize",
            "dir" : "R1108"
        },
        {
            "measure" : "RMSD",
            "direction" : "minimize",
            "dir" : "R1116"
        },
        {
            "measure" : "TM_score",
            "direction" : "maximize",
            "dir" : "R1107"
        },
        {
            "measure" : "TM_score",
            "direction" : "maximize",
            "dir" : "R1108"
        },
        {
            "measure" : "TM_score",
            "direction" : "maximize",
            "dir" : "R1116"
        },
        {
            "measure" : "INF all",
            "direction" : "maximize",
            "dir" : "R1107"
        },
        {
            "measure" : "INF all",
            "direction" : "maximize",
            "dir" : "R1108"
        },
        {
            "measure" : "INF all",
            "direction" : "maximize",
            "dir" : "R1116"
        },
        {
            "measure" : "LDDT",
            "direction" : "maximize",
            "dir" : "R1107"
        },
        {
            "measure" : "LDDT",
            "direction" : "maximize",
            "dir" : "R1108"
        },
        {
            "measure" : "LDDT",
            "direction" : "maximize",
            "dir" : "R1116"
        }
    ]

    for param_set in param_sets:
        dir = param_set["dir"]
        measure = param_set["measure"]
        direction = param_set["direction"]

        pairs = load_dataset(dir, measure_to_correlate=measure)

        study = optuna.create_study(direction=direction)
        study.optimize(make_objective(pairs), n_trials=500)

        p = study.best_params
        weights = {
            "WEIGHTS_FIT": normalise([p[f"wf_{i}"] for i in range(2)]),
            "WEIGHTS_CONCENTRATION": normalise([p[f"wc_{i}"] for i in range(4)]),
            "WEIGHTS_UNIFORMITY": normalise([p[f"wu_{i}"] for i in range(6)]),
            "WEIGHTS_COMPOSITE": normalise([p[f"ws_{i}"] for i in range(3)]),
        }
        result = {
            "dir": dir,
            "measure": measure,
            "direction": direction,
            "best_spearman_rho": study.best_value,
            "weights": weights,
        }

        output_file = f"optuna_results_{dir}_{measure}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)