import csv
import os
import subprocess
import concurrent.futures
from tqdm import tqdm
from scipy.stats import spearmanr
import math

def score_single_file(model_filename, dir_path, target_path):
    model_path = os.path.join(dir_path, model_filename)
    cmd = [
        "python3", "src/rnapolis/scorer.py", 
        '--model', model_path, 
        '--target', target_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    
    return model_filename, output

def process_directory(dir_name, measure_to_correlate="RMSD"):
    dir_path = os.path.join('values/casp-unified', dir_name)
    results = {}
    target_filename = ""
    
    pdb_files = [f for f in os.listdir(dir_path) if f.endswith('.pdb')]
    
    for file in pdb_files:
        if "-" in file:
            target_filename = file
            break
            
    if not target_filename:
        print(f"No target file found in {dir_path}")
        return

    target_path = os.path.join(dir_path, target_filename)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(score_single_file, file, dir_path, target_path): file 
            for file in pdb_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdb_files), desc=f"Scoring {dir_name}"):
            model_filename, output = future.result()
            
            tqdm.write(output)
            results[model_filename] = output
            
    for file in os.listdir(dir_path):
        if file.endswith('.csv'):
            csv_path = os.path.join(dir_path, file)
            
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                fieldnames = reader.fieldnames
                
            if 'Scorer' not in fieldnames:
                fieldnames.append('Scorer')
                
            for row in rows:
                row_id = row['ID'] + '.pdb'
                if row_id in results:
                    row['Scorer'] = results[row_id]
                    
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            x_vals, y_vals = [], []
            for row in rows:
                try:
                    x = float(row.get(measure_to_correlate, ""))
                    y = float(row.get("Scorer", ""))
                except (ValueError, TypeError):
                    continue

                if not (math.isnan(x) or math.isnan(y)):
                    x_vals.append(x)
                    y_vals.append(y)

            if len(x_vals) >= 2:
                res = spearmanr(x_vals, y_vals)
                print(
                    f"[{dir_name} | {file}] Spearman ({measure_to_correlate} vs."
                    f" Scorer, n={len(x_vals)}): rho={res.statistic:.4f},"
                    f" p={res.pvalue:.4e}"
                )
            else:
                print(
                    f"[{dir_name} | {file}] Not enough valid numeric pairs to compute"
                    " Spearman correlation."
                )

if __name__ == '__main__':
    dirs = ["R1138"]
    
    for dir_name in dirs:
        process_directory(dir_name)