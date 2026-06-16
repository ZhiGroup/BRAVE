import os
import subprocess
import concurrent.futures

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from paths import cfg

# 1. Configuration
regions = [
    "Left_Putamen", "Left_Pallidum", "Right_Hippocampus", "Left_Hippocampus",
    "Right_Amygdala", "Brain_Stem_or_4th_Ventricle", "Left_Thalamus_Proper",
    "Left_Accumbens-area", "Right_Putamen", "CSF", "Left_Amygdala",
    "Left_Caudate", "Right_Accumbens-area", "Right_Caudate",
    "Right_Thalamus-Proper", "Right_Pallidum"
]

cohort = "replication" # "discovery"  #
base_cor_dir = os.path.join(cfg.gwas.jagwas_out, f"output_{cohort}")
log_dir = "<EXTERNAL: local directory for per-region run logs>"
os.makedirs(log_dir, exist_ok=True)


def run_region(region):
    """Checks for existing output and runs JAGWAS if missing."""
    # Define expected output file path
    output_file = os.path.join(
        base_cor_dir, region, f"{region}_JAGWAS_results.txt")
    output_file_zipped = os.path.join(
        base_cor_dir, region, f"{region}_JAGWAS_results.txt.gz")
    log_file = os.path.join(log_dir, f"log_{region}_{cohort}.txt")

    # Skip logic
    if os.path.exists(output_file) or os.path.exists(output_file_zipped):
        return f"SKIP: {region} (Results already exist at {output_file})"

    print(f"STARTING: {region}...")

    # Execute the verified bash script
    # We capture output to individual log files for debugging
    cmd = ["bash", "jagwas_association_run.sh", region, cohort]

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        return f"SUCCESS: {region}"
    else:
        return f"FAILED: {region} (Check {log_file})"


# 2. Parallel Execution
# Adjust max_workers based on RAM (e.g., 4 or 8)
with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    future_to_region = {executor.submit(run_region, r): r for r in regions}

    for future in concurrent.futures.as_completed(future_to_region):
        print(future.result())
