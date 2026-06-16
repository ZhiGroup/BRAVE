import subprocess
import concurrent.futures
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from paths import cfg

# 1. Configuration
regions = [
    'Right_Amygdala', 'Left_Pallidum', 'Left_Caudate', 'Left_Putamen',
    'Left_Hippocampus', 'Right_Accumbens-area', 'Right_Pallidum', 'CSF',
    'Brain_Stem_or_4th_Ventricle', 'Right_Thalamus-Proper',
    'Left_Amygdala', 'Right_Putamen', 'Left_Accumbens-area',
    'Left_Thalamus_Proper'
]
# Removed duplicates from your list (Right_Hippocampus, Right_Caudate were repeated)

cohorts = ['replication'] #['discovery', 'replication']

# Base path for checking existing outputs
base_output_path = cfg.gwas.jagwas_out

# Create a logs directory
log_dir = "<EXTERNAL: local directory for per-region run logs>"
os.makedirs(log_dir, exist_ok=True)


def run_r_script(region, cohort):
    """Checks for existing correlation matrix and runs R script if missing."""
    # Define the expected output file for this specific region/cohort
    expected_output = os.path.join(
        base_output_path,
        f"output_{cohort}",
        region,
        f"{region}_residuals_cor.txt"
    )

    # Check if the process was already completed
    if os.path.exists(expected_output):
        return f"SKIP: {region} ({cohort}) - Output already exists."

    log_file = os.path.join(log_dir, f"log_{region}_{cohort}.txt")
    print(f"Starting: {region} ({cohort})")

    # Command to run the R script with arguments
    # Ensure jagwas_phenotype_correlation_command_line.R handles the iid vs IID fix
    cmd = ["Rscript", "jagwas_phenotype_correlation_cli.R", region, cohort]

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        return f"SUCCESS: {region} ({cohort})"
    else:
        return f"FAILED: {region} ({cohort}) - check {log_file}"


# 2. Run in Parallel
# max_workers=20 is efficient given your subsetted GRM files are small (5MB)
with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    tasks = [(r, c) for r in regions for c in cohorts]

    futures = [executor.submit(run_r_script, region, cohort)
               for region, cohort in tasks]

    for future in concurrent.futures.as_completed(futures):
        print(future.result())
