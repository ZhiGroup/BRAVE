import nibabel as nib
import xml.etree.ElementTree as ET
import pickle as pkl
import multiprocessing
import subprocess
from os import system
import pandas as pd
import numpy as np
import pickle as pkl
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from paths import cfg

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

# --- Define the worker function ---
# This function will be executed by each parallel process.
# It takes one item from your dictionary and the other necessary variables.


def run_fsl_command(item, output_dir, fnlirt_path, cmd_template):
    """
    Worker function to process a single EID.
    """
    eid, warp_path = item
    save_path = os.path.join(output_dir, f"jac_{eid}.nii.gz")

    # Format the command string
    command_txt = cmd_template.format(
        fnlirt_path=fnlirt_path,
        warp_path=warp_path,
        jac_output_path=save_path
    )

    print(f"Running command for EID {eid}...")

    try:
        # Use subprocess.run to execute the command. It's safer and more modern.
        # We split the command into a list for subprocess.

        result = subprocess.run(command_txt.split(), check=True,
                                capture_output=True,
                                text=True)
        return (eid, "Success")

    except subprocess.CalledProcessError as e:

        print(f"--- ERROR processing EID {eid} ---")
        print(f"Stderr: {e.stderr}")
        return (eid, f"Failed: {e.stderr}")


def get_jacobian():
    inv_warp_dir = "<EXTERNAL: inverse warp-field dir (nonlinear MNI inverse transforms)>"

    fnlitr_path = "<EXTERNAL: FSL fnirtfileutils binary>"

    output_jac_dir = "<EXTERNAL: Jacobian-shape output dir>"

    warp_invs = [os.path.join(inv_warp_dir, f) for f in os.listdir(
        inv_warp_dir) if f.endswith('tsfm.nii.gz')]

    eid_warp_path_map = {p.split("/")[-1].split("_")[0]: p for p in warp_invs}

    disc_path = os.path.join(cfg.cohorts.dir, cfg.cohorts.discovery)
    rep_path = os.path.join(cfg.cohorts.dir, cfg.cohorts.replication)

    disc_df = pd.read_csv(disc_path)

    discovery_eid_set = set(disc_df['eid'])

    eid_warp_path_map_discovery = {
        eid: p for eid, p in eid_warp_path_map.items() if int(eid) in discovery_eid_set}

    command_template = """ {fnlirt_path} --in={warp_path} --jac={jac_output_path} """

    tasks = [(item, output_jac_dir, fnlitr_path, command_template)
             for item in eid_warp_path_map_discovery.items()]

    num_processes = multiprocessing.cpu_count()
    print(f"Starting parallel processing on {num_processes} cores...")

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.starmap to pass multiple arguments to the worker function
        results = pool.starmap(run_fsl_command, tasks)

    print("\n--- All processing complete! ---")

    # You can optionally check the results
    for eid, status in results:
        if "Failed" in status:
            print(f"EID {eid} failed.")


def get_harvard_xml_dict():

    # Path to the XML file (replace with your actual path)
    xml_file = "<EXTERNAL: FSL HarvardOxford-Subcortical.xml atlas label file>"

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # The note you mentioned is key here
    # The labels are off by one! maxprob labels start at 1, XML indices start at 0
    # So label_id = xml_index + 1

    # Create the dictionary
    label_mapping = {}
    for structure in root.findall('.//label'):
        # The 'index' attribute in the XML is an FSL-specific convention
        xml_index = int(structure.attrib['index'])

        # Get the integer label for the maxprob file
        label_id = xml_index + 1

        # Get the human-readable name of the structure
        region_name = structure.text

        # Store the mapping
        label_mapping[label_id] = region_name

    # Now, you have a dictionary that maps your integer labels to region names
    return label_mapping


first_region_dict = {


    10: 'Left_Thalamus_Proper',
    11: 'Left_Caudate',
    12: 'Left_Putamen',
    13: 'Left_Pallidum',
    16: 'Brain_Stem _or_4th_Ventricle',
    17: 'Left_Hippocampus',
    18: 'Left_Amygdala',
    26: 'Left_Accumbens-area',
    49: 'Right_Thalamus-Proper',
    50: 'Right_Caudate',
    51: 'Right_Putamen',
    52: 'Right_Pallidum',
    53: 'Right_Hippocampus',
    54: 'Right_Amygdala',
    58: 'Right_Accumbens-area'
}


def process_jacobian_file(region_id, template_label, jco_files, region_label):
    print(f"Working on {region_id, region_label}")
    ls = []
    for jac_file_path in jco_files:
        jac_file = nib.load(jac_file_path).get_fdata()
        eid = int(jac_file_path.split("/")
                  [-1].replace('jac_', '').replace('.nii.gz', ''))

        jac_values = jac_file[np.where(template_label == region_id)].tolist()

        ls.append([eid]+jac_values)
    return {region_label: ls}


def prepare_data_for_pca():

    label_mapping = get_harvard_xml_dict()

    d = {}
    for fsl_label in label_mapping.values():
        for mni_label in first_region_dict.values():
            if fsl_label in mni_label.replace("_", " "):
                d[fsl_label] = mni_label

    selected_regions = {k: v for k, v in label_mapping.items() if v in d}

    print(f"we got the following regions: {selected_regions}")

    template_label = nib.load(
        "<EXTERNAL: FSL HarvardOxford-sub-maxprob-thr50-1mm.nii.gz atlas>").get_fdata()

    output_jac_dir = "<EXTERNAL: Jacobian-shape output dir>"

    jco_files = [os.path.join(output_jac_dir, f)
                 for f in os.listdir(output_jac_dir)]

    args_list = [[region_id, template_label, jco_files, region_label]
                 for region_id, region_label in selected_regions.items()]

    num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:

        all_results = pool.starmap(process_jacobian_file, args_list)

        consolidated_data = {}

        for result_dict in all_results:
            consolidated_data.update(result_dict)

    save_path = "<EXTERNAL: PCA shape-input pickle output>"
    with open(save_path, 'wb') as f:
        pkl.dump(consolidated_data, f)


if __name__ == "__main__":
    prepare_data_for_pca()
