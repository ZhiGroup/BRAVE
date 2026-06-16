

import pixel_embedding_records
import re
from glob import glob
from subprocess import check_output, STDOUT
from itertools import zip_longest
# from tqdm import tqdm
# from matplotlib import pyplot as plt
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl


import argparse
from glob import glob
from multiprocessing import Pool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from paths import cfg

fastgwas_root_dir = cfg.gwas.fastgwa_out
print('reloaded')


def create_combined_mimp(gm_min_p_discovery_path, wm_min_p_discovery_path, csf_min_p_discovery_path, save_dir, cohort_type='discovery'):

    gm = pd.read_table(gm_min_p_discovery_path, compression='gzip') if 'gz' in gm_min_p_discovery_path else pd.read_table(
        gm_min_p_discovery_path)
    wm = pd.read_table(wm_min_p_discovery_path, compression='gzip') if 'gz' in wm_min_p_discovery_path else pd.read_table(
        wm_min_p_discovery_path)
    csf = pd.read_table(csf_min_p_discovery_path, compression='gzip') if 'gz' in csf_min_p_discovery_path else pd.read_table(
        csf_min_p_discovery_path)
    gm['region'] = 'GM'
    wm['region'] = 'WM'
    csf['region'] = 'CSF'
    # align
    gm = gm.loc[gm.groupby('SNP')['P'].idxmin()]
    wm = wm.loc[wm.groupby('SNP')['P'].idxmin()]
    csf = csf.loc[csf.groupby('SNP')['P'].idxmin()]

    snps_gm = gm['SNP'].values.tolist()
    snps_wm = wm['SNP'].values.tolist()
    snps_csf = csf['SNP'].values.tolist()

    snps_gm.extend(snps_wm)
    snps_gm.extend(snps_csf)
    snps = list(set(snps_gm))

    gm = gm.set_index('SNP')
    wm = wm.set_index('SNP')
    csf = csf.set_index('SNP')

    gm = gm.loc[snps]
    wm = wm.loc[snps]
    csf = csf.loc[snps]
    gm = gm.reset_index()
    wm = wm.reset_index()
    csf = csf.reset_index()

    n_snps, n_features = wm.shape
    n_snps, n_features
    combined = np.hstack((gm.values, wm.values, csf.values)
                         ).reshape(n_snps, -1, n_features)
    min_p_indices = np.argmin(combined[:, :, -4], axis=1)
    output = combined[np.arange(combined.shape[0]), min_p_indices, :]

    output_df = pd.DataFrame(output, columns=gm.columns)
    save_name = os.path.join(save_dir, f'{cohort_type}_com_gm_wm_csf')
    output_df.to_csv(save_name, index=False, sep="\t")
    print(f"saved in {save_name}")


def job(x, cmd1, fast_gwas_save_dir_name, modality, cohort_type):
    reg_name = os.path.basename(x)
    sum_state_file_name = reg_name.replace('csv', '').replace(".", "_")
    # cluster_index = re.seach(r"_cluster_ID_(\d+)_QT", reg_name).group(1)

    out_file_name = f"{fast_gwas_save_dir_name}/{reg_name}.fastGWA"
    if not os.path.exists(out_file_name):

        cmd = cmd1.format(
            x,
            os.path.join(cfg.gwas.covar_dir, f"{modality}_ccovar_{cohort_type}_v2"),
            os.path.join(cfg.gwas.covar_dir, f"{modality}_qcovar_{cohort_type}_v2"),
            f"{out_file_name}")
        print(cmd)
        os.system(cmd)


# def get_data_for_torwgwas(list_of_embeddings_path):
#     with open(list_of_embeddings_path, 'rb') as f:
#         d = pkl.load(f)
#     phenos, eids = [], []
#     for p, eid in d:
#         phenos.extend(p)
#         eids.extend(eid)
#     eids = [i.item() for i in eids]

#     embed_dict = dict(zip(eids, phenos))
#     df = pd.DataFrame.from_dict(embed_dict, orient='index')

#     return df
def get_data_for_torwgwas(subject_dict_path):

    with open(subject_dict_path, 'rb') as f:

        d = pkl.load(f)
        ls = []
        for sample_id, (_, arr) in d.items():
            ls.append((sample_id, arr.tolist()))

    sample_dict = dict(ls)
    df = pd.DataFrame.from_dict(sample_dict, orient='index')
    return df


class RunFastGWA:
    def __init__(self,
                 cluster_embedding_path,
                 cohort_type,
                 project_name='CLUSTER_GWAS',
                 prefix=None):

        self.cluster_embedding_path = cluster_embedding_path

        self.cohort_type = cohort_type
        self.project_name = project_name
        self.prefix = prefix
        # save_dir = os.path.join(project_root_dir, 'output','GWAS','embeddings')

        self.embedding_path = cluster_embedding_path

        self.fast_gwa_embedding_save_dir = os.path.join(fastgwas_root_dir,
                                                        self.project_name, prefix,cohort_type,
                                                        'input')
        os.makedirs(self.fast_gwa_embedding_save_dir, exist_ok=True)

        self.fastgwa_output_dir_path = os.path.join(fastgwas_root_dir,
                                                    self.project_name, prefix,cohort_type,
                                                    'output'
                                                    )
        os.makedirs(self.fastgwa_output_dir_path, exist_ok=True)

        print(f"Input Embedding Path: {self.embedding_path}")
        print(
            f"FastGWA input files will be saved in :{self.fast_gwa_embedding_save_dir}")
        print(
            f"FastGWA output files will be saved in : {self.fastgwa_output_dir_path}")
        self.region_paths = self.get_region_paths()
        self.fastgwa_break_pheno_dic = {}

    def get_region_paths(self):
        region_name = [os.path.join(self.embedding_path, f)
                       for f in os.listdir(self.cluster_embedding_path)]
        return region_name

    def prepare_samples(self):

        # do sample preparation for all clusters
        for region_path in self.region_paths:
            base_name = os.path.basename(region_path)
            base_name = "_".join(base_name.split("_")[1:-1])
            print(f"preparing data df for {base_name} with path {region_path}")
            df = get_data_for_torwgwas(region_path)

            self.prepare_input_fast_gwa(df, base_name)
            print()

    def get_region_names_finished_gwas(self):
        search_str = rf"{self.cohort_type}_.*?_{self.prefix}_(.*)_QT.*.fastGWA.fastGWA"
        done_regions = set([re.findall(search_str, s)[0] for s in os.listdir(
            self.fastgwa_output_dir_path) if '.log' not in s])
        return done_regions

    def prepare_input_fast_gwa(self, df, basename):

        for col_name, col_content in df.items():
            pheno_df = pd.DataFrame(
                {'fiid': col_content.index, 'iid': col_content.index, f"QT{col_name}": col_content})
            pheno_save_name = f"{self.cohort_type}_{self.project_name}_{self.prefix}_{basename}_QT{col_name}"
            save_path = os.path.join(
                self.fast_gwa_embedding_save_dir, pheno_save_name)
            pheno_df.to_csv(save_path, sep="\t", index=False)
        print(
            f"Done prepare input for fast gwa. Saved in {os.path.dirname(save_path)}")

    def remove_whitespace_from_filename(self):
        files = sorted(
            glob(f"{self.fast_gwa_embedding_save_dir }/{self.cohort_type}_{self.project_name}_{self.prefix}**"))
        print(f"Got {len(files)} files")

        for file_path in files:
            if " " in file_path:
                basename = os.path.basename(file_path)

                whitespace_removed = re.sub(r"(\s+)", "", basename)
                new_path = os.path.join(os.path.dirname(
                    file_path), whitespace_removed)

                os.rename(file_path, new_path)
                print(f"{file_path}=>{new_path}")

    def run_fastgwa(self):
        print("Removing any whitespace in file name")
        self.remove_whitespace_from_filename()

        print("Performing FastGWA")

        l = sorted(
            glob(f"{self.fast_gwa_embedding_save_dir }/{self.cohort_type}_{self.project_name}_{self.prefix}**"))
        print(f"Total file found: {len(l)}")
        # l = list(filter(lambda x: "Brain_Stem_or_4th_Ventricle" not in x, l))
        # print(f"without brain stem, we got {len(l)} files")
        cmd1 = f"{cfg.tools.gcta} --bgen {cfg.ukb.bgen} --sample {cfg.ukb.bgen_sample} --grm-sparse {cfg.gwas.sparse_grm}  --fastGWA-mlm --pheno {{}} --covar {{}} --qcovar {{}} --thread-num 256 --seed 0 --out {{}}"

        # cmd1 = """
        #         {cfg.tools.gcta}
        #         --bgen {cfg.ukb.bgen}
        #         --sample {cfg.ukb.bgen_sample}
        #         --grm-sparse {cfg.gwas.sparse_grm}
        #         --fastGWA-mlm
        #         --pheno {}
        #         --covar {}
        #         --qcovar {}
        #         --thread-num 512
        #         --seed 0
        #         --out {}
        #         """.strip()

        modality = 'T1'
        cohort_type = self.cohort_type
        fast_gwas_save_dir_name = self.fastgwa_output_dir_path
        if not os.path.exists(fast_gwas_save_dir_name):
            os.makedirs(fast_gwas_save_dir_name, exist_ok=True)

        args = [(x, cmd1, self.fastgwa_output_dir_path,
                 'T1', self.cohort_type) for x in l]
        with Pool(1) as p:
            p.starmap(job, args)
        print(f"FastGWA is done. saved in {self.fastgwa_output_dir_path}")

    def do_minP(self):
        # create minpoutput
        self.minP_output_dir = os.path.join(fastgwas_root_dir,
                                            self.project_name, self.prefix,self.cohort_type,
                                            'minp0utput'
                                            )
        os.makedirs(self.minP_output_dir, exist_ok=True)

        done_regions = self.get_region_names_finished_gwas()
        print(f"done_regions: {done_regions}")
        print(f"MinP will be saved in {self.minP_output_dir}")
        for region in done_regions:

            print(f"doing minP for {region}")
            sumstats_files = [os.path.join(self.fastgwa_output_dir_path, s) for s in os.listdir(
                self.fastgwa_output_dir_path) if '.log' not in s and region in s]

            minP_cls = minP(
                fastgwa_output_dir=self.fastgwa_output_dir_path,
                minP_output_dir=self.minP_output_dir,
                cohort_type=self.cohort_type,
                pcol=1,
                mode='min',
                input_prefix=f"{self.prefix}_{region}_"
            )

            minP_cls.run_minP(sumstats_files)
            print(f"Done MinP for {region}")


def convert2float(x):
    try:
        return float(x)
    except:
        return 1.0


def extract_col(x, offset=0):
    print(x)
    out = check_output(f"awk '{{print $(NF-{offset})}}' '{x}'",
                       universal_newlines=True, shell=True, stderr=STDOUT)
    return np.array(list(map(convert2float, out.strip('\n').split('\n')[1:]))), x


def create_minP(glob_list, pcol=0, mode='min'):
    # pcol is indexed from right to left, 0 means last col

    if mode == 'min':
        op = np.argmin

    elif mode == 'max':
        op == np.argmax

    else:
        raise Exception('not implemented')

    batch = 50
    for i in tqdm(range(0, len(glob_list), batch)):
        with Pool(batch) as q:
            result = q.starmap(extract_col, zip_longest(
                glob_list[i:i+batch], (), fillvalue=pcol))
        pnew, fnew = list(zip(*result))
        if i == 0:
            pnew = np.vstack(pnew)
            idx = op(pnew, 0)
            f = np.array(fnew, dtype=object)[idx]
        else:
            pnew = list(pnew)
            pnew.append(p)
            pnew = np.vstack(pnew)
            idx = op(pnew, 0)
            mask = (idx != (pnew.shape[0]-1))
            f[mask] = np.array(fnew, dtype=object)[idx[mask]]
        p = pnew[idx, np.arange(pnew.shape[1])]
    return p, f


def get_dim(str_fl):

    str_fl = os.path.basename(str_fl)
    pattern = r'QT(\d+).'
    match = int(re.search(pattern, str_fl).group(1))
    return match


class minP:

    def __init__(self,
                 fastgwa_output_dir,
                 minP_output_dir,
                 cohort_type='discovery',
                 pcol=0,
                 mode='min',
                 input_prefix=None,
                 ):

        self.cohort_type = cohort_type

        self.fastgwa_output_dir = fastgwa_output_dir
        self.minP_output_dir = minP_output_dir
        self.output_save_prefix = f"{self.cohort_type}_{input_prefix}"
        self.pcol = pcol
        self.mode = mode
        self.input_prefix = input_prefix
        print(f" self.cohort_type: { self.cohort_type}")
        print(f" self.fastgwa_output_dir: { self.fastgwa_output_dir}")
        print(f" self.input_prefix: { self.input_prefix}")

    def run_minP(self, sumstats=[]):
        if len(sumstats) == 0:
            sum_files = sorted(
                glob(f"{self.fastgwa_output_dir}/{self.cohort_type}*_weight_{self.input_prefix}*.fastGWA"))

            print(f"find {len(sum_files)} number of summary stat files")
        else:
            sum_files = sumstats
            print(
                f"find {len(sum_files)} number of summary stat files in provided sumstats list")
        print(f"Started minP....")
        p_subc_csf, f_subc_csf = create_minP(
            sum_files, pcol=self.pcol, mode=self.mode)

        save_path = os.path.join(self.minP_output_dir,
                                 f"{self.output_save_prefix}_minP.pkl")

        with open(save_path, 'wb') as f:
            pkl.dump([p_subc_csf, f_subc_csf], f)

        df = pd.read_table(sum_files[0])
        df['P'] = p_subc_csf
        t = [get_dim(i) for i in f_subc_csf]
        df["most_sig_dim"] = t
        save_path = os.path.join(self.minP_output_dir,
                                 f"{self.output_save_prefix}_minP.csv")
        df.to_csv(save_path, sep='\t', index=False)

        save_path_f = os.path.join(
            self.minP_output_dir, f"{self.output_save_prefix}_minP_withfiles.csv")
        df['file_loc'] = f_subc_csf
        df.to_csv(save_path_f, sep='\t', index=False)

        print(f"saved summarry_state file: {save_path}")


def do_fastgwa_random_weight_model():
    print('Doing fastgwa-> minP for untrained Model')
    r = RunFastGWA(
        cluster_embedding_path=pixel_embedding_records.subject_dict_discovery_local_moA_rand,
        cohort_type='discovery',

        project_name=pixel_embedding_records.project_dir,
        prefix=pixel_embedding_records.prefix_local_moA_rand
    )

    print(f"FastGWA started................................")

    r.prepare_samples()
    r.run_fastgwa()

    print("Done running FastGWA")

    print(f"MinP started................................")

    r.do_minP()


def do_fastgwa_Base_VoxelEmbedding():
    print('Doing fastgwa-> minP Base_VoxelEmbedding')
    r = RunFastGWA(
        cluster_embedding_path=pixel_embedding_records.subject_dict_discovery_local_bm,
        cohort_type='discovery',

        project_name=pixel_embedding_records.project_dir,
        prefix=pixel_embedding_records.prefix_local_bm
    )

    print(f"FastGWA started................................")

    r.prepare_samples()
    r.run_fastgwa()

    print("Done running FastGWA")

    print(f"MinP started................................")

    r.do_minP()


def do_fastgwa_moA():
    print('Doing fastgwa-> minP ,moA combined')

    r = RunFastGWA(
        cluster_embedding_path=pixel_embedding_records.subject_dict_discovery_combined_moA,
        cohort_type='discovery',

        project_name=pixel_embedding_records.project_dir,
        prefix=pixel_embedding_records.prefix_combined_moA
    )

    print(f"FastGWA started................................")

    r.prepare_samples()
    r.run_fastgwa()

    print("Done running FastGWA")

    print(f"MinP started................................")

    r.do_minP()

def do_fastgwa_vcic():
    print('Doing fastgwa-> minP for untrained Model')
    r = RunFastGWA(
        cluster_embedding_path=pixel_embedding_records.subject_dict_replication_local_vcic,
        cohort_type='replication',

        project_name=pixel_embedding_records.project_dir,
        prefix=pixel_embedding_records.prefix_local_vcic
    )

    print(f"FastGWA started................................")

    r.prepare_samples()
    r.run_fastgwa()

    print("Done running FastGWA")

    print(f"MinP started................................")

    r.do_minP()
def main():

    parser = argparse.ArgumentParser(
        description="Generate MRI cluster label images based on clustering results.")
    parser.add_argument("--cluster_embedding_path", type=str,
                        help="Path to pkl file containign dictionary of each cluster ID.")
    parser.add_argument("--cohort_type", type=str, default='discovery')

    parser.add_argument("--project_name", type=str, default='CLUSTER_GWAS',
                        help="This name is required to save output of fastgwa in cfg.gwas.fastgwa_out")
    parser.add_argument('--prefix', type=str, default=None,
                        help='this is to specify identifier in case of running mulitple gwas with different weights or any other cofig')

    parser.add_argument("--do_fasgwa", action='store_true')
    parser.add_argument("--do_minP", action='store_true')

    parser.add_argument('--minP_output_dir', type=str,
                        help='The directory where the output of minP will be stored')

    parser.add_argument('--output_save_prefix', type=str,
                        help='usually it is to identify the minP summary state file')

    parser.add_argument('--pcol', type=int, default=0,
                        help='the position of P column from the end ')
    parser.add_argument('--mode', type=str, default='min',
                        help='min if the P column is raw pvaleu, max if it is logP')

    args = parser.parse_args()
    print(args)
    r = RunFastGWA(
        cluster_embedding_path=args.cluster_embedding_path,
        cohort_type=args.cohort_type,

        project_name=args.project_name,
        prefix=args.prefix
    )

    if args.do_fasgwa:
        print(f"FastGWA started................................")

        r.prepare_samples()
        r.run_fastgwa()

        print("Done running FastGWA")

    if args.do_minP:
        print(f"MinP started................................")

        r.do_minP()


if __name__ == '__main__':
    # main()
    do_fastgwa_vcic()
