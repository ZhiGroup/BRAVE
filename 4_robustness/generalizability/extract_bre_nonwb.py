"""Extract Brain Region Embeddings for the non-WB replication cohort (step 2).

Uses the published forward path -- the same one extract_bre.py reproduces bit-for-bit
against the stored embeddings -- with two substitutions:

  * masks come from the regenerated directory (validated in NONWB_REPLICATION_PLAN.md
    section 4: BRE cosine >= 0.9999994 against the stored masks)
  * T1s come from <EXTERNAL: T1_extracted>, which is where the non-WB subjects live

ALIGNMENT is the hazard this script exists to prevent. covarQ_t1only.npy, the per-group
MAF arrays and the genotype columns are all aligned to sample_order_t1only.txt (6,474
rows). Our analysis cohort is the 6,470 that survive the re-prune against aim_3's
cohorts. Subjects are therefore emitted in sample_order_t1only.txt order, restricted to
the keep list, and the 0-based row indices into that original order are written out
alongside. Subset the covariates and genotypes with those indices; never re-derive an
order.

The --validate-stored option first pushes a few discovery/replication subjects through
THIS script's exact code path, using the published masks, and compares against the
stored per-region .pt embeddings. That proves the path end to end before 6,470 subjects
are committed to it.

Usage
-----
    python extract_bre_nonwb.py --ckpt <published.ckpt> --validate-stored 3 --gpu cuda:1
    python extract_bre_nonwb.py --ckpt <published.ckpt> --limit 20      # smoke test
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.


# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PIX = "<EXTERNAL: PIXEL_EMBEDDING>"
STACK = os.path.join(PIX, "aim_3", "reproducibility", "train_stack")
GATE_DIR = os.path.join(PIX, "aim_3", "reproducibility")
NONWB = os.path.join(PIX, "aim_1", "nonwb_replication")

NONWB_T1_ROOT = "<EXTERNAL: T1_extracted>/"
PUBLISHED_T1_ROOT = "<EXTERNAL: T1_extracted>/"
NONWB_MASKS = "<EXTERNAL: nonwb_masks_mni_linear>"
STORED_REPLICATION = ("<EXTERNAL: PIXEL_EMBEDDING>"
                      "/RBGWAS_RELATED_DATA/<EXTERNAL: encoder checkpoint>"
                      "LINEAR_MNI_replication_local")

DS, WS, HS = 182 + 10, 218 + 6, 182 + 10
X_LB, X_RB, Y_LB, Y_RB, Z_LB, Z_RB = 5, 5, 3, 3, 5, 5
EXCLUDED_SUBJECTS = ("<SUBJECT_ID>", "<SUBJECT_ID>")


def read_ids(path):
    """EIDs from a one-per-line file, tolerating a header."""
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip().isdigit()]


def build_regions(region_dict_dicts):
    out = []
    for dict_id in sorted(region_dict_dicts):
        for region_id in sorted(region_dict_dicts[dict_id]):
            out.append((dict_id, region_id, region_dict_dicts[dict_id][region_id]))
    return out


def stored_vector(subject, region_id, label):
    path = os.path.join(STORED_REPLICATION, subject,
                        "{0}_{1}_{2}_mean.pt".format(subject, region_id, label))
    if not os.path.exists(path):
        hits = glob.glob(os.path.join(STORED_REPLICATION, subject,
                                      "*_{0}_{1}_mean.pt".format(region_id, label)))
        if not hits:
            return None
        path = hits[0]
    v = torch.load(path, map_location="cpu")
    return np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v).ravel()


def run(fpn, device, ex, subjects, img_dir_dict, region_dict_dicts, regions,
        num_workers, report_every, label):
    """Forward every subject once; return (n, n_regions, 128) float32."""
    from torch.utils.data import DataLoader
    sample_dict = {i: s for i, s in enumerate(subjects)}
    dataset = ex.patch_dataset_from_file_linear_MNI(
        imgs_paths=subjects, sample_dict=sample_dict, img_dir_dict=img_dir_dict)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=ex.collate_fn, num_workers=num_workers)

    bre = np.zeros((len(subjects), len(regions), 128), dtype=np.float32)
    empty = []
    filled, t0 = 0, time.time()
    with torch.no_grad():
        for data, ids, masks in loader:
            local_embed, _g = fpn(data.to(device))
            local_embed = F.interpolate(local_embed, size=(DS, WS, HS),
                                        mode="trilinear", align_corners=False)
            com = local_embed[:, :, X_LB:-X_RB, Y_LB:-Y_RB, Z_LB:-Z_RB][0]
            templates = {d: masks[d].cpu().numpy() for d in region_dict_dicts}
            row = int(ids[0].item())
            for r_i, (dict_id, region_id, lab) in enumerate(regions):
                x, y, z = np.where(templates[dict_id] == region_id)
                if len(x) == 0:
                    empty.append((sample_dict[row], lab))
                    continue
                bre[row, r_i, :] = com[:, x, y, z].mean(dim=-1).cpu().numpy()
            filled += 1
            if filled % report_every == 0 or filled == len(subjects):
                el = time.time() - t0
                rate = filled / el
                print("  [{0}] {1:6d}/{2}  {3:.2f} subj/s  elapsed {4:.1f} min  "
                      "eta {5:.1f} min".format(label, filled, len(subjects), rate,
                                               el / 60,
                                               (len(subjects) - filled) / rate / 60
                                               if rate else float("nan")))
                sys.stdout.flush()
            del local_embed, com
    return bre, empty


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eid-file",
                    default=os.path.join(GATE_DIR, "nonwb_eids_reprune_aim3.txt"))
    ap.add_argument("--sample-order",
                    default=os.path.join(NONWB, "sample_order_t1only.txt"))
    ap.add_argument("--mask-dir", default=NONWB_MASKS)
    ap.add_argument("--out-dir", default=os.path.join(GATE_DIR, "bre_nonwb"))
    ap.add_argument("--gpu", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--validate-stored", type=int, default=0,
                    help="first push N replication subjects through this same path "
                         "with the published masks and compare to stored embeddings")
    args = ap.parse_args()
    args.ckpt = os.path.abspath(args.ckpt)
    args.out_dir = os.path.abspath(args.out_dir)

    sys.path.insert(0, STACK)
    os.chdir(STACK)
    import config
    import experiment_configurations
    import extract_embed_linear_mni as ex
    import main as main_mod
    import models

    with open(os.path.join(config.data_path(), "RBGWAS_RELATED_DATA",
                           "region_dict_lin_min.pkl"), "rb") as fh:
        region_dict_dicts = pickle.load(fh)
    regions = build_regions(region_dict_dicts)

    conf = experiment_configurations.conf_model_4_exp_3()
    conf.n_batch = 4
    conf.max_epoch = 40
    margs = main_mod.get_args(config_file_lambda="", device=torch.device("cpu"),
                              use_config_obj=True, config_obj_name=conf)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    fpn = models.get_pretrained_model_voxel_classification_model(
        args.ckpt, full_wt_path_given=True, fully_random_weight=False,
        model_type="vanila", model_kwargs=margs)
    fpn.to(device).eval()
    print("checkpoint : {0}".format(os.path.basename(args.ckpt)))
    print("device     : {0}".format(device))
    print("regions    : {0}\n".format(len(regions)))

    # ---- gate: reproduce stored embeddings through this same code path ----
    if args.validate_stored:
        sd_path = os.path.join(config.data_path(), "<EXTERNAL: encoder checkpoint>",
                               "replication_gwas_sample_dict.pkl")
        with open(sd_path, "rb") as fh:
            fsl = pickle.load(fh)
        fsl = {k: v.split("/")[-3] for k, v in fsl.items()}
        for bad in EXCLUDED_SUBJECTS:
            for k in [k for k, v in fsl.items() if v == bad]:
                del fsl[k]
        subs = [fsl[k] for k in sorted(fsl.keys())[:args.validate_stored]]
        pub_dirs = {"t1_unbiased_root_dit": PUBLISHED_T1_ROOT,
                    "converted_mask_root_dir": os.path.join(
                        config.data_path(), "RBGWAS_RELATED_DATA", "MASKS_MNI_LINEAR")}
        vb, _ = run(fpn, device, ex, subs, pub_dirs, region_dict_dicts, regions,
                    args.num_workers, 1, "validate")
        worst, checked = 0.0, 0
        for i, s in enumerate(subs):
            for r_i, (_d, region_id, lab) in enumerate(regions):
                ref = stored_vector(s, region_id, lab)
                if ref is not None and ref.shape == (128,):
                    checked += 1
                    worst = max(worst, float(np.max(np.abs(vb[i, r_i] - ref))))
        print("\nstored-embedding gate: {0} vectors, max abs diff {1:.3e}".format(
            checked, worst))
        if checked == 0:
            sys.exit("GATE INCONCLUSIVE: no stored embeddings found")
        if worst > 1e-5:
            sys.exit("GATE FAIL: max abs diff {0:.3e} exceeds 1e-5".format(worst))
        print("GATE PASS\n")

    # ---- the cohort, in sample_order_t1only.txt order ----
    keep = set(read_ids(args.eid_file))
    order = read_ids(args.sample_order)
    rows = [(i, e) for i, e in enumerate(order) if e in keep]
    missing = keep - set(order)
    if missing:
        sys.exit("{0} kept EIDs are absent from {1}".format(
            len(missing), args.sample_order))
    if args.limit:
        rows = rows[:args.limit]
    sample_index = np.array([i for i, _e in rows], dtype=np.int64)
    eids = [e for _i, e in rows]
    subjects = ["{0}_20252_2_0".format(e) for e in eids]
    print("sample_order rows : {0:,}".format(len(order)))
    print("analysis cohort   : {0:,}".format(len(subjects)))

    for s in subjects[:0] + subjects:
        p = os.path.join(args.mask_dir,
                         "{0}_linear_T1_brain_seg.nii.gz".format(s))
        if not os.path.exists(p):
            sys.exit("missing mask for {0}".format(s))
    print("masks present     : all\n")

    img_dir_dict = {"t1_unbiased_root_dit": NONWB_T1_ROOT,
                    "converted_mask_root_dir": args.mask_dir}
    bre, empty = run(fpn, device, ex, subjects, img_dir_dict, region_dict_dicts,
                     regions, args.num_workers, args.report_every, "nonwb")

    # ---- runtime guards ----
    n_bad = int(np.isnan(bre).sum() + np.isinf(bre).sum())
    zero_rows = int((np.abs(bre).sum(axis=2) == 0).sum())
    print("\nnon-finite values : {0}".format(n_bad))
    print("empty region means: {0} (region, subject) pairs".format(zero_rows))
    if empty:
        for s, lab in empty[:10]:
            print("   empty: {0} {1}".format(s, lab))
    if n_bad:
        sys.exit("non-finite values in the embedding array")

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    np.save(os.path.join(args.out_dir, "bre_nonwb.npy"), bre)
    np.save(os.path.join(args.out_dir, "sample_index.npy"), sample_index)
    with open(os.path.join(args.out_dir, "subjects.txt"), "w") as fh:
        for e in eids:
            fh.write("{0}\n".format(e))
    with open(os.path.join(args.out_dir, "regions.txt"), "w") as fh:
        for dict_id, region_id, lab in regions:
            fh.write("{0}\t{1}\t{2}\n".format(dict_id, region_id, lab))
    print("\nwrote {0}".format(args.out_dir))
    print("  bre_nonwb.npy   {0}".format(bre.shape))
    print("  sample_index.npy -> rows of {0}".format(
        os.path.basename(args.sample_order)))


if __name__ == "__main__":
    main()
