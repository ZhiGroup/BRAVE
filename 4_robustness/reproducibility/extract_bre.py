"""Extract Brain Region Embeddings for a cohort, into one consolidated array.

The published pipeline wrote one tiny .pt per subject per region -- about
414,000 files of ~1.3 kB each per seed, which is the worst possible access
pattern for a shared filesystem. The numbers are identical either way, so this
writes a single (n_subjects, n_regions, 128) float32 array plus index files.

The forward path is the published one, verified to reproduce the stored BREs
bit-for-bit (see verify_extraction_checkpoint.py): FPN -> trilinear upsample to
the padded volume -> crop the padding -> mean over the mask voxels of each
region. Only the fine-scale ("local") branch is used, matching the
<EXTERNAL: encoder checkpoint> outputs; the coarse branch is not upsampled here since it
does not enter the local embedding.

Usage
-----
    # validate the pipeline against stored embeddings (published checkpoint)
    python extract_bre.py --ckpt <published.ckpt> --cohort replication \\
        --limit 5 --validate

    # real run
    python extract_bre.py --ckpt <seed ckpt> --cohort discovery \\
        --out <dir> --gpu cuda:4
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
DATA = ("<EXTERNAL: PIXEL_EMBEDDING>")

# padding removed after upsampling, as in the published extraction
DS, WS, HS = 182 + 10, 218 + 6, 182 + 10
X_LB, X_RB, Y_LB, Y_RB, Z_LB, Z_RB = 5, 5, 3, 3, 5, 5

# excluded in the published run: first-stage segmentation failed for these
EXCLUDED_SUBJECTS = ("<SUBJECT_ID>", "<SUBJECT_ID>")

STORED_REPLICATION = (DATA + "/RBGWAS_RELATED_DATA/<EXTERNAL: encoder checkpoint>"
                      "revised_new/LINEAR_MNI_replication_local")


def load_inputs(cohort):
    import config
    sample_dict_path = os.path.join(config.data_path(), "<EXTERNAL: encoder checkpoint>",
                                    "{0}_gwas_sample_dict.pkl".format(cohort))
    with open(sample_dict_path, "rb") as fh:
        fsl = pickle.load(fh)
    # values are full paths; the subject id is the folder three levels up
    fsl = {k: v.split("/")[-3] for k, v in fsl.items()}
    for bad in EXCLUDED_SUBJECTS:
        for k in [k for k, v in fsl.items() if v == bad]:
            del fsl[k]

    region_dict_path = os.path.join(config.data_path(), "RBGWAS_RELATED_DATA",
                                    "region_dict_lin_min.pkl")
    with open(region_dict_path, "rb") as fh:
        region_dict_dicts = pickle.load(fh)

    img_dir_dict = {
        "t1_unbiased_root_dit": "<EXTERNAL: T1_extracted>/",
        "converted_mask_root_dir": os.path.join(
            config.data_path(), "RBGWAS_RELATED_DATA", "MASKS_MNI_LINEAR"),
    }
    return fsl, region_dict_dicts, img_dir_dict


def build_region_index(region_dict_dicts):
    """Flat, stable ordering of (mask_dict_id, region_id, label)."""
    out = []
    for dict_id in sorted(region_dict_dicts):
        for region_id in sorted(region_dict_dicts[dict_id]):
            out.append((dict_id, region_id, region_dict_dicts[dict_id][region_id]))
    return out


def stored_vector(subject, region_id, label):
    path = os.path.join(STORED_REPLICATION, subject,
                        "{0}_{1}_{2}_mean.pt".format(subject, region_id, label))
    if not os.path.exists(path):
        # anchor on the label too: a bare '*_<id>_*' glob also matches the
        # subject id's own suffix (ids look like <SUBJECT_ID>)
        hits = glob.glob(os.path.join(
            STORED_REPLICATION, subject,
            "*_{0}_{1}_mean.pt".format(region_id, label)))
        if not hits:
            return None
        path = hits[0]
    v = torch.load(path, map_location="cpu")
    return np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v).ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cohort", default="discovery",
                    choices=["discovery", "replication", "remaining"])
    ap.add_argument("--out", default=None,
                    help="output directory; required unless --validate")
    ap.add_argument("--gpu", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N subjects (timing / validation)")
    ap.add_argument("--validate", action="store_true",
                    help="compare against the stored replication embeddings")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--report-every", type=int, default=25)
    args = ap.parse_args()

    # resolve before chdir, so relative --ckpt/--out stay meaningful
    args.ckpt = os.path.abspath(args.ckpt)
    if args.out:
        args.out = os.path.abspath(args.out)

    sys.path.insert(0, STACK)
    os.chdir(STACK)
    import experiment_configurations
    import extract_embed_linear_mni as ex
    import main as main_mod
    import models
    from torch.utils.data import DataLoader

    fsl, region_dict_dicts, img_dir_dict = load_inputs(args.cohort)
    regions = build_region_index(region_dict_dicts)
    print("cohort            : {0}".format(args.cohort))
    print("subjects          : {0:,}".format(len(fsl)))
    print("regions           : {0}".format(len(regions)))

    keys = sorted(fsl.keys())
    if args.limit:
        keys = keys[:args.limit]
    img_path_ls = [fsl[k] for k in keys]

    dataset = ex.patch_dataset_from_file_linear_MNI(
        imgs_paths=img_path_ls, sample_dict=fsl, img_dir_dict=img_dir_dict)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=ex.collate_fn, num_workers=args.num_workers)

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
    print("checkpoint        : {0}".format(os.path.basename(args.ckpt)))
    print("device            : {0}\n".format(device))

    n_subj = len(img_path_ls)
    bre = np.zeros((n_subj, len(regions), 128), dtype=np.float32)
    subject_ids, filled = [], 0
    mismatches, checked = [], 0
    t0 = time.time()

    with torch.no_grad():
        for data, ids, masks in loader:
            subject = fsl[int(ids[0].item())]
            local_embed, _global = fpn(data.to(device))
            local_embed = F.interpolate(local_embed, size=(DS, WS, HS),
                                        mode="trilinear", align_corners=False)
            com = local_embed[:, :, X_LB:-X_RB, Y_LB:-Y_RB, Z_LB:-Z_RB][0]

            templates = {d: masks[d].cpu().numpy() for d in region_dict_dicts}
            for r_i, (dict_id, region_id, label) in enumerate(regions):
                x, y, z = np.where(templates[dict_id] == region_id)
                if len(x) == 0:
                    continue
                vec = com[:, x, y, z].mean(dim=-1).cpu().numpy()
                bre[filled, r_i, :] = vec
                if args.validate:
                    ref = stored_vector(subject, region_id, label)
                    if ref is not None and ref.shape == vec.shape:
                        checked += 1
                        d = float(np.max(np.abs(vec - ref)))
                        if d > 1e-5:
                            mismatches.append((subject, label, d))

            subject_ids.append(subject)
            filled += 1
            if filled % args.report_every == 0 or filled == n_subj:
                el = time.time() - t0
                rate = filled / el
                print("  {0:6d}/{1}  {2:.2f} subj/s  elapsed {3:.1f} min  "
                      "eta {4:.1f} min".format(
                          filled, n_subj, rate, el / 60,
                          (n_subj - filled) / rate / 60 if rate else float("nan")))
            del local_embed, com
            if device.type == "cuda":
                torch.cuda.empty_cache()

    el = time.time() - t0
    print("\nextracted {0} subjects in {1:.1f} min ({2:.2f} subj/s)".format(
        filled, el / 60, filled / el))

    if args.validate:
        print("\nvalidation against stored embeddings:")
        print("  region-vectors compared : {0}".format(checked))
        print("  mismatches (>1e-5)      : {0}".format(len(mismatches)))
        for s, lab, d in mismatches[:5]:
            print("    {0} {1}: max|diff|={2:.3e}".format(s, lab, d))
        print("  -> {0}".format("EXACT MATCH" if checked and not mismatches
                                else "DIFFERS"))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.save(os.path.join(args.out, "bre_{0}.npy".format(args.cohort)),
                bre[:filled])
        with open(os.path.join(args.out, "subject_ids.txt"), "w") as fh:
            fh.write("\n".join(subject_ids) + "\n")
        with open(os.path.join(args.out, "region_order.txt"), "w") as fh:
            for dict_id, region_id, label in regions:
                fh.write("{0}\t{1}\t{2}\n".format(dict_id, region_id, label))
        print("\nwrote {0} -> bre_{1}.npy {2}".format(
            args.out, args.cohort, bre[:filled].shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
