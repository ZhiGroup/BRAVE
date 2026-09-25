"""Generate MASKS_MNI_LINEAR files for the non-WB replication cohort.

Reproduces the published conversion (extract_embed_linear_mni.py:2468)

    flirt -in <native seg> -ref T1_unbiased_brain_linear -init T1_to_MNI_linear.mat
          -interp nearestneighbour -applyxfm

using the Python applyxfm validated in validate_mask_flirt_python.py (>= 0.999972
exact voxel match) and gated at the BRE level in bre_mask_gate.py (cosine >= 0.9999994).

Writes to a FRESH directory on a surviving NFS mount rather than into the published
MASKS_MNI_LINEAR, so no shared published asset is mutated. All 6,474 subjects are
generated, not just the 5,770 that lack masks: the 704 that already have stored masks
are regenerated too and act as a built-in QC set on every run.

I/O discipline: each subject is 3 small reads (~0.5 MB) and 2 small writes (~0.5 MB),
so the whole job is roughly 9 GB in / 3 GB out. Workers are capped (default 12 of 96
cores) because the bottleneck is the shared filesystem, not CPU. Resumable: a subject
whose two outputs already exist is skipped.

Usage
-----
    python generate_nonwb_masks.py --workers 12
    python generate_nonwb_masks.py --limit 20 --workers 4     # smoke test
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np

GATE_DIR = "<EXTERNAL: reproducibility>"
NONWB = "<EXTERNAL: nonwb_replication>"
STORED_MASK_DIR = ("<EXTERNAL: PIXEL_EMBEDDING>/"
                   "RBGWAS_RELATED_DATA/MASKS_MNI_LINEAR")
DEFAULT_OUT = "<EXTERNAL: nonwb_masks_mni_linear>"
T1_ROOTS = ["<EXTERNAL: T1_extracted>",
            "<EXTERNAL: T1_extracted>"]

# stored-file suffix -> native segmentation, relative to <subject>/T1
NATIVE_OF = [
    ("linear_T1_first_all_fast_firstseg.nii.gz",
     "T1_first/T1_first_all_fast_firstseg.nii.gz"),
    ("linear_T1_brain_seg.nii.gz", "T1_fast/T1_brain_seg.nii.gz"),
]

sys.path.insert(0, GATE_DIR)
from validate_mask_flirt_python import apply_xfm  # noqa: E402

_OUT_DIR = None


def find_t1_dir(subject):
    for root in T1_ROOTS:
        d = os.path.join(root, subject, "T1")
        if os.path.isdir(d):
            return d
    return None


def _init(out_dir):
    global _OUT_DIR
    _OUT_DIR = out_dir


def process(eid):
    """Generate both masks for one EID. Returns a result dict."""
    subject = "{0}_20252_2_0".format(eid)
    res = {"eid": eid, "status": "ok", "regenerated": 0, "skipped": 0,
           "qc_exact": "", "qc_mask": ""}

    outs = [os.path.join(_OUT_DIR, "{0}_{1}".format(subject, sfx))
            for sfx, _ in NATIVE_OF]
    if all(os.path.exists(p) for p in outs):
        res["status"] = "exists"
        res["skipped"] = len(outs)
        return res

    t1 = find_t1_dir(subject)
    if t1 is None:
        res["status"] = "no_t1_dir"
        return res
    ref_path = os.path.join(t1, "T1_unbiased_brain_linear.nii.gz")
    mat_path = os.path.join(t1, "transforms", "T1_to_MNI_linear.mat")
    if not (os.path.exists(ref_path) and os.path.exists(mat_path)):
        res["status"] = "missing_ref_or_mat"
        return res

    try:
        ref_img = nib.load(ref_path)
        mat = np.loadtxt(mat_path)
        for (sfx, rel), out_path in zip(NATIVE_OF, outs):
            src_path = os.path.join(t1, rel)
            if not os.path.exists(src_path):
                res["status"] = "missing_src"
                return res
            src_img = nib.load(src_path)
            arr = apply_xfm(src_img, ref_img, mat)
            arr = arr.astype(src_img.get_data_dtype(), copy=False)

            out_img = nib.Nifti1Image(arr, ref_img.affine, ref_img.header)
            out_img.set_data_dtype(src_img.get_data_dtype())
            # write to a temp name then rename, so an interrupted job never
            # leaves a truncated file that a resume would treat as done.
            # the temp name must keep the .nii.gz extension -- nibabel infers
            # the output format from the filename, so a bare .tmp suffix fails
            tmp = out_path[:-len(".nii.gz")] + ".partial.nii.gz"
            nib.save(out_img, tmp)
            os.rename(tmp, out_path)
            res["regenerated"] += 1

            # free QC: compare against the published file when one exists
            stored = os.path.join(STORED_MASK_DIR, "{0}_{1}".format(subject, sfx))
            if os.path.exists(stored):
                ref_arr = np.asarray(nib.load(stored).dataobj)
                if ref_arr.shape == arr.shape:
                    exact = float((np.rint(ref_arr) == np.rint(arr)).sum()) / arr.size
                    res["qc_exact"] = "{0:.8f}".format(exact)
                    res["qc_mask"] = sfx
    except Exception as exc:                                  # noqa: BLE001
        res["status"] = "error:{0}".format(type(exc).__name__)
        res["error"] = str(exc)[:200]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eid-file",
                    default=os.path.join(GATE_DIR, "nonwb_have_mask_inputs.txt"),
                    help="EIDs to generate, one per line")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=12,
                    help="capped deliberately: the shared filesystem is the "
                         "bottleneck, not CPU")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=250)
    ap.add_argument("--out-csv", default=os.path.join(GATE_DIR,
                                                     "nonwb_mask_generation.csv"))
    args = ap.parse_args()

    eids = [l.strip() for l in open(args.eid_file) if l.strip().isdigit()]
    if args.limit:
        eids = eids[:args.limit]
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    print("subjects  : {0:,}".format(len(eids)))
    print("out dir   : {0}".format(args.out_dir))
    print("workers   : {0}\n".format(args.workers))
    sys.stdout.flush()

    rows, t0 = [], time.time()
    pool = Pool(processes=args.workers, initializer=_init, initargs=(args.out_dir,))
    try:
        for i, res in enumerate(pool.imap_unordered(process, eids, chunksize=8), 1):
            rows.append(res)
            if i % args.report_every == 0 or i == len(eids):
                el = time.time() - t0
                rate = i / el
                print("  {0:6d}/{1}  {2:.1f} subj/s  elapsed {3:.1f} min  "
                      "eta {4:.1f} min".format(
                          i, len(eids), rate, el / 60,
                          (len(eids) - i) / rate / 60 if rate else float("nan")))
                sys.stdout.flush()
    finally:
        pool.close()
        pool.join()

    ok = [r for r in rows if r["status"] == "ok"]
    exists = [r for r in rows if r["status"] == "exists"]
    bad = [r for r in rows if r["status"] not in ("ok", "exists")]
    qc = [float(r["qc_exact"]) for r in rows if r["qc_exact"]]

    print("\n=== mask generation ===")
    print("generated       : {0:,} subjects ({1:,} files)".format(
        len(ok), sum(r["regenerated"] for r in ok)))
    print("already present : {0:,}".format(len(exists)))
    print("failed          : {0:,}".format(len(bad)))
    for r in bad[:20]:
        print("   {0}  {1}  {2}".format(r["eid"], r["status"], r.get("error", "")))
    if qc:
        print("QC vs published : {0:,} masks compared, exact-match min {1:.8f} "
              "mean {2:.8f}".format(len(qc), min(qc), float(np.mean(qc))))
    print("elapsed         : {0:.1f} min".format((time.time() - t0) / 60))

    keys = ["eid", "status", "regenerated", "skipped", "qc_exact", "qc_mask", "error"]
    with open(args.out_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote {0}".format(args.out_csv))


if __name__ == "__main__":
    main()
