"""Write per-region, per-dimension QT phenotype files from a BRE array.

The published phenotypes are the raw embedding values -- verified: Pearson
r = 1.000000 and max|difference| = 8.9e-16 against the subject dicts, so no
rank or normal transform is applied here either.

Format matches the published files exactly: tab-separated `fiid iid QT<dim>`
with a header row, one file per region per dimension.

Naming keeps the published template so the downstream correlation script and
JAGWAS launcher recognise the files; only the directory differs per seed:

    <cohort>_<EXTERNAL: encoder checkpoint><region>_QT<dim>

Regions written are the 16 used for GWAS (GM and WM are excluded).
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import os
import sys

import numpy as np

EXCLUDE_LABELS = ("GM", "WM")
TEMPLATE = "{cohort}_<EXTERNAL: encoder checkpoint>{region}_QT{dim}"


def load_seed(path):
    bre = np.load(os.path.join(path, "bre_discovery.npy"))
    ids = [l.strip() for l in open(os.path.join(path, "subject_ids.txt"))
           if l.strip()]
    regions = [l.rstrip("\n").split("\t")
               for l in open(os.path.join(path, "region_order.txt"))
               if l.strip()]
    return bre, ids, regions


def region_filename_token(label):
    """Match the published filenames, which strip spaces from the label."""
    return label.replace(" ", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", required=True,
                    help="directory holding bre_discovery.npy + index files")
    ap.add_argument("--out", required=True, help="phenotype output directory")
    ap.add_argument("--cohort", default="discovery")
    ap.add_argument("--regions", nargs="*", default=None,
                    help="restrict to these region labels")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bre, ids, regions = load_seed(args.embeddings)
    eids = [s.split("_")[0] for s in ids]
    print("subjects : {0:,}".format(len(eids)))

    wanted = []
    for r_i, (dict_id, region_id, label) in enumerate(regions):
        if label in EXCLUDE_LABELS:
            continue
        # Callers may name a region either by its raw label or by the
        # space-stripped token used in filenames. One region label contains a
        # stray space ("Brain_Stem _or_4th_Ventricle"), so accept both.
        if args.regions and not (label in args.regions
                                 or region_filename_token(label)
                                 in args.regions):
            continue
        wanted.append((r_i, label))
    print("regions  : {0}".format(len(wanted)))
    n_dim = bre.shape[2]
    print("dims     : {0}".format(n_dim))
    print("files    : {0:,}".format(len(wanted) * n_dim))

    if args.dry_run:
        for r_i, label in wanted[:3]:
            name = TEMPLATE.format(cohort=args.cohort,
                                   region=region_filename_token(label), dim=0)
            print("  example: {0}".format(name))
        return 0

    os.makedirs(args.out, exist_ok=True)
    written = 0
    for r_i, label in wanted:
        token = region_filename_token(label)
        block = bre[:, r_i, :]
        n_bad = int((~np.isfinite(block)).any(axis=1).sum())
        if n_bad:
            print("  {0}: {1} subjects with non-finite values "
                  "(written as NA)".format(label, n_bad))
        for d in range(n_dim):
            name = TEMPLATE.format(cohort=args.cohort, region=token, dim=d)
            path = os.path.join(args.out, name)
            col = block[:, d]
            with open(path, "w") as fh:
                fh.write("fiid\tiid\tQT{0}\n".format(d))
                for e, v in zip(eids, col):
                    fh.write("{0}\t{1}\t{2}\n".format(
                        e, e, "NA" if not np.isfinite(v) else repr(float(v))))
            written += 1
        print("  {0}: {1} files".format(label, n_dim))

    print("\nwrote {0:,} phenotype files -> {1}".format(written, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
