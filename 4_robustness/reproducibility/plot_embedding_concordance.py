"""Figures for the embedding-level reproducibility analysis.

Four panels, each answering one question:

  a  Do the three arms agree equally? (the central result)
  b  Which regions reproduce, and which do not?
  c  Where in the spectrum does agreement break down?
  d  Why does the choice of metric matter?

Panel (d) exists because CCA and CKA disagree sharply about the regions: CCA is
invariant to any invertible linear map, so it can report agreement by rescaling
low-variance directions, and it compresses the between-region range that CKA
resolves.

Recomputes from the embedding arrays rather than reading the summary CSV,
because the canonical-correlation spectrum in panel (c) is not in that file.
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

import numpy as np

sys.path.insert(0, "<EXTERNAL: aim_3>/"
                   "jagwas_paper/post_gwas_analysis/scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from fig_style import apply_mpl_style, save_mpl, MM, FS_LABEL, FS_TITLE  # noqa

PUB_DICT = ("<EXTERNAL: PIXEL_EMBEDDING>/"
            "RBGWAS_RELATED_DATA/SUBJECT_DICTS/<EXTERNAL: encoder checkpoint>"
            "discovery_local")
EXCLUDE = ("GM", "WM")

PRETTY = {
    "Left_Thalamus_Proper": "L. Thalamus", "Right_Thalamus-Proper": "R. Thalamus",
    "Left_Caudate": "L. Caudate", "Right_Caudate": "R. Caudate",
    "Left_Putamen": "L. Putamen", "Right_Putamen": "R. Putamen",
    "Left_Pallidum": "L. Pallidum", "Right_Pallidum": "R. Pallidum",
    "Left_Hippocampus": "L. Hippocampus", "Right_Hippocampus": "R. Hippocampus",
    "Left_Amygdala": "L. Amygdala", "Right_Amygdala": "R. Amygdala",
    "Left_Accumbens-area": "L. Accumbens",
    "Right_Accumbens-area": "R. Accumbens",
    "Brain_Stem _or_4th_Ventricle": "Brainstem", "CSF": "CSF",
}

# one hue per comparison; seed-vs-seed is the reference, so it reads darkest
C_S1S2 = "#1f3b5c"
C_S1P  = "#4e8ab5"
C_S2P  = "#9dc6de"
C_HL   = "#b03a2e"      # highlight for the thalamus outliers


def clean(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[ok], Y[ok]


def cca_spectrum(X, Y):
    X, Y = clean(X, Y)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    try:
        s = np.linalg.svd(qx.T.dot(qy), compute_uv=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as sp_svd
        s = sp_svd(qx.T.dot(qy), compute_uv=False, lapack_driver="gesvd")
    return np.clip(s, 0.0, 1.0)


def linear_cka(X, Y):
    X, Y = clean(X, Y)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = float((X.T.dot(Y) ** 2).sum())
    den = (np.linalg.norm(X.T.dot(X), "fro")
           * np.linalg.norm(Y.T.dot(Y), "fro"))
    return num / den if den else float("nan")


def load_seed(path):
    bre = np.load(os.path.join(path, "bre_discovery.npy"))
    ids = [l.strip() for l in open(os.path.join(path, "subject_ids.txt"))
           if l.strip()]
    regions = [l.rstrip("\n").split("\t")
               for l in open(os.path.join(path, "region_order.txt"))
               if l.strip()]
    return bre, ids, regions


def load_published(region_id, label):
    hits = [h for h in glob.glob(os.path.join(PUB_DICT,
                                              "{0}_*_mean.pkl".format(region_id)))
            if label.replace(" ", "") in os.path.basename(h).replace(" ", "")]
    if not hits:
        hits = glob.glob(os.path.join(PUB_DICT,
                                      "{0}_*_mean.pkl".format(region_id)))
    if not hits:
        return None
    with open(hits[0], "rb") as fh:
        raw = pickle.load(fh)
    return {str(k).split("_")[0]:
            np.asarray(v[1] if isinstance(v, tuple) else v).ravel()
            for k, v in raw.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed1", required=True)
    ap.add_argument("--seed2", required=True)
    ap.add_argument("--out", required=True, help="output path stem")
    args = ap.parse_args()

    b1, ids1, regions = load_seed(args.seed1)
    b2, _, _ = load_seed(args.seed2)
    eids = [s.split("_")[0] for s in ids1]

    rows, spectra = [], {"s1s2": [], "s1p": [], "s2p": []}
    for r_i, (dict_id, region_id, label) in enumerate(regions):
        if label in EXCLUDE:
            continue
        A, B = b1[:, r_i, :], b2[:, r_i, :]
        rec = {"region": PRETTY.get(label, label),
               "cka_s1s2": linear_cka(A, B)}
        spectra["s1s2"].append(cca_spectrum(A, B))

        pub = load_published(region_id, label)
        if pub:
            keep = [i for i, e in enumerate(eids) if e in pub]
            P = np.vstack([pub[eids[i]] for i in keep])
            rec["cka_s1p"] = linear_cka(b1[keep][:, r_i, :], P)
            rec["cka_s2p"] = linear_cka(b2[keep][:, r_i, :], P)
            spectra["s1p"].append(cca_spectrum(b1[keep][:, r_i, :], P))
            spectra["s2p"].append(cca_spectrum(b2[keep][:, r_i, :], P))
        rec["cca_s1s2"] = float(spectra["s1s2"][-1].mean())
        rows.append(rec)
        print("  {0:16s} CKA={1:.3f}  CCA={2:.3f}".format(
            rec["region"], rec["cka_s1s2"], rec["cca_s1s2"]))
        sys.stdout.flush()

    apply_mpl_style()
    fig = plt.figure(figsize=(MM(180), MM(125)))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.38,
                          left=0.09, right=0.985, top=0.93, bottom=0.09)

    # ── a. the three arms agree equally ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    means = {
        "CKA": [np.mean([r["cka_s1s2"] for r in rows]),
                np.mean([r.get("cka_s1p", np.nan) for r in rows]),
                np.mean([r.get("cka_s2p", np.nan) for r in rows])],
        "CCA\n(all 128)": [np.mean([s.mean() for s in spectra["s1s2"]]),
                           np.mean([s.mean() for s in spectra["s1p"]]),
                           np.mean([s.mean() for s in spectra["s2p"]])],
        "CCA\n(top 10)": [np.mean([s[:10].mean() for s in spectra["s1s2"]]),
                          np.mean([s[:10].mean() for s in spectra["s1p"]]),
                          np.mean([s[:10].mean() for s in spectra["s2p"]])],
    }
    x = np.arange(len(means))
    w = 0.26
    for k, (lab, col) in enumerate([("seed 1 vs seed 2", C_S1S2),
                                    ("seed 1 vs published", C_S1P),
                                    ("seed 2 vs published", C_S2P)]):
        vals = [means[m][k] for m in means]
        ax.bar(x + (k - 1) * w, vals, w, label=lab, color=col,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(list(means.keys()))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("agreement", fontsize=FS_LABEL)
    ax.set_title("a   All three arms agree equally", fontsize=FS_TITLE,
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.1)

    # ── b. per-region CKA ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    order = sorted(rows, key=lambda r: r["cka_s1s2"])
    ypos = np.arange(len(order))
    cols = [C_HL if "Thalamus" in r["region"] else C_S1S2 for r in order]
    ax.barh(ypos, [r["cka_s1s2"] for r in order], color=cols, height=0.68)
    ax.plot([r.get("cka_s1p", np.nan) for r in order], ypos, "o",
            ms=2.2, color=C_S1P, label="seed 1 vs pub.")
    ax.plot([r.get("cka_s2p", np.nan) for r in order], ypos, "o",
            ms=2.2, color=C_S2P, label="seed 2 vs pub.")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["region"] for r in order], fontsize=5.6)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("CKA (bars: seed 1 vs seed 2)", fontsize=FS_LABEL)
    ax.set_title("b   Both thalami reproduce far worse",
                 fontsize=FS_TITLE, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.0)

    # ── c. canonical-correlation spectrum ────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for key, lab, col in [("s1s2", "seed 1 vs seed 2", C_S1S2),
                          ("s1p", "seed 1 vs published", C_S1P),
                          ("s2p", "seed 2 vs published", C_S2P)]:
        M = np.vstack(spectra[key])          # (n_regions, n_components)
        mean_curve = M.mean(axis=0)
        comp = np.arange(1, len(mean_curve) + 1)
        ax.plot(comp, mean_curve, lw=1.2, color=col, label=lab)
        # envelope across regions, not a confidence interval: its lower edge is
        # essentially the thalamus curve at every component
        ax.fill_between(comp, M.min(0), M.max(0),
                        color=col, alpha=0.14, linewidth=0)
    ax.axvline(10, color="#888888", lw=0.7, ls=":")
    ax.text(13, 0.93, "leading 10", fontsize=6, color="#666666")
    ax.set_xlabel("canonical component", fontsize=FS_LABEL)
    ax.set_ylabel("canonical correlation", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.02)
    ax.set_title("c   Leading subspace reproduces; the tail does not",
                 fontsize=FS_TITLE, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6, loc="lower left", handlelength=1.1)

    # ── d. why the metric matters ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    xs = [r["cca_s1s2"] for r in rows]
    ys = [r["cka_s1s2"] for r in rows]
    is_th = ["Thalamus" in r["region"] for r in rows]
    ax.scatter([x for x, t in zip(xs, is_th) if not t],
               [y for y, t in zip(ys, is_th) if not t],
               s=16, color=C_S1S2, zorder=3, label="other regions")
    ax.scatter([x for x, t in zip(xs, is_th) if t],
               [y for y, t in zip(ys, is_th) if t],
               s=20, color=C_HL, zorder=4, label="thalamus")
    for r in rows:
        if "Thalamus" in r["region"]:
            ax.annotate(r["region"], (r["cca_s1s2"], r["cka_s1s2"]),
                        xytext=(5, -1), textcoords="offset points",
                        fontsize=5.6, color=C_HL)
    # span of each metric across regions, drawn as reference rules
    ax.annotate("", xy=(min(xs), 0.16), xytext=(max(xs), 0.16),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#888888"))
    ax.text((min(xs) + max(xs)) / 2, 0.19,
            "CCA spans {0:.2f}".format(max(xs) - min(xs)),
            ha="center", fontsize=5.6, color="#666666")
    ax.set_xlabel("mean CCA (seed 1 vs seed 2)", fontsize=FS_LABEL)
    ax.set_ylabel("CKA (seed 1 vs seed 2)", fontsize=FS_LABEL)
    ax.set_xlim(0.50, 0.65)
    ax.set_ylim(0.10, 1.0)
    ax.set_title("d   CCA hides a range CKA resolves",
                 fontsize=FS_TITLE, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1.0)

    save_mpl(fig, args.out)
    print("\nwrote {0}.pdf and {0}.png".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
