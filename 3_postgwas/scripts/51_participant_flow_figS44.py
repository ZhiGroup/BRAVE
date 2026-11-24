#!/usr/bin/env python
"""Redraw Supplementary Fig. S44 (participant flow), CONSORT layout. Every count is READ from the evidence files;
nothing is typed in. Asserts each chain adds up and that every label fits its box.
Inputs: the encoder / PGS participant-accounting TSVs and a counts JSON (pairs, kinship minimum,
discovery N from the FastGWA logs), produced from the cohort files as in 28_compute_demographics.py."""
from __future__ import print_function
import argparse, json, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ap = argparse.ArgumentParser()
ap.add_argument("--evidence-dir", required=True, help="provenance_closure folder with the two accounting TSVs")
ap.add_argument("--counts-json", required=True, help="s44_counts.json from s44_counts_check.py")
ap.add_argument("--style-dir", required=True, help="folder holding fig_style.py")
ap.add_argument("--out-stem", required=True)
ap.add_argument("--font-file", required=True, help="Arial TTF (not installed on the server)")
a = ap.parse_args()
sys.path.insert(0, a.style_dir)
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402
apply_mpl_style()
font_manager.fontManager.addfont(a.font_file)
plt.rcParams["font.sans-serif"] = ["Arial"]
used = font_manager.findfont(font_manager.FontProperties(family=["Arial"]), fallback_to_default=False)
print("font file used:", used)
assert os.path.basename(used).lower() == "arial.ttf", "Arial not resolved"

# ---------------- counts (read, never typed) ----------------
enc = pd.read_table(os.path.join(a.evidence_dir, "encoder_participant_accounting.tsv")).set_index("group").N
pgs = pd.read_table(os.path.join(a.evidence_dir, "pgs_participant_overlap.tsv"))
pgs = pd.Series(pgs.N.values, index=pgs.set_or_overlap.str.strip())
cj = json.load(open(a.counts_json))

n_enc = int(enc["development allocation"]); n_train = int(enc["gradient-training set"])
n_held = int(enc["held-out monitoring set"])
n_noimg = int(enc["training list without image folder"]) + int(enc["held-out list without image folder"])
n_img = n_train + n_held
n_alloc = int(pgs["replication allocation (cohort CSV)"]); n_rel = int(pgs["relatedness-excluded replication participants"])
n_after_rel = int(pgs["replication after exclusion"]); n_pgs = int(pgs["of which in BGEN sample file (= lead-SNP PGS set L)"])
n_analysed = int(pgs["FastGWA analysed set F = L & complete covariates"]); n_no_cov = int(pgs["L - F"])
n_no_geno = n_after_rel - n_pgs
n_pairs = int(cj["cross_pairs"]); n_disc = cj["discovery_analysed_n_values"]

assert n_train + n_held + n_noimg == n_enc, "encoder groups do not sum"
assert n_alloc - n_rel == n_after_rel, "relatedness step does not sum"
assert int(pgs["L - replication after exclusion"]) == 0, "PGS set not nested in the post-exclusion set"
assert int(pgs["F - L"]) == 0, "analysed set not nested in the PGS set"
assert n_pgs - n_no_cov == n_analysed, "covariate step does not sum"
assert cj["cross_people_excluded"] == n_rel, "pair re-derivation disagrees with the evidence TSV"
assert len(n_disc) == 1, "discovery N differs across the 18 logs"
n_disc = int(n_disc[0])
kin = ">" if cj["cross_pairs_at_0_0442"] == 0 and cj["cross_min_kinship"] > 0.0442 else "≥"
f = lambda n: "{:,}".format(n)
print("counts:", dict(enc=n_enc, img=n_img, train=n_train, held=n_held, noimg=n_noimg, alloc=n_alloc, rel=n_rel,
                      pairs=n_pairs, after_rel=n_after_rel, no_geno=n_no_geno, pgs=n_pgs, no_cov=n_no_cov,
                      analysed=n_analysed, disc=n_disc, total=n_disc + n_analysed, kinship_sign=kin))

# ---------------- drawing (coordinates in mm) ----------------
W, H = 180.0, 150.0
fig = plt.figure(figsize=(MM(W), MM(H)))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
LW = 0.7; PAD = 1.6
texts = []

def box(cx, cy, w, h, txt, excl=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                lw=LW * (0.8 if excl else 1), ec="black", fc="#EFEFEF" if excl else "white"))
    t = ax.text(cx, cy, txt, ha="center", va="center", fontsize=7, linespacing=1.25)
    texts.append((t, w, h, txt.split("\n")[0]))
    return dict(top=cy + h / 2, bot=cy - h / 2, left=cx - w / 2, right=cx + w / 2, cx=cx, cy=cy)

def line(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color="black", lw=LW, solid_capstyle="butt")

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=LW, color="black", mutation_scale=7, shrinkA=0, shrinkB=0))

top = box(90, 143, 120, 8, "UK Biobank participants with brain T1-weighted MRI")

# encoder branch (left)
E1 = box(24, 122, 46, 18, "BRE encoder development cohort\nN = %s\nmixed ancestry; no overlap\nwith the GWAS cohorts" % f(n_enc))
ex0 = box(63, 103, 31, 9, "Excluded: n = %s\nno preprocessed image" % f(n_noimg), excl=True)
E2 = box(24, 88, 46, 10, "Preprocessed images\nN = %s" % f(n_img))
T = box(18.5, 62, 35, 16, "Training set\nN = %s\nimages used to update\nthe encoder weights" % f(n_train))
Hb = box(56.5, 62, 37, 16, "Held-out monitoring set\nN = %s\nloss recorded each epoch;\nnot used for training" % f(n_held))

# GWAS branch (right)
G = box(128, 124, 102, 10, "Genetically defined White British GWAS cohort\npartitioned into discovery and replication sets")
CXR, CXS, CXD = 126.5, 163.5, 91.0
R1 = box(CXR, 106, 40, 10, "Replication allocation\nN = %s" % f(n_alloc))
ex1 = box(CXS, 93, 31, 15, "Excluded: n = %s\nrelatives of discovery\nparticipants (%s pairs;\nkinship %s 0.0442)" % (f(n_rel), f(n_pairs), kin), excl=True)
R2 = box(CXR, 80, 40, 10, "After relatedness exclusion\nN = %s" % f(n_after_rel))
ex2 = box(CXS, 67.5, 31, 12, "Excluded: n = %s\nabsent from the analysis\ngenotype file" % f(n_no_geno), excl=True)
R3 = box(CXR, 54, 40, 14, "Genotyped, with BRE phenotypes\nN = %s\n(polygenic-score evaluation set)" % f(n_pgs))
ex3 = box(CXS, 40.5, 31, 12, "Excluded: n = %s\nno record in the final\ncovariate files" % f(n_no_cov), excl=True)
R4 = box(CXR, 27, 40, 12, "Replication\nassociation analyses\nN = %s" % f(n_analysed))
D = box(CXD, 27, 28, 12, "Discovery\nassociation analyses\nN = %s" % f(n_disc))

# connectors
yb = 135.5
line(top["cx"], top["bot"], top["cx"], yb); line(E1["cx"], yb, G["cx"], yb)
arrow(E1["cx"], yb, E1["cx"], E1["top"]); arrow(G["cx"], yb, G["cx"], G["top"])
arrow(E1["cx"], E1["bot"], E2["cx"], E2["top"]); arrow(E1["cx"], ex0["cy"], ex0["left"], ex0["cy"])
ys = 76.5
line(E2["cx"], E2["bot"], E2["cx"], ys); line(T["cx"], ys, Hb["cx"], ys)
arrow(T["cx"], ys, T["cx"], T["top"]); arrow(Hb["cx"], ys, Hb["cx"], Hb["top"])
yg = 115.5
line(G["cx"], G["bot"], G["cx"], yg); line(min(CXD, CXR, G["cx"]), yg, max(CXD, CXR, G["cx"]), yg)
arrow(CXD, yg, CXD, D["top"]); arrow(CXR, yg, CXR, R1["top"])
for upper, lower, ex in ((R1, R2, ex1), (R2, R3, ex2), (R3, R4, ex3)):
    arrow(CXR, upper["bot"], CXR, lower["top"]); arrow(CXR, ex["cy"], ex["left"], ex["cy"])

ax.text(90, 9, "Analysed GWAS participants: %s discovery + %s replication = %s.\n"
               "%s is a count of discovery–replication relative pairs, not of participants."
        % (f(n_disc), f(n_analysed), f(n_disc + n_analysed), f(n_pairs)), ha="center", va="center", fontsize=7,
        linespacing=1.3)

# ---------------- fit check: every label inside its box ----------------
fig.canvas.draw(); r = fig.canvas.get_renderer(); px2mm = 25.4 / fig.dpi; bad = []
for t, w, h, first in texts:
    bb = t.get_window_extent(r)
    tw, th = bb.width * px2mm, bb.height * px2mm
    if tw > w - PAD or th > h - 0.8:
        bad.append("%s: text %.1fx%.1f mm in box %.0fx%.0f mm" % (first, tw, th, w, h))
print("label fit:", "ALL FIT" if not bad else "OVERFLOW -> " + " | ".join(bad))
assert not bad, "a label overflows its box"
save_mpl(fig, a.out_stem)
