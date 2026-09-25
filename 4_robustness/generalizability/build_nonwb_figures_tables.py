"""Build the supplementary tables and figures for the non-WB replication.

Tables are deliberately column-matched to the existing WB replication pair (S2 per-locus,
S3 per-region) so the two arms can be read side by side: same 276 aggregate loci, same
per-region lead SNPs, same GW / Bonferroni / nominal thresholds, N = 12,359 same-ancestry
versus N = 6,470 heterogeneous.

Outputs
-------
    results/nonwb_replication/TableS_nonwb_per_locus.csv     (276 rows, twin of S2)
    results/nonwb_replication/TableS_nonwb_per_region.csv    (16 rows,  twin of S3)
    figures/nonwb_replication/S57_nonwb_replication.{pdf,png}
    figures/nonwb_replication/S58_nonwb_calibration.{pdf,png}

Usage
-----
    python build_nonwb_figures_tables.py
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import csv
import os
import sys

import numpy as np

REPRO = "<EXTERNAL: reproducibility>"
PGA = ("<EXTERNAL: jagwas_paper>/"
       "post_gwas_analysis")
RES = os.path.join(PGA, "results", "nonwb_replication")
FIG = os.path.join(PGA, "figures", "nonwb_replication")
sys.path.insert(0, os.path.join(PGA, "scripts"))

from fig_style import apply_mpl_style, save_mpl, MM        # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

THRESH_GW = 3.125e-9
NOMINAL = 0.05


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def norm_key(s):
    """Region key tolerant of the spacing/punctuation drift between the BRE labels
    ('Brain_Stem _or_4th_Ventricle') and the GWAS folder names."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def main():
    for d in (RES, FIG):
        if not os.path.isdir(d):
            os.makedirs(d)

    loci = read_csv(os.path.join(REPRO, "replication_results.csv"))
    leads = read_csv(os.path.join(REPRO, "replication_results_leads.csv"))
    wb = read_csv(os.path.join(PGA, "results", "replication",
                               "per_region_replication.csv"))
    lam = read_csv(os.path.join(REPRO, "genomewide_nonwb",
                                "lambda_gc_maf_filtered.csv"))
    defl = read_csv(os.path.join(REPRO, "lambda_deflation_explained.csv"))

    # ---------------- Table: per locus (twin of S2) ----------------
    per_locus = []
    for r in loci:
        pj = fnum(r["rep_p_joint"])
        per_locus.append({
            "al_id": r["al_id"], "region": r["region"], "lead_snp": r["rsid"],
            "chr": r["chr"], "pos": r["pos"], "a1": r["a1"], "a2": r["a2"],
            "maf_nonwb": r["maf_nonwb"], "n": r["n"],
            "discovery_p": r["disc_p_jagwas"],
            "rep_chi2": "{0:.4f}".format(fnum(r["rep_chi2"])),
            "rep_p_joint": "{0:.6g}".format(pj),
            "rep_nominal": pj < NOMINAL,
            "rep_bonferroni": pj < NOMINAL / len(loci),
            "rep_genomewide": pj < THRESH_GW,
            "discovery_minp_dim": r["disc_minp_dim"],
            "discovery_beta": r["disc_minp_beta"],
            "rep_beta": r["rep_beta_dim"], "rep_p_dim": r["rep_p_dim"],
            "direction_concordant": r["same_direction"],
        })
    out1 = os.path.join(RES, "TableS_nonwb_per_locus.csv")
    with open(out1, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_locus[0].keys()))
        w.writeheader()
        for r in per_locus:
            w.writerow(r)
    print("wrote {0}  ({1} rows)".format(out1, len(per_locus)))

    # ---------------- Table: per region (twin of S3) ----------------
    disp = {r["folder"]: r["display"] for r in wb}
    disp_by_key = {norm_key(r["folder"]): r["display"] for r in wb}
    wb_by = {r["folder"]: r for r in wb}
    lam_by = {norm_key(r["region"]): r for r in lam}
    defl_by = {norm_key(r["region"]): r for r in defl}

    by_region = {}
    for r in leads:
        by_region.setdefault(r["region"], []).append(r)

    per_region = []
    for folder in sorted(by_region):
        sub = by_region[folder]
        n = len(sub)
        pj = np.array([fnum(x["rep_p_joint"]) for x in sub])
        conc = sum(1 for x in sub if x["same_direction"] == "True")
        bonf_thr = NOMINAL / n
        key = norm_key(folder)
        lm = lam_by.get(key, {})
        df_ = defl_by.get(key, {})
        per_region.append({
            "display": disp.get(folder, folder), "folder": folder,
            "n_lead_snps": n, "n_found_in_rep": n,
            "p_bonf_threshold": bonf_thr,
            "n_gw": int((pj < THRESH_GW).sum()),
            "n_bonf": int((pj < bonf_thr).sum()),
            "n_nom": int((pj < NOMINAL).sum()),
            "n_fail": int((pj >= NOMINAL).sum()),
            "pct_gw": 100.0 * (pj < THRESH_GW).mean(),
            "pct_bonf": 100.0 * (pj < bonf_thr).mean(),
            "pct_nom": 100.0 * (pj < NOMINAL).mean(),
            "n_direction_concordant": conc,
            "pct_direction_concordant": 100.0 * conc / n,
            "lambda_gc": lm.get("lambda_maf", ""),
            "effective_df": df_.get("effective_df", ""),
            "n_clamped": df_.get("n_clamped", ""),
        })
    out2 = os.path.join(RES, "TableS_nonwb_per_region.csv")
    with open(out2, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_region[0].keys()))
        w.writeheader()
        for r in per_region:
            w.writerow(r)
    print("wrote {0}  ({1} rows)".format(out2, len(per_region)))

    apply_mpl_style()

    # ---------------- Figure S57: replication ----------------
    fig = plt.figure(figsize=(MM(183), MM(80)))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.78)

    # (a) discovery vs replication effect size, 276 aggregate loci
    ax = fig.add_subplot(gs[0, 0])
    bd = np.array([fnum(r["disc_minp_beta"]) for r in loci])
    br = np.array([fnum(r["rep_beta_dim"]) for r in loci])
    ok = np.isfinite(bd) & np.isfinite(br)
    same = np.array([r["same_direction"] == "True" for r in loci])
    ax.axhline(0, lw=0.4, color="0.75", zorder=0)
    ax.axvline(0, lw=0.4, color="0.75", zorder=0)
    lim = np.nanmax(np.abs(np.concatenate([bd[ok], br[ok]]))) * 1.08
    ax.plot([-lim, lim], [-lim, lim], lw=0.6, color="0.55", ls="--", zorder=1)
    ax.scatter(bd[ok & ~same], br[ok & ~same], s=5, lw=0, color="#C44E52",
               alpha=0.85, zorder=2, label="discordant")
    ax.scatter(bd[ok & same], br[ok & same], s=5, lw=0, color="#4C72B0",
               alpha=0.85, zorder=3, label="concordant")
    r_beta = float(np.corrcoef(bd[ok], br[ok])[0, 1])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Discovery effect size")
    ax.set_ylabel("Non-WB effect size")
    ax.set_title("a", loc="left", fontweight="bold")
    ax.text(0.04, 0.94, "r = {0:.3f}\n{1}/{2} concordant".format(
        r_beta, int(same.sum()), len(loci)), transform=ax.transAxes,
        va="top", ha="left")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2,
              borderaxespad=0.1, markerscale=1.4, fontsize=6,
              labelspacing=0.25)

    # (b) per-region nominal replication, WB vs non-WB
    ax = fig.add_subplot(gs[0, 1])
    order = sorted(per_region, key=lambda r: -r["pct_nom"])
    labels = [r["display"] for r in order]
    nonwb_pct = [r["pct_nom"] for r in order]
    wb_pct = [fnum(wb_by[r["folder"]]["pct_nom"]) if r["folder"] in wb_by
              else np.nan for r in order]
    y = np.arange(len(order))
    ax.barh(y - 0.2, wb_pct, height=0.4, color="#8C8C8C", label="White British (N = 12,359)")
    ax.barh(y + 0.2, nonwb_pct, height=0.4, color="#4C72B0", label="Non-WB (N = 6,470)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Lead SNPs replicating at p < 0.05 (%)")
    ax.set_xlim(0, 100)
    ax.set_title("b", loc="left", fontweight="bold", pad=14)
    # legend above the axes: the bars fill the plotting area at every row
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, fontsize=6, handlelength=1.4, handletextpad=0.4,
              columnspacing=1.1, borderaxespad=0.0)

    # (c) QQ of the joint-test p at the 276 leads
    ax = fig.add_subplot(gs[0, 2])
    p = np.sort(np.array([fnum(r["rep_p_joint"]) for r in loci]))
    p = p[np.isfinite(p)]
    n = len(p)
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    obs = -np.log10(np.maximum(p, 1e-300))
    # separate limits: expected tops out near 2.4 while observed reaches ~46,
    # so forcing a square range leaves the panel almost empty
    xmx, ymx = exp.max() * 1.08, obs.max() * 1.08
    ax.plot([0, xmx], [0, xmx], lw=0.6, color="0.55", ls="--")
    ax.scatter(exp, obs, s=5, lw=0, color="#4C72B0")
    ax.set_xlabel(r"Expected $-\log_{10}p$")
    ax.set_ylabel(r"Observed $-\log_{10}p$")
    ax.set_xlim(0, xmx)
    ax.set_ylim(0, ymx)
    ax.set_title("c", loc="left", fontweight="bold")
    n_nom = int((p < NOMINAL).sum())
    ax.text(0.04, 0.94, "{0}/{1} at p < 0.05\n({2:.0f} expected)".format(
        n_nom, n, NOMINAL * n), transform=ax.transAxes, va="top", ha="left")

    save_mpl(fig, os.path.join(FIG, "S57_nonwb_replication"))
    plt.close(fig)

    # ---------------- Figure S58: calibration ----------------
    fig = plt.figure(figsize=(MM(165), MM(78)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.72)

    ax = fig.add_subplot(gs[0, 0])
    pred = np.array([fnum(r["pred_lambda"]) for r in defl])
    obsl = np.array([fnum(r["obs_lambda"]) for r in defl])
    good = np.isfinite(pred) & np.isfinite(obsl)
    lo = min(pred[good].min(), obsl[good].min()) - 0.02
    hi = max(pred[good].max(), obsl[good].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], lw=0.6, color="0.55", ls="--")
    ax.scatter(pred[good], obsl[good], s=7, lw=0, color="#4C72B0")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Predicted $\lambda_{GC}$ from the clamp")
    ax.set_ylabel(r"Observed $\lambda_{GC}$")
    ax.set_title("a", loc="left", fontweight="bold")
    rr = float(np.corrcoef(pred[good], obsl[good])[0, 1])
    ax.text(0.04, 0.94, "r = {0:.4f}".format(rr), transform=ax.transAxes,
            va="top", ha="left")

    ax = fig.add_subplot(gs[0, 1])
    lam_order = sorted(defl, key=lambda r: fnum(r["obs_lambda"]))
    y = np.arange(len(lam_order))
    ax.barh(y, [fnum(r["obs_lambda"]) for r in lam_order], height=0.62,
            color="#4C72B0")
    ax.axvline(1.0, lw=0.6, color="0.35")
    ax.axvline(1.05, lw=0.6, color="#C44E52", ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([disp_by_key.get(norm_key(r["region"]),
                                        r["region"].replace("_", " ").strip())
                        for r in lam_order])
    ax.set_xlabel(r"$\lambda_{GC}$ (MAF $\geq$ 0.01)")
    ax.set_xlim(0, 1.15)
    ax.set_title("b", loc="left", fontweight="bold")

    save_mpl(fig, os.path.join(FIG, "S58_nonwb_calibration"))
    plt.close(fig)

    print("\nsummary for FACTS:")
    print("  per-locus  276: nominal {0}, bonf {1}, gw {2}".format(
        sum(1 for r in per_locus if r["rep_nominal"]),
        sum(1 for r in per_locus if r["rep_bonferroni"]),
        sum(1 for r in per_locus if r["rep_genomewide"])))
    print("  per-lead 1263: nominal {0}, concordant {1}".format(
        sum(r["n_nom"] for r in per_region),
        sum(r["n_direction_concordant"] for r in per_region)))
    print("  beta r (276 loci) = {0:.4f}".format(r_beta))


if __name__ == "__main__":
    main()
