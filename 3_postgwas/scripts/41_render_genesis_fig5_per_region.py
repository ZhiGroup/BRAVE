#!/usr/bin/env python3
# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.
"""
41_render_genesis_fig5_per_region.py
Per-region BRE-vs-volume Fig 5 panels.

For each of the 16 BRE regions: one subplot comparing
  - per-region BRE GENESIS curve
  - anatomically-matched ENIGMA subcortical volume curve
  - Height (reference, same across every subplot)

Three figures produced (one per Wen Fig 5 panel):
  fig5_panel_b_discoveries_per_region.{pdf,png}  — # GWS loci vs N
  fig5_panel_c_h2_captured_per_region.{pdf,png}  — % h² captured vs N
  fig5_panel_a_effect_size_per_region.{pdf,png}  — effect-size density
"""
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM

PROJECT_BASE = SCRIPT_DIR.parents[1]
GENESIS_DIR = PROJECT_BASE / "post_gwas_analysis" / "results" / "genesis"
FIG_DIR = PROJECT_BASE / "post_gwas_analysis" / "figures" / "genesis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RSCRIPT = "<EXTERNAL: Rscript>"
ZHANG_LIB = "<EXTERNAL: library_zhang_genesis>"

# 16 BRE regions × matched ENIGMA volume (using ICV proxy for CSF)
# (subplot_label, BRE_trait, ENIGMA_trait)
PAIRS = [
    ("L.Caudate",      "BRE_L.Caudate_QT52",       "ENIGMA_Caudate"),
    ("R.Caudate",      "BRE_R.Caudate_QT81",       "ENIGMA_Caudate"),
    ("L.Putamen",      "BRE_L.Putamen_QT18",       "ENIGMA_Putamen"),
    ("R.Putamen",      "BRE_R.Putamen_QT118",      "ENIGMA_Putamen"),
    ("L.Pallidum",     "BRE_L.Pallidum_QT11",      "ENIGMA_Pallidum"),
    ("R.Pallidum",     "BRE_R.Pallidum_QT20",      "ENIGMA_Pallidum"),
    ("L.Hippocampus",  "BRE_L.Hippocampus_QT27",   "ENIGMA_Hippocampus"),
    ("R.Hippocampus",  "BRE_R.Hippocampus_QT35",   "ENIGMA_Hippocampus"),
    ("L.Amygdala",     "BRE_L.Amygdala_QT68",      "ENIGMA_Amygdala"),
    ("R.Amygdala",     "BRE_R.Amygdala_QT54",      "ENIGMA_Amygdala"),
    ("L.Accumbens",    "BRE_L.Accumbens_QT100",    "ENIGMA_Accumbens"),
    ("R.Accumbens",    "BRE_R.Accumbens_QT31",     "ENIGMA_Accumbens"),
    ("L.Thalamus",     "BRE_L.Thalamus_QT83",      "ENIGMA_Thalamus"),
    ("R.Thalamus",     "BRE_R.Thalamus_QT40",      "ENIGMA_Thalamus"),
    ("Brainstem",      "BRE_BrainStem_QT127",      "ENIGMA_Brainstem"),
    ("CSF",            "BRE_CSF_QT28",             "ENIGMA_ICV"),
]
HEIGHT_TRAIT = "height_REFERENCE"

OUR_N = 22878

# Colours
COL_BRE = "#1f77b4"      # blue
COL_VOL = "#ff7f0e"      # orange
COL_HT  = "#888888"      # grey


def load_params(trait):
    rds = GENESIS_DIR / trait / "fit_2comp.rds"
    if not rds.exists():
        return None
    r_cmd = f"""
.libPaths(c('{ZHANG_LIB}', .libPaths()))
fit <- readRDS('{rds}')
est <- fit$estimates
par <- est[['Parameter (pic, sigmasq, a) estimates']]
h2_str <- est[['Total heritability in log-odds-ratio scale (sd)']]
nssnp_str <- est[['Number of sSNPs (sd)']]
cat(sprintf('%.10g\\n', par[1]))
cat(sprintf('%.10g\\n', par[2]))
cat(sprintf('%s\\n', h2_str))
cat(sprintf('%s\\n', nssnp_str))
"""
    r = subprocess.run([RSCRIPT, "-e", r_cmd], capture_output=True, text=True)
    lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
    if len(lines) < 4:
        return None
    def parse_paren(s):
        try:
            main, sd = s.split(" (")
            return float(main), float(sd.rstrip(")"))
        except Exception:
            return float(s), 0.0
    h2, _ = parse_paren(lines[2])
    nssnp, _ = parse_paren(lines[3])
    return dict(pic=float(lines[0]), sigmasq=float(lines[1]),
                h2=h2, nssnp=nssnp)


def load_projection(trait):
    pf = GENESIS_DIR / trait / "projection.csv"
    if not pf.exists():
        return None
    df = pd.read_csv(pf)
    # As established in pilot: p[1]=h²(constant), p[2]=#disc, p[3]=fraction h² captured
    return df.rename(columns={
        "n_discoveries": "h2_const",
        "GV_pct": "n_disc",
        "herit_explained": "frac_h2",
    })


def fig_per_region_panel_b(traits_cache):
    """# GWS loci vs N — 4x4 grid, one subplot per region."""
    apply_mpl_style()
    fig, axes = plt.subplots(4, 4, figsize=(MM(180), MM(180)), sharex=True)
    for idx, (label, bre, vol) in enumerate(PAIRS):
        ax = axes.flat[idx]
        for trait, color, name in [(HEIGHT_TRAIT, COL_HT, "Height"),
                                    (vol, COL_VOL, "ENIGMA vol"),
                                    (bre, COL_BRE, "BRE")]:
            p = traits_cache.get(trait, {}).get("proj")
            if p is None:
                continue
            ls = ":" if trait == HEIGHT_TRAIT else "-"
            ax.semilogx(p["N"], p["n_disc"], "o" if trait != HEIGHT_TRAIT else "",
                       linestyle=ls, color=color, label=name, ms=2.5, lw=1.1)
        ax.axvline(OUR_N, color="red", linestyle="--", lw=0.5, alpha=0.4)
        ax.set_title(label, fontsize=7.5)
        ax.tick_params(labelsize=6)
        if idx >= 12:  # bottom row
            ax.set_xlabel("N (log)", fontsize=7)
        if idx % 4 == 0:  # left column
            ax.set_ylabel("# GWS loci", fontsize=7)
        if idx == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Predicted # GWAS-significant loci vs sample size — per region",
                 fontsize=9, y=1.005)
    fig.tight_layout()
    save_mpl(fig, FIG_DIR / "fig5_panel_b_discoveries_per_region")
    plt.close(fig)


def fig_per_region_panel_c(traits_cache):
    """% h² captured vs N — 4x4 grid."""
    apply_mpl_style()
    fig, axes = plt.subplots(4, 4, figsize=(MM(180), MM(180)), sharex=True, sharey=True)
    for idx, (label, bre, vol) in enumerate(PAIRS):
        ax = axes.flat[idx]
        for trait, color, name in [(HEIGHT_TRAIT, COL_HT, "Height"),
                                    (vol, COL_VOL, "ENIGMA vol"),
                                    (bre, COL_BRE, "BRE")]:
            p = traits_cache.get(trait, {}).get("proj")
            if p is None:
                continue
            ls = ":" if trait == HEIGHT_TRAIT else "-"
            ax.semilogx(p["N"], 100 * p["frac_h2"], "o" if trait != HEIGHT_TRAIT else "",
                       linestyle=ls, color=color, label=name, ms=2.5, lw=1.1)
        ax.axvline(OUR_N, color="red", linestyle="--", lw=0.5, alpha=0.4)
        ax.set_title(label, fontsize=7.5)
        ax.tick_params(labelsize=6)
        if idx >= 12:
            ax.set_xlabel("N (log)", fontsize=7)
        if idx % 4 == 0:
            ax.set_ylabel("% h² captured", fontsize=7)
        if idx == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 100)
    fig.suptitle("Predicted % heritability captured by GWS loci vs sample size — per region",
                 fontsize=9, y=1.005)
    fig.tight_layout()
    save_mpl(fig, FIG_DIR / "fig5_panel_c_h2_captured_per_region")
    plt.close(fig)


def fig_per_region_panel_a(traits_cache):
    """Effect-size density — 4x4 grid."""
    apply_mpl_style()
    fig, axes = plt.subplots(4, 4, figsize=(MM(180), MM(180)))
    xrange = np.linspace(-0.06, 0.06, 800)
    for idx, (label, bre, vol) in enumerate(PAIRS):
        ax = axes.flat[idx]
        for trait, color, name in [(HEIGHT_TRAIT, COL_HT, "Height"),
                                    (vol, COL_VOL, "ENIGMA vol"),
                                    (bre, COL_BRE, "BRE")]:
            p = traits_cache.get(trait, {}).get("params")
            if p is None:
                continue
            sigma = np.sqrt(p["sigmasq"])
            y = p["pic"] * norm.pdf(xrange, loc=0, scale=sigma)
            ls = ":" if trait == HEIGHT_TRAIT else "-"
            ax.plot(xrange, y, linestyle=ls, color=color, label=name, lw=1.2)
        ax.set_title(label, fontsize=7.5)
        ax.tick_params(labelsize=6)
        if idx >= 12:
            ax.set_xlabel("β", fontsize=7)
        if idx % 4 == 0:
            ax.set_ylabel("density × π_c", fontsize=7)
        if idx == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax.set_xlim(-0.05, 0.05)
    fig.suptitle("Per-SNP effect-size distribution — per region", fontsize=9, y=1.005)
    fig.tight_layout()
    save_mpl(fig, FIG_DIR / "fig5_panel_a_effect_size_per_region")
    plt.close(fig)


def main():
    needed = {HEIGHT_TRAIT}
    for _, bre, vol in PAIRS:
        needed.add(bre)
        needed.add(vol)
    cache = {}
    print(f"Loading {len(needed)} GENESIS fits ...")
    missing = []
    for t in sorted(needed):
        cache[t] = dict(params=load_params(t), proj=load_projection(t))
        if cache[t]["params"] is None or cache[t]["proj"] is None:
            missing.append(t)
    if missing:
        print(f"MISSING traits ({len(missing)}): {missing}")
    print(f"Loaded {len(needed) - len(missing)}/{len(needed)}")
    fig_per_region_panel_b(cache)
    fig_per_region_panel_c(cache)
    fig_per_region_panel_a(cache)
    print("Done:")
    for fn in ("fig5_panel_a_effect_size_per_region",
               "fig5_panel_b_discoveries_per_region",
               "fig5_panel_c_h2_captured_per_region"):
        print(f"  {FIG_DIR}/{fn}.{{pdf,png}}")


if __name__ == "__main__":
    main()
