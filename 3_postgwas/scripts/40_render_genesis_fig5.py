#!/usr/bin/env python3
# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.
"""
40_render_genesis_fig5.py
Render Wen et al. (2025) Fig 5 equivalent from GENESIS pilot output.

Three panels (a, b, c) following Wen Fig 5:
  a — Effect-size distribution density (per-trait overlay)
  b — Predicted # GWS-significant loci vs sample size
  c — Predicted % heritability captured vs sample size

Inputs:
  results/genesis/<trait>/pilot_summary_params.csv  (per-trait π_c, σ², h²)
  results/genesis/<trait>/projection.csv             (per-trait N grid + # disc + h² captured)
Outputs:
  figures/genesis/fig5_genesis_polygenicity.{pdf,png}
"""
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

# Pilot trait set (used if full sweep not done yet) — same colours throughout
PILOT_TRAITS = [
    ("BRE_R.Caudate_QT81",   "BRE R.Caudate (h² top)",  "#1f77b4"),
    ("BRE_R.Thalamus_QT40",  "BRE R.Thalamus (h² low)", "#aec7e8"),
    ("ENIGMA_Caudate",       "ENIGMA Caudate volume",   "#ff7f0e"),
    ("ENIGMA_Thalamus",      "ENIGMA Thalamus volume",  "#ffbb78"),
    ("height_REFERENCE",     "Height (reference)",      "#888888"),
]


def load_fit_params(trait):
    """Load 2-component fit parameters using R via Rscript (avoids loading rds in Python)."""
    rscript = "<EXTERNAL: Rscript>"
    zhang_lib = "<EXTERNAL: library_zhang_genesis>"
    rds = GENESIS_DIR / trait / "fit_2comp.rds"
    if not rds.exists():
        return None
    r_cmd = f"""
.libPaths(c('{zhang_lib}', .libPaths()))
fit <- readRDS('{rds}')
est <- fit$estimates
par <- est[['Parameter (pic, sigmasq, a) estimates']]
sd  <- est[['S.D. of parameter estimates']]
h2_str <- est[['Total heritability in log-odds-ratio scale (sd)']]
nssnp_str <- est[['Number of sSNPs (sd)']]
M <- est[['Total number of SNPs in the GWAS study after quality control']]
cat(sprintf('%.10g\\n', par[1]))
cat(sprintf('%.10g\\n', par[2]))
cat(sprintf('%.10g\\n', par[3]))
cat(sprintf('%s\\n', h2_str))
cat(sprintf('%s\\n', nssnp_str))
cat(sprintf('%d\\n', M))
"""
    import subprocess
    res = subprocess.run([rscript, "-e", r_cmd], capture_output=True, text=True)
    lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
    if len(lines) < 6:
        return None
    pic, sigmasq, a = float(lines[0]), float(lines[1]), float(lines[2])
    h2_str = lines[3]
    nssnp_str = lines[4]
    M = int(lines[5])
    # Parse "0.418 (0.05275)" -> (0.418, 0.05275)
    def parse_paren(s):
        try:
            main, sd = s.split(" (")
            return float(main), float(sd.rstrip(")"))
        except Exception:
            return float(s), 0.0
    h2, h2_sd = parse_paren(h2_str)
    nssnp, nssnp_sd = parse_paren(nssnp_str)
    return dict(pic=pic, sigmasq=sigmasq, intercept=a,
                h2=h2, h2_sd=h2_sd,
                nssnp=nssnp, nssnp_sd=nssnp_sd,
                M=M)


def load_projection(trait):
    pf = GENESIS_DIR / trait / "projection.csv"
    if not pf.exists():
        return None
    # projection.csv cols: N, n_discoveries (which is actually heritability constant), GV_pct (actually # discoveries), herit_explained (% h² captured)
    # Per inspection, p[1]=h² (const), p[2]=# discoveries, p[3]=% h² captured
    df = pd.read_csv(pf)
    df = df.rename(columns={
        "n_discoveries": "h2_const",      # constant per trait (= h²)
        "GV_pct": "n_disc",                # # discoveries
        "herit_explained": "frac_h2",     # fraction of h² captured
    })
    return df


def panel_a_effect_size(ax, traits_data):
    """Effect-size distribution density (Gaussian with variance σ² weighted by π_c)."""
    x = np.linspace(-0.08, 0.08, 1000)
    for trait, label, color in PILOT_TRAITS:
        d = traits_data.get(trait)
        if d is None or d["params"] is None:
            continue
        pic = d["params"]["pic"]
        sigma = np.sqrt(d["params"]["sigmasq"])
        # Causal-effect density: pic * N(0, σ²) — peaks higher when σ² smaller
        y = pic * norm.pdf(x, loc=0, scale=sigma)
        ax.plot(x, y, color=color, label=label, lw=1.5)
    ax.set_xlabel("Per-SNP effect size β")
    ax.set_ylabel("Density × π_c")
    ax.set_title("a   Effect-size distribution")
    ax.set_xlim(-0.05, 0.05)
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def panel_b_discoveries(ax, traits_data):
    for trait, label, color in PILOT_TRAITS:
        d = traits_data.get(trait)
        if d is None or d["proj"] is None:
            continue
        p = d["proj"]
        ax.semilogx(p["N"], p["n_disc"], "o-", color=color, label=label, ms=4, lw=1.2)
    # Mark our N = 22,878
    ax.axvline(22878, color="red", linestyle="--", lw=0.8, alpha=0.5)
    ax.text(22878, ax.get_ylim()[1] * 0.95, "  our N",
            color="red", fontsize=7, va="top")
    ax.set_xlabel("Sample size N (log scale)")
    ax.set_ylabel("Predicted # GWAS-significant loci")
    ax.set_title("b   # discoveries vs N")
    ax.grid(True, alpha=0.3)


def panel_c_h2_captured(ax, traits_data):
    for trait, label, color in PILOT_TRAITS:
        d = traits_data.get(trait)
        if d is None or d["proj"] is None:
            continue
        p = d["proj"]
        ax.semilogx(p["N"], 100 * p["frac_h2"], "o-", color=color, label=label, ms=4, lw=1.2)
    ax.axvline(22878, color="red", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Sample size N (log scale)")
    ax.set_ylabel("% h² captured by GWS loci")
    ax.set_title("c   % heritability captured vs N")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)


def main():
    # Load all trait data
    traits_data = {}
    for trait, label, color in PILOT_TRAITS:
        traits_data[trait] = dict(
            params=load_fit_params(trait),
            proj=load_projection(trait),
        )
    n_loaded = sum(1 for d in traits_data.values()
                   if d["params"] is not None and d["proj"] is not None)
    print(f"Loaded {n_loaded}/{len(PILOT_TRAITS)} traits")
    for t, d in traits_data.items():
        if d["params"] is None:
            print(f"  MISSING: {t}")
        else:
            p = d["params"]
            print(f"  {t:30s} π_c={p['pic']:.4f} σ²={p['sigmasq']:.2e} h²={p['h2']:.3f} #sSNPs={p['nssnp']:.0f}")

    apply_mpl_style()
    fig, axes = plt.subplots(1, 3, figsize=(MM(180), MM(70)))
    panel_a_effect_size(axes[0], traits_data)
    panel_b_discoveries(axes[1], traits_data)
    panel_c_h2_captured(axes[2], traits_data)
    fig.tight_layout()
    save_mpl(fig, FIG_DIR / "fig5_genesis_polygenicity")
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR}/fig5_genesis_polygenicity.{{pdf,png}}")


if __name__ == "__main__":
    main()
