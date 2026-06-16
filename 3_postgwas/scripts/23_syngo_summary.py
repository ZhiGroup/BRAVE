#!/usr/bin/env python
"""Script 23: SynGO ORA summary figure.

Reads per-region SynGO ORA output and produces a two-panel supplementary figure:
  Panel A — dot plot: each tested (region × GO term) as a dot, y=-log10(FDR q),
             colour by GO domain; FDR=0.05 threshold line; sparse x-axis = regions.
  Panel B — summary bar: genes mapped to SynGO background per region.

Output: figures/syngo/syngo_ora_summary.pdf/.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fig_style import apply_mpl_style, save_mpl, MM

import openpyxl


REGION_ORDER = [
    "L_Thalamus", "R_Thalamus",
    "L. Caudate", "R_Caudate",
    "L_Putamen", "R_Putamen",
    "L_Pallidum", "R_Pallidum",
    "L._Accumbens", "R._Accumbens",
    "L. Hippocampus", "R_Hippocampus",
    "L-_Amygdala", "R_Amygdala",
    "Brain_Stem", "CSF",
]

REGION_LABELS = {
    "L_Thalamus":     "L.Thalamus",
    "R_Thalamus":     "R.Thalamus",
    "L. Caudate":     "L.Caudate",
    "R_Caudate":      "R.Caudate",
    "L_Putamen":      "L.Putamen",
    "R_Putamen":      "R.Putamen",
    "L_Pallidum":     "L.Pallidum",
    "R_Pallidum":     "R.Pallidum",
    "L._Accumbens":   "L.Accumbens",
    "R._Accumbens":   "R.Accumbens",
    "L. Hippocampus": "L.Hippocampus",
    "R_Hippocampus":  "R.Hippocampus",
    "L-_Amygdala":    "L.Amygdala",
    "R_Amygdala":     "R.Amygdala",
    "Brain_Stem":     "BrainStem",
    "CSF":            "CSF",
}

DOMAIN_COLOURS = {"BP": "#4878CF", "CC": "#6ACC65", "MF": "#D65F5F"}


def load_syngo_region(base, region):
    """Return (results_df, n_mapped) for one region."""
    fpath = os.path.join(base, region,
                         "syngo_ontologies_with_annotations_matching_user_input.xlsx")
    wb = openpyxl.load_workbook(fpath)
    ws = wb.active
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    df = pd.DataFrame(rows, columns=headers)
    df.rename(columns={
        "GO term ID": "go_id",
        "GO domain": "domain",
        "GO term name": "go_term",
        "GSEA p-value": "pval",
        "GSEA FDR corrected p-value": "fdr",
        "GSEA count foreground/input": "fg_count",
    }, inplace=True)
    # keep only tested rows (have a p-value)
    df = df[df["pval"].notna() & (df["pval"] != "")].copy()
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
    df["fdr"] = pd.to_numeric(df["fdr"], errors="coerce")
    df = df.dropna(subset=["pval", "fdr"])

    # gene mapping count
    gpath = os.path.join(base, region, "user_genelist_map_to_syngo_genes.xlsx")
    wb2 = openpyxl.load_workbook(gpath)
    ws2 = wb2.active
    grows = list(ws2.iter_rows(min_row=2, values_only=True))
    n_mapped = sum(1 for r in grows if r[0] not in ("", None))

    return df, n_mapped


def main(args):
    base = args.syngo_dir
    out_stem = args.out_stem

    records = []
    n_mapped_dict = {}
    for region in REGION_ORDER:
        df, nm = load_syngo_region(base, region)
        n_mapped_dict[region] = nm
        for _, row in df.iterrows():
            records.append({
                "region": region,
                "go_id": row["go_id"],
                "go_term": row["go_term"],
                "domain": row["domain"],
                "pval": row["pval"],
                "fdr": row["fdr"],
            })

    data = pd.DataFrame(records)
    data["neglog_fdr"] = -np.log10(data["fdr"].clip(lower=1e-10))

    apply_mpl_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(MM(180), MM(65)),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.35},
    )

    # ── Panel A: dot plot ──────────────────────────────────────────────────
    x_positions = {r: i for i, r in enumerate(REGION_ORDER)}
    FDR_THRESH = -np.log10(0.05)

    jitter_seed = 42
    rng = np.random.default_rng(jitter_seed)

    for domain, colour in DOMAIN_COLOURS.items():
        sub = data[data["domain"] == domain]
        if sub.empty:
            continue
        x = np.array([x_positions[r] for r in sub["region"]]) + rng.uniform(-0.3, 0.3, len(sub))
        ax1.scatter(x, sub["neglog_fdr"].values,
                    s=6, alpha=0.65, color=colour, linewidths=0,
                    zorder=3, label=domain)

    # FDR threshold line
    ax1.axhline(FDR_THRESH, color="#CC0000", linewidth=0.8, linestyle="--",
                zorder=4, label="FDR = 0.05")

    # Mark FDR-significant hits with larger open circles
    sig = data[data["fdr"] < 0.05].copy()
    for _, row in sig.iterrows():
        xi = x_positions[row["region"]]
        ax1.scatter(xi, row["neglog_fdr"],
                    s=28, facecolors="none", edgecolors="#CC0000",
                    linewidths=0.8, zorder=5)

    # Single annotation for the repeated term across 3 regions
    if not sig.empty:
        # use the rightmost of the three FDR-sig regions
        xi_right = max(x_positions[r] for r in sig["region"])
        y_top = sig["neglog_fdr"].max()
        ax1.annotate(
            "GO:0099159\n(reg. postsynaptic\nstructure modif.)\n3 regions",
            xy=(xi_right, y_top),
            xytext=(xi_right + 1.0, y_top + 0.7),
            fontsize=4.5,
            color="#CC0000",
            arrowprops=dict(arrowstyle="-", color="#CC0000", lw=0.5),
            ha="left", va="bottom",
        )

    ax1.set_xticks(range(len(REGION_ORDER)))
    ax1.set_xticklabels(
        [REGION_LABELS[r] for r in REGION_ORDER],
        rotation=45, ha="right", fontsize=6,
    )
    ax1.set_ylabel("−log₁₀(FDR q)", fontsize=8)
    ax1.set_title(
        "SynGO ORA: no classical synaptic enrichment in any region",
        fontsize=9, pad=4,
    )
    ax1.set_xlim(-0.7, len(REGION_ORDER) - 0.3)
    ax1.set_ylim(bottom=0)

    # deduplicate legend (one annotation per domain + threshold)
    seen_labels = []
    seen_handles = []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        if l not in seen_labels:
            seen_labels.append(l)
            seen_handles.append(h)
    ax1.legend(seen_handles, seen_labels,
               fontsize=6, frameon=False,
               bbox_to_anchor=(1.01, 1.0), loc="upper right")

    # Panel A label
    ax1.text(-0.06, 1.04, "A", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top")

    # ── Panel B: genes mapped bar ──────────────────────────────────────────
    bars = [n_mapped_dict[r] for r in REGION_ORDER]
    y_pos = np.arange(len(REGION_ORDER))
    ax2.barh(y_pos, bars, height=0.65, color="#999999", edgecolor="none")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([REGION_LABELS[r] for r in REGION_ORDER], fontsize=6)
    ax2.set_xlabel("Genes mapped to SynGO", fontsize=8)
    ax2.set_title("Gene input", fontsize=9, pad=4)
    ax2.invert_yaxis()

    # Panel B label
    ax2.text(-0.18, 1.04, "B", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top")

    save_mpl(fig, out_stem)
    print(f"Saved: {out_stem}.pdf / .png")

    # Print summary table
    summary = []
    for region in REGION_ORDER:
        sub = data[data["region"] == region]
        summary.append({
            "region": REGION_LABELS[region],
            "n_tested": len(sub),
            "n_p05": int((sub["pval"] < 0.05).sum()),
            "n_fdr05": int((sub["fdr"] < 0.05).sum()),
            "n_mapped": n_mapped_dict[region],
        })
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynGO ORA summary figure")
    parser.add_argument(
        "--syngo_dir",
        default="<EXTERNAL: per-region SynGO ORA output root directory (SYNGO_OUTPUT)>",
        help="Path to SynGO output root directory",
    )
    parser.add_argument(
        "--out_stem",
        default="figures/syngo/syngo_ora_summary",
        help="Output stem (PDF+PNG written automatically)",
    )
    args = parser.parse_args()
    main(args)
