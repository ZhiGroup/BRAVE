#!/usr/bin/env python3
"""
21_oct_locus_plot.py
────────────────────────────────────────────────────────────────────
Regional association comparison plot for GC-survived pair:
  Top panel    : R.Hippocampus BRE Dim 74 FastGWA  −log10(p)
  Middle panel : INL thickness (left eye) OCT GWAS  −log10(p)
  Bottom panel : Gene track (optional, UCSC refGene format)

Dots are coloured by significance bin (analogous to LD r² bins
in LocusZoom, but without requiring LD computation).

Two-step workflow
─────────────────
Step 1 — find loci with INL signal:
  python 21_oct_locus_plot.py --search

Step 2 — make locus plot (inspect locus_search_hipp_inl.csv first):
  python 21_oct_locus_plot.py --plot --chr 4 --start 53200000 --end 54000000 [--gene_bed refGene_hg19.txt]
────────────────────────────────────────────────────────────────────
"""

import argparse
import io
import os
import subprocess
import sys
from typing import Optional

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(BASE, 'results', 'gc_oct')
FIGURES_DIR = os.path.join(BASE, 'figures', 'gc_oct')

sys.path.insert(0, BASE)
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402

_DIM74_DEFAULT = os.path.join(
    str(cfg.gwas.fastgwa_out),
    'discovery_Right_Hippocampus_QT74.fastGWA.fastGWA'
)
_INL_DEFAULT = os.path.join(
    str(cfg.postgwas.oct_sumstats_dir),
    'eye_oct_80k_march10_2022_pheno11.fastGWA'
)
_LOCI_DEFAULT = os.path.join(BASE, 'results', 'cross_region', 'aggregate_loci.csv')

# ── Significance-bin colours (analogous to LD r² bins) ───────────────────────
_P_THRESHOLDS = [1e-8, 1e-6, 1e-4, 1e-2]
_P_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#1f77b4']
_P_LABELS = ['p < 10⁻⁸', 'p < 10⁻⁶', 'p < 10⁻⁴', 'p < 10⁻²', 'p ≥ 10⁻²']


# ── Colour assignment ─────────────────────────────────────────────────────────
def assign_colors(p):
    # type: (pd.Series) -> pd.Series
    """Assign significance-bin colour to each SNP p-value."""
    colors = pd.Series(_P_COLORS[-1], index=p.index)
    for thr, col in zip(_P_THRESHOLDS, _P_COLORS):
        colors[p < thr] = col
    return colors


# ── Fast file reading via awk ─────────────────────────────────────────────────
def _awk_region(path, chrom, start, end):
    # type: (str, int, int, int) -> pd.DataFrame
    """Extract SNPs in [chrom:start-end] from a FastGWA file using awk."""
    cmd = (
        "awk 'NR==1 || ($1=={c} && $3>={s} && $3<={e})' {f}"
        .format(c=chrom, s=start, e=end, f=path)
    )
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not out.stdout.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(out.stdout), sep='\t')
    except Exception as exc:
        print('  WARNING: parse error in {}: {}'.format(path, exc))
        return pd.DataFrame()


def _awk_multi_region(path, windows):
    # type: (str, list) -> pd.DataFrame
    """Read multiple (chrom, start, end) windows from a FastGWA file in one awk pass."""
    conditions = ' || '.join(
        '($1=={c} && $3>={s} && $3<={e})'.format(c=c, s=s, e=e)
        for c, s, e in windows
    )
    cmd = "awk 'NR==1 || ({cond})' {f}".format(cond=conditions, f=path)
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not out.stdout.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(out.stdout), sep='\t')
    except Exception as exc:
        print('  WARNING: parse error: {}'.format(exc))
        return pd.DataFrame()


# ── Locus search ──────────────────────────────────────────────────────────────
def load_hipp_right_loci(path):
    # type: (str) -> pd.DataFrame
    df = pd.read_csv(path)
    return df[df['regions'].str.contains('R. Hippocampus', na=False)].copy()


def search_loci(loci_df, dim74_path, inl_path, padding=300000):
    # type: (pd.DataFrame, str, str, int) -> pd.DataFrame
    """For each R.Hippocampus locus check dim74 and INL signals (single awk pass each)."""
    windows = []
    for _, locus in loci_df.iterrows():
        c = int(locus['chr'])
        s = max(1, int(locus['start']) - padding)
        e = int(locus['end']) + padding
        windows.append((c, s, e))

    print('Reading BRE Dim74 FastGWA ({} windows, single pass)...'.format(len(windows)))
    dim74_all = _awk_multi_region(dim74_path, windows)
    print('  -> {} SNPs'.format(len(dim74_all)))

    print('Reading INL_L FastGWA (single pass)...')
    inl_all = _awk_multi_region(inl_path, windows)
    print('  -> {} SNPs'.format(len(inl_all)))

    records = []
    for (c, s, e), (_, locus) in zip(windows, loci_df.iterrows()):
        def _min_p(df):
            if len(df) == 0:
                return np.nan
            sub = df[(df['CHR'] == c) & (df['POS'] >= s) & (df['POS'] <= e)]
            return float(sub['P'].min()) if len(sub) > 0 else np.nan

        records.append({
            'al_id':       locus['al_id'],
            'chr':         c,
            'locus_start': int(locus['start']),
            'locus_end':   int(locus['end']),
            'plot_start':  s,
            'plot_end':    e,
            'n_regions':   locus['n_regions'],
            'jagwas_best_p': locus['best_p'],
            'dim74_min_p': _min_p(dim74_all),
            'inl_min_p':   _min_p(inl_all),
        })

    return pd.DataFrame(records).sort_values('inl_min_p')


# ── Gene track ────────────────────────────────────────────────────────────────
def load_genes(gene_bed, chrom, start, end):
    # type: (Optional[str], int, int, int) -> Optional[pd.DataFrame]
    """Load UCSC refGene.txt genes overlapping [chrom:start-end]."""
    if gene_bed is None or not os.path.exists(gene_bed):
        return None
    cols = ['bin', 'name', 'chrom', 'strand', 'txStart', 'txEnd',
            'cdsStart', 'cdsEnd', 'exonCount', 'exonStarts', 'exonEnds',
            'score', 'name2']
    try:
        ref = pd.read_csv(
            gene_bed, sep='\t', header=None,
            names=cols + ['cdsStartStat', 'cdsEndStat', 'exonFrames'],
            usecols=['name2', 'chrom', 'strand', 'txStart', 'txEnd',
                     'exonStarts', 'exonEnds'],
        )
    except Exception:
        try:
            ref = pd.read_csv(
                gene_bed, sep='\t', header=None, names=cols,
                usecols=['name2', 'chrom', 'strand', 'txStart', 'txEnd',
                         'exonStarts', 'exonEnds'],
            )
        except Exception as exc:
            print('WARNING: could not read gene file: {}'.format(exc))
            return None
    ref = ref[ref['chrom'] == 'chr{}'.format(chrom)]
    ref = ref[(ref['txEnd'] >= start) & (ref['txStart'] <= end)]
    # Keep longest transcript per gene
    ref = ref.copy()
    ref['_len'] = ref['txEnd'] - ref['txStart']
    ref = ref.sort_values('_len', ascending=False).drop_duplicates('name2')
    return ref if len(ref) > 0 else None


def draw_gene_track(ax, gene_df, chrom, start, end):
    # type: (plt.Axes, Optional[pd.DataFrame], int, int, int) -> None
    """Draw a simplified gene model track."""
    ax.set_xlim(start / 1e6, end / 1e6)
    ax.set_ylim(-0.2, 1.8)
    ax.set_yticks([])
    for spine in ['left', 'right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.axhline(0.5, color='#aaaaaa', linewidth=0.5, zorder=0)

    if gene_df is None or len(gene_df) == 0:
        ax.text(0.5, 0.5, 'Gene annotation not available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=6, color='grey')
        return

    used_label_x = []
    for _, gene in gene_df.iterrows():
        gs = max(int(gene['txStart']), start)
        ge = min(int(gene['txEnd']), end)
        if gs >= ge:
            continue
        # Gene body
        ax.barh(0.5, ge - gs, left=gs, height=0.2, color='#4472C4', alpha=0.75, linewidth=0)
        # Exons (thicker)
        try:
            ex_s = [int(x) for x in str(gene['exonStarts']).rstrip(',').split(',') if x]
            ex_e = [int(x) for x in str(gene['exonEnds']).rstrip(',').split(',') if x]
            for es, ee in zip(ex_s, ex_e):
                if ee < start or es > end:
                    continue
                ax.barh(0.5, min(ee, end) - max(es, start), left=max(es, start),
                        height=0.35, color='#4472C4', alpha=1.0, linewidth=0)
        except Exception:
            pass
        # Label
        mid = (gs + ge) / 2 / 1e6
        arrow = ' \u2192' if gene['strand'] == '+' else ' \u2190'
        label = gene['name2'] + arrow
        # Avoid overlap
        too_close = any(abs(mid - x) < (end - start) / 1e6 / 12 for x in used_label_x)
        y_label = 0.95 if not too_close else 1.30
        ax.text(mid, y_label, label, ha='center', va='bottom',
                fontsize=5.5, style='italic', clip_on=True)
        used_label_x.append(mid)


# ── Main locus plot ───────────────────────────────────────────────────────────
_JAGWAS_DEFAULT = os.path.join(
    str(cfg.gwas.jagwas_out),
    'jwas_Right_Hippocampus'
)


def make_locus_plot(chrom, plot_start, plot_end,
                    dim74_path, inl_path,
                    top_label='R.Hippocampus BRE Dim 74',
                    top_marker_label='BRE Dim 74 lead SNP',
                    gene_bed=None, out_stem='hipp_inl_locus'):
    # type: (int, int, int, str, str, str, str, Optional[str], str) -> None
    """Create the twin-panel regional association plot."""
    print('Reading top panel GWAS for chr{}:{:,}-{:,}...'.format(chrom, plot_start, plot_end))
    dim74 = _awk_region(dim74_path, chrom, plot_start, plot_end)
    print('  {} SNPs'.format(len(dim74)))

    print('Reading INL_L OCT GWAS...')
    inl = _awk_region(inl_path, chrom, plot_start, plot_end)
    print('  {} SNPs'.format(len(inl)))

    if len(dim74) == 0 and len(inl) == 0:
        print('ERROR: No SNPs found. Check --chr/--start/--end and file format.')
        return

    gene_df = load_genes(gene_bed, chrom, plot_start, plot_end)
    apply_mpl_style()

    n_panels = 3 if gene_df is not None else 2
    hr = [3, 3, 1] if gene_df is not None else [3, 3]
    fig_h = 145 if gene_df is not None else 120

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(MM(178), MM(fig_h)),
        gridspec_kw={'height_ratios': hr, 'hspace': 0.06},
    )

    xmin = plot_start / 1e6
    xmax = plot_end  / 1e6

    # ── Panel 1: BRE Dim 74 ──────────────────────────────────────────────────
    ax1 = axes[0]
    if len(dim74) > 0:
        dim74 = dim74.copy()
        dim74['nlp'] = -np.log10(dim74['P'].clip(lower=1e-300))
        c1 = assign_colors(dim74['P'])
        order = np.argsort(dim74['P'].values)[::-1]
        ax1.scatter(
            dim74['POS'].values[order] / 1e6,
            dim74['nlp'].values[order],
            c=c1.values[order], s=8, linewidth=0, rasterized=True, zorder=2,
        )
        lead1 = dim74.loc[dim74['P'].idxmin()]
        ax1.scatter(lead1['POS'] / 1e6, lead1['nlp'],
                    c='#7030A0', s=60, marker='D',
                    linewidth=0.5, edgecolor='k', zorder=5)
        ax1.annotate(
            lead1['SNP'],
            xy=(lead1['POS'] / 1e6, lead1['nlp']),
            xytext=(5, 3), textcoords='offset points', fontsize=7,
        )
    ax1.axhline(-np.log10(5e-8), color='#d62728', ls='--', lw=0.8, alpha=0.9)
    ax1.axhline(-np.log10(1e-5), color='#4472C4', ls=':', lw=0.6, alpha=0.7)
    ax1.set_ylabel('{}\n\u2212log\u2081\u2080(p)'.format(top_label), fontsize=8)
    ax1.set_xlim(xmin, xmax)
    ax1.tick_params(axis='x', labelbottom=False, bottom=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # cytogenetic-style title (approximate band from position)
    ax1.set_title('chr{}'.format(chrom), fontsize=9, fontweight='bold', pad=4)

    # ── Panel 2: INL_L ───────────────────────────────────────────────────────
    ax2 = axes[1]
    if len(inl) > 0:
        inl = inl.copy()
        inl['nlp'] = -np.log10(inl['P'].clip(lower=1e-300))
        c2 = assign_colors(inl['P'])
        order = np.argsort(inl['P'].values)[::-1]
        ax2.scatter(
            inl['POS'].values[order] / 1e6,
            inl['nlp'].values[order],
            c=c2.values[order], s=8, linewidth=0, rasterized=True, zorder=2,
        )
        lead2 = inl.loc[inl['P'].idxmin()]
        ax2.scatter(lead2['POS'] / 1e6, lead2['nlp'],
                    c='darkorange', s=60, marker='o',
                    linewidth=0.5, edgecolor='k', zorder=5)
        if 'SNP' in inl.columns:
            ax2.annotate(
                lead2['SNP'],
                xy=(lead2['POS'] / 1e6, lead2['nlp']),
                xytext=(5, 3), textcoords='offset points', fontsize=7,
            )
    ax2.axhline(-np.log10(5e-8), color='#d62728', ls='--', lw=0.8, alpha=0.9)
    ax2.axhline(-np.log10(1e-5), color='#4472C4', ls=':', lw=0.6, alpha=0.7)
    ax2.set_ylabel('INL thickness (left eye)\nOCT GWAS \u2212log\u2081\u2080(p)', fontsize=8)
    ax2.set_xlim(xmin, xmax)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Panel 3: Gene track ───────────────────────────────────────────────────
    if gene_df is not None:
        ax3 = axes[2]
        draw_gene_track(ax3, gene_df, chrom, plot_start, plot_end)
        ax3.set_xlim(xmin, xmax)
        ax3.set_xlabel('Chromosome {} position (Mb)'.format(chrom), fontsize=8)
        ax2.tick_params(axis='x', labelbottom=False, bottom=False)
        ax2.spines['bottom'].set_visible(False)
    else:
        ax2.set_xlabel('Chromosome {} position (Mb)'.format(chrom), fontsize=8)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(_P_COLORS, _P_LABELS)
    ]
    legend_markers = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#7030A0',
               markeredgecolor='k', markersize=6, linewidth=0,
               label=top_marker_label),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',
               markeredgecolor='k', markersize=6, linewidth=0,
               label='INL GWAS lead SNP'),
        Line2D([0], [0], color='#d62728', ls='--', lw=1, label='p = 5\u00d710\u207b\u2078'),
        Line2D([0], [0], color='#4472C4', ls=':', lw=1, label='p = 10\u207b\u2075'),
    ]
    axes[0].legend(
        handles=legend_patches + legend_markers,
        fontsize=6.5, frameon=False,
        bbox_to_anchor=(1.01, 1.02), loc='upper left',
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, out_stem)
    save_mpl(fig, out_path)
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    # type: () -> None
    ap = argparse.ArgumentParser(
        description='Locus plot: R.Hippocampus BRE Dim74 vs OCT INL_L',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python 21_oct_locus_plot.py --search\n'
            '  python 21_oct_locus_plot.py --plot --chr 4 --start 53200000 --end 54000000\n'
            '  python 21_oct_locus_plot.py --plot --chr 4 --start 53200000 --end 54000000'
            ' --gene_bed /path/to/refGene_hg19.txt'
        ),
    )

    ap.add_argument('--dim74', default=_DIM74_DEFAULT,
                    help='Top-panel GWAS path (BRE Dim74 FastGWA by default)')
    ap.add_argument('--jagwas', action='store_true',
                    help='Use JAGWAS multivariate file for top panel instead of Dim74')
    ap.add_argument('--top_label', default=None,
                    help='Top panel y-axis label (auto-set based on --jagwas flag if omitted)')
    ap.add_argument('--inl',   default=_INL_DEFAULT,
                    help='INL_thickness_left fastGWA path')
    ap.add_argument('--loci',  default=_LOCI_DEFAULT,
                    help='aggregate_loci.csv path')
    ap.add_argument('--gene_bed', default=None,
                    help='UCSC refGene.txt gene annotation file (optional)')
    ap.add_argument('--padding', type=int, default=300000,
                    help='Flanking bp for locus search (default: 300000)')

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--search', action='store_true',
                      help='Search R.Hippocampus loci for INL signal')
    mode.add_argument('--plot',   action='store_true',
                      help='Make locus plot for a specified region')

    ap.add_argument('--chr',   type=int, dest='chrom', metavar='CHR',
                    help='Chromosome (required for --plot)')
    ap.add_argument('--start', type=int, metavar='START',
                    help='Region start bp (required for --plot)')
    ap.add_argument('--end',   type=int, metavar='END',
                    help='Region end bp (required for --plot)')
    ap.add_argument('--out',   default='hipp_inl_locus',
                    help='Output filename stem (default: hipp_inl_locus)')

    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.search:
        print('Loading R.Hippocampus JAGWAS loci...')
        loci_df = load_hipp_right_loci(args.loci)
        print('  {} loci found'.format(len(loci_df)))

        results = search_loci(loci_df, args.dim74, args.inl, padding=args.padding)

        out_csv = os.path.join(RESULTS_DIR, 'locus_search_hipp_inl.csv')
        results.to_csv(out_csv, index=False)
        print('\nSaved: {}'.format(out_csv))
        print('\nTop 10 loci by INL signal:')
        cols = ['al_id', 'chr', 'locus_start', 'locus_end',
                'jagwas_best_p', 'dim74_min_p', 'inl_min_p']
        print(results[cols].head(10).to_string(index=False))

    elif args.plot:
        for flag, val in [('--chr', args.chrom), ('--start', args.start), ('--end', args.end)]:
            if val is None:
                ap.error('{} is required with --plot'.format(flag))
        # Resolve top-panel path and labels
        if args.jagwas:
            top_path = _JAGWAS_DEFAULT
            top_label = args.top_label or 'R.Hippocampus JAGWAS'
            top_marker_label = 'JAGWAS lead SNP'
        else:
            top_path = args.dim74
            top_label = args.top_label or 'R.Hippocampus BRE Dim 74'
            top_marker_label = 'BRE Dim 74 lead SNP'
        make_locus_plot(
            chrom=args.chrom,
            plot_start=args.start,
            plot_end=args.end,
            dim74_path=top_path,
            inl_path=args.inl,
            top_label=top_label,
            top_marker_label=top_marker_label,
            gene_bed=args.gene_bed,
            out_stem=args.out,
        )


if __name__ == '__main__':
    main()
