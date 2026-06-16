#!/usr/bin/env python3
"""
22_heart_locus_plot.py
────────────────────────────────────────────────────────────────
Regional association comparison plots for the two FDR-significant
limbic–cardiac GC pairs (Script 18b):

  Pair A (hipp_laef):  L.Hippocampus BRE Dim 110  ×  LAEF (pheno31)
  Pair B (amyg_lavmin): R.Amygdala   BRE Dim  95  ×  LAV_min (pheno29)

Two-step workflow (run for each pair independently)
─────────────────────────────────────────────────────
Step 1 — search loci for a pair:
  python 22_heart_locus_plot.py --search --pair hipp_laef
  python 22_heart_locus_plot.py --search --pair amyg_lavmin

Step 2 — inspect the search CSV, then make the locus plot:
  python 22_heart_locus_plot.py --plot --pair hipp_laef  --chr <C> --start <S> --end <E>
  python 22_heart_locus_plot.py --plot --pair amyg_lavmin --chr <C> --start <S> --end <E>
────────────────────────────────────────────────────────────────
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

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(BASE, 'results', 'gc_heart')
FIGURES_DIR = os.path.join(BASE, 'figures', 'gc_heart')

sys.path.insert(0, BASE)
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402

_GWAS_BASE = os.path.join(
    str(cfg.gwas.fastgwa_out),
    'discovery_{region}_QT{dim}.fastGWA.fastGWA'
)

_HEART_BASE = os.path.join(
    str(cfg.postgwas.heart_sumstats_dir),
    'ukb_phase1to3_heart_may_2022_pheno{pheno}.fastGWA'
)

_LOCI_DEFAULT = os.path.join(BASE, 'results', 'cross_region', 'aggregate_loci.csv')

# ── Pair definitions ──────────────────────────────────────────────────────────
PAIRS = {
    'hipp_laef': {
        'bre_region':  'Left_Hippocampus',
        'bre_dim':     110,
        'region_display': 'L.Hippocampus',
        'region_col_name': 'L. Hippocampus',  # in aggregate_loci 'regions' column
        'trait_pheno': 31,
        'trait_name':  'LAEF',
        'trait_label': 'LA ejection fraction (LAEF)',
        'top_label':   'L.Hippocampus BRE Dim 110',
        'bot_label':   'LAEF (pheno31)\nHeart GWAS \u2212log\u2081\u2080(p)',
        'search_csv':  'locus_search_hipp_laef.csv',
        'out_stem':    'hipp_laef_locus',
        'lead_marker_label': 'BRE Dim 110 lead SNP',
    },
    'amyg_lavmin': {
        'bre_region':  'Right_Amygdala',
        'bre_dim':     95,
        'region_display': 'R.Amygdala',
        'region_col_name': 'R. Amygdala',
        'trait_pheno': 29,
        'trait_name':  'LAV_min',
        'trait_label': 'LA min volume (LAV_min)',
        'top_label':   'R.Amygdala BRE Dim 95',
        'bot_label':   'LAV_min (pheno29)\nHeart GWAS \u2212log\u2081\u2080(p)',
        'search_csv':  'locus_search_amyg_lavmin.csv',
        'out_stem':    'amyg_lavmin_locus',
        'lead_marker_label': 'BRE Dim 95 lead SNP',
    },
}

# ── Significance-bin colours ───────────────────────────────────────────────────
_P_THRESHOLDS = [1e-8, 1e-6, 1e-4, 1e-2]
_P_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#1f77b4']
_P_LABELS = ['p < 10\u207b\u2078', 'p < 10\u207b\u2076', 'p < 10\u207b\u2074',
             'p < 10\u207b\u00b2', 'p \u2265 10\u207b\u00b2']


def assign_colors(p):
    # type: (pd.Series) -> pd.Series
    colors = pd.Series(_P_COLORS[-1], index=p.index)
    for thr, col in zip(_P_THRESHOLDS, _P_COLORS):
        colors[p < thr] = col
    return colors


# ── Fast file reading via awk ─────────────────────────────────────────────────
def _awk_region(path, chrom, start, end):
    # type: (str, int, int, int) -> pd.DataFrame
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
def load_region_loci(loci_path, region_col_name):
    # type: (str, str) -> pd.DataFrame
    df = pd.read_csv(loci_path)
    return df[df['regions'].str.contains(region_col_name, na=False)].copy()


def search_loci(loci_df, bre_path, heart_path, pair_cfg, padding=300000):
    # type: (pd.DataFrame, str, str, dict, int) -> pd.DataFrame
    windows = []
    for _, locus in loci_df.iterrows():
        c = int(locus['chr'])
        s = max(1, int(locus['start']) - padding)
        e = int(locus['end']) + padding
        windows.append((c, s, e))

    print('Reading BRE {} FastGWA ({} windows, single pass)...'.format(
        pair_cfg['top_label'], len(windows)))
    bre_all = _awk_multi_region(bre_path, windows)
    print('  -> {} SNPs'.format(len(bre_all)))

    print('Reading {} Heart GWAS (single pass)...'.format(pair_cfg['trait_name']))
    heart_all = _awk_multi_region(heart_path, windows)
    print('  -> {} SNPs'.format(len(heart_all)))

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
            'bre_min_p':   _min_p(bre_all),
            'heart_min_p': _min_p(heart_all),
        })

    return pd.DataFrame(records).sort_values('heart_min_p')


# ── Gene track ────────────────────────────────────────────────────────────────
def load_genes(gene_bed, chrom, start, end):
    # type: (Optional[str], int, int, int) -> Optional[pd.DataFrame]
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
    ref = ref.copy()
    ref['_len'] = ref['txEnd'] - ref['txStart']
    ref = ref.sort_values('_len', ascending=False).drop_duplicates('name2')
    return ref if len(ref) > 0 else None


def draw_gene_track(ax, gene_df, chrom, start, end):
    # type: (plt.Axes, Optional[pd.DataFrame], int, int, int) -> None
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
        ax.barh(0.5, ge - gs, left=gs, height=0.2, color='#4472C4', alpha=0.75, linewidth=0)
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
        mid = (gs + ge) / 2 / 1e6
        arrow = ' \u2192' if gene['strand'] == '+' else ' \u2190'
        label = gene['name2'] + arrow
        too_close = any(abs(mid - x) < (end - start) / 1e6 / 12 for x in used_label_x)
        y_label = 0.95 if not too_close else 1.30
        ax.text(mid, y_label, label, ha='center', va='bottom',
                fontsize=5.5, style='italic', clip_on=True)
        used_label_x.append(mid)


# ── Locus plot ────────────────────────────────────────────────────────────────
def make_locus_plot(chrom, plot_start, plot_end, bre_path, heart_path,
                    pair_cfg, gene_bed=None):
    # type: (int, int, int, str, str, dict, Optional[str]) -> None
    print('Reading top panel {} for chr{}:{:,}-{:,}...'.format(
        pair_cfg['top_label'], chrom, plot_start, plot_end))
    bre = _awk_region(bre_path, chrom, plot_start, plot_end)
    print('  {} SNPs'.format(len(bre)))

    print('Reading {} Heart GWAS...'.format(pair_cfg['trait_name']))
    heart = _awk_region(heart_path, chrom, plot_start, plot_end)
    print('  {} SNPs'.format(len(heart)))

    if len(bre) == 0 and len(heart) == 0:
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
    xmax = plot_end   / 1e6

    # ── Panel 1: BRE dim ──────────────────────────────────────────────────────
    ax1 = axes[0]
    if len(bre) > 0:
        bre = bre.copy()
        bre['nlp'] = -np.log10(bre['P'].clip(lower=1e-300))
        c1 = assign_colors(bre['P'])
        order = np.argsort(bre['P'].values)[::-1]
        ax1.scatter(
            bre['POS'].values[order] / 1e6,
            bre['nlp'].values[order],
            c=c1.values[order], s=8, linewidth=0, rasterized=True, zorder=2,
        )
        lead1 = bre.loc[bre['P'].idxmin()]
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
    ax1.set_ylabel('{}\n\u2212log\u2081\u2080(p)'.format(pair_cfg['top_label']), fontsize=8)
    ax1.set_xlim(xmin, xmax)
    ax1.tick_params(axis='x', labelbottom=False, bottom=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('chr{}'.format(chrom), fontsize=9, fontweight='bold', pad=4)

    # ── Panel 2: Heart trait ──────────────────────────────────────────────────
    ax2 = axes[1]
    if len(heart) > 0:
        heart = heart.copy()
        heart['nlp'] = -np.log10(heart['P'].clip(lower=1e-300))
        c2 = assign_colors(heart['P'])
        order = np.argsort(heart['P'].values)[::-1]
        ax2.scatter(
            heart['POS'].values[order] / 1e6,
            heart['nlp'].values[order],
            c=c2.values[order], s=8, linewidth=0, rasterized=True, zorder=2,
        )
        lead2 = heart.loc[heart['P'].idxmin()]
        ax2.scatter(lead2['POS'] / 1e6, lead2['nlp'],
                    c='darkorange', s=60, marker='o',
                    linewidth=0.5, edgecolor='k', zorder=5)
        if 'SNP' in heart.columns:
            ax2.annotate(
                lead2['SNP'],
                xy=(lead2['POS'] / 1e6, lead2['nlp']),
                xytext=(5, 3), textcoords='offset points', fontsize=7,
            )
    ax2.axhline(-np.log10(5e-8), color='#d62728', ls='--', lw=0.8, alpha=0.9)
    ax2.axhline(-np.log10(1e-5), color='#4472C4', ls=':', lw=0.6, alpha=0.7)
    ax2.set_ylabel(pair_cfg['bot_label'], fontsize=8)
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

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(_P_COLORS, _P_LABELS)
    ]
    legend_markers = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#7030A0',
               markeredgecolor='k', markersize=6, linewidth=0,
               label=pair_cfg['lead_marker_label']),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',
               markeredgecolor='k', markersize=6, linewidth=0,
               label='{} lead SNP'.format(pair_cfg['trait_name'])),
        Line2D([0], [0], color='#d62728', ls='--', lw=1, label='p = 5\u00d710\u207b\u2078'),
        Line2D([0], [0], color='#4472C4', ls=':', lw=1, label='p = 10\u207b\u2075'),
    ]
    axes[0].legend(
        handles=legend_patches + legend_markers,
        fontsize=6.5, frameon=False,
        bbox_to_anchor=(1.01, 1.02), loc='upper left',
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, pair_cfg['out_stem'])
    save_mpl(fig, out_path)
    plt.close()
    print('Saved: {}.pdf/.png'.format(out_path))


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    # type: () -> None
    ap = argparse.ArgumentParser(
        description='Locus plots for FDR-significant heart GC pairs (Script 18b)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python 22_heart_locus_plot.py --search --pair hipp_laef\n'
            '  python 22_heart_locus_plot.py --search --pair amyg_lavmin\n'
            '  python 22_heart_locus_plot.py --plot --pair hipp_laef '
            '--chr 6 --start 26000000 --end 27000000\n'
            '  python 22_heart_locus_plot.py --plot --pair amyg_lavmin '
            '--chr 4 --start 110000000 --end 111000000\n'
        ),
    )
    ap.add_argument('--pair', required=True, choices=list(PAIRS.keys()),
                    help='Which pair to analyse: hipp_laef or amyg_lavmin')
    ap.add_argument('--loci', default=_LOCI_DEFAULT,
                    help='aggregate_loci.csv path')
    ap.add_argument('--gene_bed', default=None,
                    help='UCSC refGene.txt gene annotation file (optional)')
    ap.add_argument('--padding', type=int, default=300000,
                    help='Flanking bp for locus search (default: 300000)')

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--search', action='store_true',
                      help='Search region loci for heart trait signal')
    mode.add_argument('--plot', action='store_true',
                      help='Make locus plot for a specified region')

    ap.add_argument('--chr',   type=int, dest='chrom', metavar='CHR',
                    help='Chromosome (required for --plot)')
    ap.add_argument('--start', type=int, metavar='START',
                    help='Region start bp (required for --plot)')
    ap.add_argument('--end',   type=int, metavar='END',
                    help='Region end bp (required for --plot)')

    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cfg = PAIRS[args.pair]
    bre_path = _GWAS_BASE.format(region=cfg['bre_region'], dim=cfg['bre_dim'])
    heart_path = _HEART_BASE.format(pheno=cfg['trait_pheno'])

    print('Pair: {} x {}'.format(cfg['region_display'], cfg['trait_name']))
    print('BRE path : {}'.format(bre_path))
    print('Heart path: {}'.format(heart_path))

    if args.search:
        print('\nLoading {} loci...'.format(cfg['region_display']))
        loci_df = load_region_loci(args.loci, cfg['region_col_name'])
        print('  {} loci found'.format(len(loci_df)))

        results = search_loci(loci_df, bre_path, heart_path, cfg, padding=args.padding)

        out_csv = os.path.join(RESULTS_DIR, cfg['search_csv'])
        results.to_csv(out_csv, index=False)
        print('\nSaved: {}'.format(out_csv))
        print('\nTop 10 loci by heart signal:')
        cols = ['al_id', 'chr', 'locus_start', 'locus_end',
                'jagwas_best_p', 'bre_min_p', 'heart_min_p']
        print(results[cols].head(10).to_string(index=False))

    elif args.plot:
        for flag, val in [('--chr', args.chrom), ('--start', args.start), ('--end', args.end)]:
            if val is None:
                ap.error('{} is required with --plot'.format(flag))
        make_locus_plot(
            chrom=args.chrom,
            plot_start=args.start,
            plot_end=args.end,
            bre_path=bre_path,
            heart_path=heart_path,
            pair_cfg=cfg,
            gene_bed=args.gene_bed,
        )


if __name__ == '__main__':
    main()
