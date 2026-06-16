#!/usr/bin/env python
"""
Script 26: FinnGen R10 PheWAS of 276 JAGWAS lead SNPs.

For each FinnGen R10 disease endpoint, streams the summary statistics file
and extracts rows matching any of the 276 JAGWAS aggregate-loci lead SNPs.
Results are aggregated by ICD category and plotted.

Outputs
-------
results/finngen_phewas/
    finngen_hits.csv              all SNP × endpoint hits (p < pthresh)
    finngen_summary_category.csv  per simplified-category: n_loci, n_endpoints
    finngen_summary_region.csv    per region × category: n_loci

figures/finngen_phewas/
    finngen_category_bar.pdf/.png  bar: n_loci per category
    finngen_bubble.pdf/.png        bubble: region × category (n_loci, size=n_hits)

Usage
-----
# Phase 1: download + query (writes checkpoint after every endpoint)
python 26_finngen_phewas.py --run

# Phase 2: aggregate + plot from saved hits
python 26_finngen_phewas.py --plot

# Both phases
python 26_finngen_phewas.py --run --plot

# Override defaults
python 26_finngen_phewas.py --run --workers 32 --pthresh 5e-8
"""

import argparse
import os
import subprocess
import tempfile
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests  # used only for manifest download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.join(SCRIPT_DIR, '..')

DEFAULT_MANIFEST  = '/tmp/finngen_r10_manifest.tsv'
DEFAULT_LOCI      = os.path.join(BASE, 'results', 'cross_region', 'aggregate_loci.csv')
DEFAULT_OUTDIR    = os.path.join(BASE, 'results', 'finngen_phewas')
DEFAULT_FIGDIR    = os.path.join(BASE, 'figures', 'finngen_phewas')
MANIFEST_URL      = ('https://storage.googleapis.com/finngen-public-data-r10/'
                     'summary_stats/R10_manifest.tsv')

# ── Category mapping: FinnGen category → simplified label ────────────────────
CATEGORY_MAP = {
    'V Mental and behavioural disorders (F5_)':                          'Psychiatric',
    'Psychiatric endpoints from Katri Räikkönen':                        'Psychiatric',
    'VI Diseases of the nervous system (G6_)':                           'Neurological',
    'Neurological endpoints':                                            'Neurological',
    'Comorbidities of Neurological endpoints':                           'Neurological',
    'IX Diseases of the circulatory system (I9_)':                       'Cardiovascular',
    'Cardiometabolic endpoints':                                         'Cardiovascular',
    'IV Endocrine, nutritional and metabolic diseases (E4_)':            'Metabolic/Endocrine',
    'Diabetes endpoints':                                                'Metabolic/Endocrine',
    'Comorbidities of Diabetes':                                         'Metabolic/Endocrine',
    'VII Diseases of the eye and adnexa (H7_)':                         'Eye/Vision',
    'X Diseases of the respiratory system (J10_)':                       'Respiratory',
    'Asthma and related endpoints':                                      'Respiratory',
    'Interstitial lung disease endpoints':                               'Respiratory',
    'Comorbidities of Asthma':                                           'Respiratory',
    'COPD and related endpoints':                                        'Respiratory',
    'Comorbidities of COPD':                                             'Respiratory',
    'Comorbidities of Interstitial lung disease endpoints':              'Respiratory',
    'XIII Diseases of the musculoskeletal system and connective tissue (M13_)': 'Musculoskeletal',
    'Rheuma endpoints':                                                  'Musculoskeletal',
    'Diseases marked as autimmune origin':                               'Autoimmune',
    'XI Diseases of the digestive system (K11_)':                        'Gastrointestinal',
    'Gastrointestinal endpoints':                                        'Gastrointestinal',
    'Comorbidities of Gastrointestinal endpoints':                       'Gastrointestinal',
    'II Neoplasms from hospital discharges (CD2_)':                      'Cancer',
    'II Neoplasms, from cancer register (ICD-O-3)':                      'Cancer',
    'XII Diseases of the skin and subcutaneous tissue (L12_)':           'Dermatological',
    'XIV Diseases of the genitourinary system (N14_)':                   'Genitourinary',
    'VIII Diseases of the ear and mastoid process (H8_)':               'Ear/Hearing',
    'I Certain infectious and parasitic diseases (AB1_)':                'Infectious',
    'XVII Congenital malformations, deformations and chromosomal abnormalities (Q17)': 'Congenital',
    'XV Pregnancy, childbirth and the puerperium (O15_)':                'Reproductive',
    'XVI Certain conditions originating in the perinatal period (P16_)': 'Reproductive',
    'III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism (D3_)': 'Haematological',
    'Alcohol related diseases':                                          'Substance/Behavioural',
    'XIX Injury, poisoning and certain other consequences of external causes (ST19_)': 'Injury/Poisoning',
    'Dental endpoints':                                                  'Dental',
    'XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R18_)': 'Other',
    'XXII Codes for special purposes (U22_)':                            'Other',
    'Miscellaneous, not yet classified endpoints':                       'Other',
    'Other, not yet classified endpoints (same as #MISC)':               'Other',
    'Common endpoint':                                                   'Other',
    'XXI Factors influencing health status and contact with health services (Z21_)': 'Other',
    'Drug purchase endpoints':                                           'EXCLUDE',
    'Quantitative endpoints':                                            'EXCLUDE',
}

# Ordered categories for plotting (most biologically relevant first)
CATEGORY_ORDER = [
    'Psychiatric', 'Neurological', 'Cardiovascular', 'Metabolic/Endocrine',
    'Eye/Vision', 'Musculoskeletal', 'Autoimmune', 'Gastrointestinal',
    'Cancer', 'Haematological', 'Respiratory', 'Genitourinary',
    'Dermatological', 'Infectious', 'Ear/Hearing', 'Reproductive',
    'Congenital', 'Substance/Behavioural', 'Injury/Poisoning', 'Dental', 'Other',
]

CATEGORY_COLORS = {
    'Psychiatric':          '#E63946',
    'Neurological':         '#F4A261',
    'Cardiovascular':       '#E76F51',
    'Metabolic/Endocrine':  '#2A9D8F',
    'Eye/Vision':           '#264653',
    'Musculoskeletal':      '#8338EC',
    'Autoimmune':           '#FB8500',
    'Gastrointestinal':     '#6A994E',
    'Cancer':               '#BC4749',
    'Haematological':       '#A8DADC',
    'Respiratory':          '#457B9D',
    'Genitourinary':        '#C77DFF',
    'Dermatological':       '#FFBE0B',
    'Infectious':           '#B5838D',
    'Ear/Hearing':          '#6D6875',
    'Reproductive':         '#F72585',
    'Congenital':           '#90E0EF',
    'Substance/Behavioural':'#D62828',
    'Injury/Poisoning':     '#CCCCCC',
    'Dental':               '#888888',
    'Other':                '#AAAAAA',
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── Manifest loading ──────────────────────────────────────────────────────────
def load_manifest(manifest_path):
    # type: (str) -> pd.DataFrame
    """Load and filter FinnGen R10 manifest to disease endpoints."""
    if not os.path.exists(manifest_path):
        log.info('Downloading FinnGen R10 manifest...')
        r = requests.get(MANIFEST_URL, timeout=60)
        r.raise_for_status()
        with open(manifest_path, 'wb') as fh:
            fh.write(r.content)
    m = pd.read_csv(manifest_path, sep='\t')
    m['simplified_category'] = m['category'].map(CATEGORY_MAP).fillna('Other')
    m = m[m['simplified_category'] != 'EXCLUDE'].copy()
    log.info('Manifest: %d disease endpoints after filtering', len(m))
    return m


# ── SNP lookup ────────────────────────────────────────────────────────────────
def load_snps(loci_path):
    # type: (str) -> Tuple[Set[str], pd.DataFrame]
    """Return set of lead SNP rsIDs and the full loci DataFrame."""
    loci = pd.read_csv(loci_path)
    snp_set = set(loci['best_lead_snp'].dropna().tolist())
    return snp_set, loci


# ── Per-endpoint streaming query ──────────────────────────────────────────────
def query_endpoint(row, snp_patterns_file, pthresh):
    # type: (pd.Series, str, float) -> List[dict]
    """
    Download one FinnGen endpoint gz file via wget, decompress with zcat,
    filter matching SNP rows with grep -F (Aho-Corasick, one pass).
    No Python GIL bottleneck — all heavy work done in shell subprocesses.
    snp_patterns_file: path to a file with one rsID per line (grep -Ff input).
    """
    url       = row['path_https']
    phenocode = row['phenocode']
    phenotype = row['phenotype']
    category  = row['simplified_category']
    hits = []  # type: List[dict]
    try:
        # Pipeline: wget -q -O - URL | zcat | grep -Ff patterns
        cmd = (
            'wget -q --timeout=180 --tries=2 -O - "{url}" '
            '| zcat | grep -Ff "{pf}"'
        ).format(url=url, pf=snp_patterns_file)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, timeout=300
        )
        if result.returncode not in (0, 1):  # 1 = grep found nothing
            log.warning('wget/grep failed %s (rc=%d): %s',
                        phenocode, result.returncode,
                        result.stderr.decode(errors='replace')[:200])
            return hits
        lines = result.stdout.decode('utf-8', errors='replace').strip().split('\n')
        for line in lines:
            if not line:
                continue
            parts = line.split('\t')
            # FinnGen columns: chrom pos ref alt rsids nearest_genes pval mlogp beta sebeta ...
            if len(parts) < 9:
                continue
            rsid_field = parts[4]
            rsids_in_line = set(rsid_field.split(','))
            # re-load snp_set from file for matching (subprocess workaround)
            # Instead, filter by pval first then return all fields
            try:
                pval = float(parts[6])
            except (ValueError, IndexError):
                continue
            if pval >= pthresh:
                continue
            try:
                beta = float(parts[8])
                se   = float(parts[9])
            except (ValueError, IndexError):
                beta = se = float('nan')
            hits.append({
                'phenocode': phenocode,
                'phenotype': phenotype,
                'category':  category,
                'rsid_field': rsid_field,
                'pval':      pval,
                'beta':      beta,
                'se':        se,
            })
    except subprocess.TimeoutExpired:
        log.warning('Timeout %s', phenocode)
    except Exception as exc:
        log.warning('Failed %s: %s', phenocode, exc)
    return hits


# ── Main run phase ────────────────────────────────────────────────────────────
def run_phewas(manifest, snp_set, loci, outdir, workers, pthresh):
    # type: (pd.DataFrame, Set[str], pd.DataFrame, str, int, float) -> pd.DataFrame
    os.makedirs(outdir, exist_ok=True)
    hits_path       = os.path.join(outdir, 'finngen_hits.csv')
    checkpoint_path = os.path.join(outdir, 'completed_phenocodes.txt')

    # Write SNP patterns file for grep -Ff (one rsID per line)
    snp_patterns_file = os.path.join(outdir, 'snp_patterns.txt')
    with open(snp_patterns_file, 'w') as fh:
        fh.write('\n'.join(sorted(snp_set)) + '\n')
    log.info('SNP patterns file: %s (%d rsIDs)', snp_patterns_file, len(snp_set))

    # Load completed phenocodes (checkpoint/resume)
    completed = set()  # type: Set[str]
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as fh:
            completed = set(l.strip() for l in fh if l.strip())
        log.info('Resuming: %d endpoints already completed', len(completed))

    todo = manifest[~manifest['phenocode'].isin(completed)].copy()
    log.info('Querying %d endpoints with %d workers at p < %g',
             len(todo), workers, pthresh)

    all_hits = []  # type: List[dict]
    if completed and os.path.exists(hits_path):
        all_hits = pd.read_csv(hits_path).to_dict('records')

    n_done = len(completed)
    total  = len(manifest)

    with open(checkpoint_path, 'a') as ckpt_fh:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(query_endpoint, row, snp_patterns_file, pthresh): row['phenocode']
                for _, row in todo.iterrows()
            }
            for future in as_completed(futures):
                phenocode = futures[future]
                try:
                    hits = future.result()
                    all_hits.extend(hits)
                except Exception as exc:
                    log.warning('Error %s: %s', phenocode, exc)
                n_done += 1
                ckpt_fh.write(phenocode + '\n')
                ckpt_fh.flush()
                if n_done % 50 == 0 or n_done == total:
                    log.info('Progress: %d/%d endpoints | hits so far: %d',
                             n_done, total, len(all_hits))
                    if all_hits:
                        pd.DataFrame(all_hits).to_csv(hits_path, index=False)

    hits_df = pd.DataFrame(all_hits) if all_hits else pd.DataFrame(
        columns=['phenocode','phenotype','category','rsid_field','pval','beta','se'])
    hits_df.to_csv(hits_path, index=False)
    log.info('Run complete: %d hits in %d endpoints', len(hits_df),
             hits_df['phenocode'].nunique() if len(hits_df) else 0)
    return hits_df


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate(hits_df, loci, outdir):
    # type: (pd.DataFrame, pd.DataFrame, str) -> Tuple[pd.DataFrame, pd.DataFrame]
    """Aggregate hits by category and by region × category."""
    if hits_df.empty:
        log.warning('No hits to aggregate')
        return pd.DataFrame(), pd.DataFrame()

    # Resolve matched SNP from rsid_field (may be comma-separated)
    snp_set = set(loci['best_lead_snp'].tolist())
    def resolve_snp(rsid_field):
        # type: (str) -> str
        for r in rsid_field.split(','):
            if r in snp_set:
                return r
        return rsid_field.split(',')[0]
    hits_df = hits_df.copy()
    hits_df['snp'] = hits_df['rsid_field'].apply(resolve_snp)

    # Merge with loci to get region info per SNP
    snp2regions = loci.set_index('best_lead_snp')['regions'].to_dict()
    snp2alid    = loci.set_index('best_lead_snp')['al_id'].to_dict()
    hits_df['regions'] = hits_df['snp'].map(snp2regions).fillna('')
    hits_df['al_id']   = hits_df['snp'].map(snp2alid).fillna('')

    # ── Summary by category ───────────────────────────────────────────────
    cat_rows = []
    for cat in CATEGORY_ORDER:
        sub = hits_df[hits_df['category'] == cat]
        if sub.empty:
            continue
        cat_rows.append({
            'category':    cat,
            'n_hits':      len(sub),
            'n_loci':      sub['snp'].nunique(),
            'n_endpoints': sub['phenocode'].nunique(),
        })
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(os.path.join(outdir, 'finngen_summary_category.csv'), index=False)

    # ── Summary by region × category ─────────────────────────────────────
    region_rows = []
    # explode multi-region loci
    hits_exp = hits_df.copy()
    hits_exp['region_list'] = hits_exp['regions'].str.split(';')
    hits_exp = hits_exp.explode('region_list')
    hits_exp = hits_exp.rename(columns={'region_list': 'region'})
    hits_exp['region'] = hits_exp['region'].str.strip()

    for (reg, cat), grp in hits_exp.groupby(['region', 'category']):
        region_rows.append({
            'region':      reg,
            'category':    cat,
            'n_loci':      grp['snp'].nunique(),
            'n_hits':      len(grp),
        })
    reg_df = pd.DataFrame(region_rows)
    reg_df.to_csv(os.path.join(outdir, 'finngen_summary_region.csv'), index=False)

    log.info('Aggregated: %d categories, %d region×category pairs',
             len(cat_df), len(reg_df))
    return cat_df, reg_df


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(cat_df, reg_df, figdir):
    # type: (pd.DataFrame, pd.DataFrame, str) -> None
    import sys
    sys.path.insert(0, SCRIPT_DIR)
    from fig_style import apply_mpl_style, save_mpl, MM
    apply_mpl_style()
    os.makedirs(figdir, exist_ok=True)

    if cat_df.empty:
        log.warning('No data to plot')
        return

    # ── Fig A: bar chart of n_loci per category ───────────────────────────
    cat_plot = cat_df[cat_df['category'].isin(CATEGORY_ORDER)].copy()
    cat_plot['cat_order'] = cat_plot['category'].map(
        {c: i for i, c in enumerate(CATEGORY_ORDER)})
    cat_plot = cat_plot.sort_values('cat_order')

    fig, ax = plt.subplots(figsize=(MM(130), MM(80)))
    colors = [CATEGORY_COLORS.get(c, '#AAAAAA') for c in cat_plot['category']]
    bars = ax.barh(cat_plot['category'], cat_plot['n_loci'],
                   color=colors, height=0.7, edgecolor='white', linewidth=0.4)
    # annotate bars with n_loci
    for bar, val in zip(bars, cat_plot['n_loci']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va='center', ha='left', fontsize=7)
    ax.set_xlabel('Number of JAGWAS loci with FinnGen association (p < 5×10⁻⁸)', fontsize=8)
    ax.set_title('FinnGen R10 PheWAS — JAGWAS loci by disease category', fontsize=9)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    save_mpl(fig, os.path.join(figdir, 'finngen_category_bar'))
    plt.close(fig)

    # ── Fig B: bubble plot region × category ─────────────────────────────
    if reg_df.empty:
        return

    region_order = [
        'L. Thalamus', 'R. Thalamus',
        'L. Caudate', 'R. Caudate',
        'L. Putamen', 'R. Putamen',
        'L. Pallidum', 'R. Pallidum',
        'L. Hippocampus', 'R. Hippocampus',
        'L. Amygdala', 'R. Amygdala',
        'L. Accumbens', 'R. Accumbens',
        'Brain Stem / 4th V.', 'CSF',
    ]
    cat_bub_order = [c for c in CATEGORY_ORDER if c in reg_df['category'].unique()]
    reg_df_filt = reg_df[reg_df['region'].isin(region_order) &
                          reg_df['category'].isin(cat_bub_order)].copy()

    pivot = reg_df_filt.pivot_table(
        index='region', columns='category', values='n_loci', fill_value=0)
    # reindex to canonical order
    pivot = pivot.reindex(
        index=region_order,
        columns=[c for c in cat_bub_order if c in pivot.columns],
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(MM(160), MM(90)))
    for ri, reg in enumerate(pivot.index):
        for ci, cat in enumerate(pivot.columns):
            val = pivot.loc[reg, cat]
            if val == 0:
                continue
            color = CATEGORY_COLORS.get(cat, '#AAAAAA')
            size  = val * 80
            ax.scatter(ci, ri, s=size, c=color, alpha=0.8,
                       edgecolors='white', linewidths=0.5, zorder=3)
            ax.text(ci, ri, str(int(val)), ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold', zorder=4)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title('FinnGen R10 PheWAS — JAGWAS loci by region and disease category',
                 fontsize=9)
    ax.set_xlim(-0.7, len(pivot.columns) - 0.3)
    ax.set_ylim(-0.7, len(pivot.index) - 0.3)
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_mpl(fig, os.path.join(figdir, 'finngen_bubble'))
    plt.close(fig)
    log.info('Figures saved to %s', figdir)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',  default=DEFAULT_MANIFEST,
                   help='Path to FinnGen R10 manifest TSV (downloaded if absent)')
    p.add_argument('--loci',      default=DEFAULT_LOCI,
                   help='aggregate_loci.csv with best_lead_snp column')
    p.add_argument('--outdir',    default=DEFAULT_OUTDIR,
                   help='Output directory for CSV results')
    p.add_argument('--figdir',    default=DEFAULT_FIGDIR,
                   help='Output directory for figures')
    p.add_argument('--workers',   type=int, default=32,
                   help='Parallel download workers (default: 32)')
    p.add_argument('--pthresh',   type=float, default=5e-8,
                   help='p-value threshold for hits (default: 5e-8)')
    p.add_argument('--focus',     action='store_true',
                   help='Restrict to key categories: Psychiatric/Neurological/'
                        'Cardiovascular/Eye/Metabolic (~500 endpoints, faster)')
    p.add_argument('--run',       action='store_true',
                   help='Run the PheWAS query phase')
    p.add_argument('--plot',      action='store_true',
                   help='Run the aggregation + plotting phase')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.run and not args.plot:
        print('Specify --run, --plot, or both. Use --help for details.')
        sys.exit(1)

    manifest = load_manifest(args.manifest)
    if args.focus:
        focus_cats = {
            'Psychiatric', 'Neurological', 'Cardiovascular',
            'Eye/Vision', 'Metabolic/Endocrine',
        }
        manifest = manifest[manifest['simplified_category'].isin(focus_cats)].copy()
        log.info('Focus mode: %d endpoints in key categories', len(manifest))
    snp_set, loci = load_snps(args.loci)
    log.info('Loaded %d lead SNPs from %d loci', len(snp_set), len(loci))

    hits_path = os.path.join(args.outdir, 'finngen_hits.csv')

    if args.run:
        hits_df = run_phewas(manifest, snp_set, loci,
                             args.outdir, args.workers, args.pthresh)
    else:
        if not os.path.exists(hits_path):
            log.error('No hits file found at %s — run with --run first', hits_path)
            sys.exit(1)
        hits_df = pd.read_csv(hits_path)
        log.info('Loaded %d existing hits from %s', len(hits_df), hits_path)

    if args.plot:
        cat_df, reg_df = aggregate(hits_df, loci, args.outdir)
        plot(cat_df, reg_df, args.figdir)


if __name__ == '__main__':
    main()
