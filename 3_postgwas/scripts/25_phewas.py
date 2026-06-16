#!/usr/bin/env python
"""Script 25: PheWAS from existing GWAS Catalog cross-trait data.

Re-analyses the 88,787 GWAS Catalog associations from Script 16
(cross_trait_all_categories.csv) to produce a disease-level PheWAS
comparable to Xue et al. 2026 (Nat Commun 17:526, brainstem paper Fig 6).

Workflow (single run — no API required)
---------------------------------------
    python 25_phewas.py [--run] [--plot]

Outputs
-------
results/phewas/
    phewas_disease.csv          non-brain-imaging disease associations
    phewas_disease_summary.csv  per-category: n_diseases, n_loci, n_regions
    phewas_region_category.csv  region × category count matrix

figures/phewas/
    phewas_compound.pdf/.png    compound diagram (category × n_diseases,
                                 bar = n_associations) — like Xue 2026 Fig 6
    phewas_bubble.pdf/.png      bubble: BRE region × disease category

Disease categories
------------------
Derived from Organ_Category in existing data, refined by Trait keyword mapping
to distinguish disease associations from quantitative traits (e.g., height, BMI).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE         = os.path.join(SCRIPT_DIR, '..')
RESULTS      = os.path.join(BASE, 'results', 'phewas')
FIGURES      = os.path.join(BASE, 'figures', 'phewas')
CROSS_TRAIT  = os.path.join(BASE, 'results', 'cross_trait',
                             'cross_trait_all_categories.csv')

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from fig_style import apply_mpl_style, save_mpl, MM

# ── Disease keyword filter ────────────────────────────────────────────────────
# Keep traits that are disease/disorder entities; exclude pure quantitative
# traits (height, BMI, blood pressure measurement, etc.) unless they are
# clinically defined conditions.

DISEASE_KEYWORDS = [
    'disease', 'disorder', 'syndrome', 'schizophrenia', 'depression',
    'bipolar', 'autism', 'adhd', 'attention deficit', 'alzheimer',
    'dementia', 'parkinson', 'als', 'amyotrophic', 'epilepsy', 'seizure',
    'migraine', 'stroke', 'infarct', 'atrial fibrillation', 'heart failure',
    'hypertension', 'diabetes', 'obesity', 'asthma', 'copd', 'cancer',
    'carcinoma', 'tumor', 'tumour', 'lymphoma', 'leukemia', 'leukaemia',
    'glaucoma', 'macular', 'retinal', 'cataract', 'age-related',
    'inflammatory', 'autoimmune', 'arthritis', 'lupus', 'multiple sclerosis',
    'hypothyroidism', 'hyperthyroidism', 'chronic kidney', 'renal failure',
    'osteoporosis', 'anxiety', 'anorexia', 'ptsd', 'post-traumatic',
    'insomnia', 'sleep apn', 'narcolepsy', 'neuroticism',
    'major depressive', 'stress-related', 'psychosis',
]

EXCLUDE_KEYWORDS = [
    'vertex-wise', 'sulcal', 'cortical thickness', 'surface area',
    'white matter', 'grey matter', 'gray matter', 'brain volume',
    'fractional anisotropy', 'mean diffusivity', 'resting-state',
    'mri', 'imaging', 'electroencephalogram', 'eeg',
]

# ── Category mapping (refined from Organ_Category) ───────────────────────────
CATEGORY_REMAP = {
    'Psychiatric':          'Psychiatric',
    'Neurological':         'Neurological',
    'Cardiovascular':       'Cardiovascular',
    'Metabolic':            'Metabolic/Endocrine',
    'Pulmonary/Lung':       'Respiratory',
    'Eye/Vision':           'Eye/Vision',
    'Renal/Kidney':         'Renal',
    'Hepatic/Liver':        'Hepatic',
    'Cognitive/Education':  'Cognitive',
}

CATEGORY_ORDER = [
    'Psychiatric', 'Neurological', 'Cardiovascular',
    'Metabolic/Endocrine', 'Respiratory', 'Eye/Vision',
    'Renal', 'Hepatic', 'Cognitive', 'Other disease',
]

CATEGORY_COLORS = {
    'Psychiatric':        '#9DC3E6',
    'Neurological':       '#4472C4',
    'Cardiovascular':     '#FF0000',
    'Metabolic/Endocrine':'#FFC000',
    'Respiratory':        '#92D050',
    'Eye/Vision':         '#2E75B6',
    'Renal':              '#70AD47',
    'Hepatic':            '#ED7D31',
    'Cognitive':          '#7030A0',
    'Other disease':      '#AAAAAA',
}

REGION_DISPLAY = {
    'Left_Thalamus_Proper':         'L. Thalamus',
    'Right_Thalamus_Proper':        'R. Thalamus',
    'Left_Hippocampus':             'L. Hippocampus',
    'Right_Hippocampus':            'R. Hippocampus',
    'Left_Amygdala':                'L. Amygdala',
    'Right_Amygdala':               'R. Amygdala',
    'Left_Caudate':                 'L. Caudate',
    'Right_Caudate':                'R. Caudate',
    'Left_Putamen':                 'L. Putamen',
    'Right_Putamen':                'R. Putamen',
    'Left_Pallidum':                'L. Pallidum',
    'Right_Pallidum':               'R. Pallidum',
    'Left_Accumbens-area':          'L. Accumbens',
    'Right_Accumbens-area':         'R. Accumbens',
    'Brain_Stem_or_4th_Ventricle':  'BrainStem',
    'CSF':                          'CSF',
}

REGION_ORDER = [
    'L. Thalamus', 'R. Thalamus',
    'L. Hippocampus', 'R. Hippocampus',
    'L. Amygdala', 'R. Amygdala',
    'L. Caudate', 'R. Caudate',
    'L. Putamen', 'R. Putamen',
    'L. Pallidum', 'R. Pallidum',
    'L. Accumbens', 'R. Accumbens',
    'BrainStem', 'CSF',
]


def is_disease(trait: str) -> bool:
    t = trait.lower()
    for excl in EXCLUDE_KEYWORDS:
        if excl in t:
            return False
    for kw in DISEASE_KEYWORDS:
        if kw in t:
            return True
    return False


def remap_category(organ_cat: str, trait: str) -> str:
    cat = CATEGORY_REMAP.get(organ_cat, None)
    if cat:
        return cat
    # Try to assign from trait keyword
    t = trait.lower()
    if any(k in t for k in ['schizophrenia', 'bipolar', 'depression', 'autism',
                             'adhd', 'anxiety', 'anorexia', 'ptsd', 'psychosis',
                             'neuroticism', 'insomnia', 'stress-related']):
        return 'Psychiatric'
    if any(k in t for k in ['alzheimer', 'parkinson', 'epilepsy', 'dementia',
                             'stroke', 'als', 'migraine', 'multiple sclerosis']):
        return 'Neurological'
    if any(k in t for k in ['atrial', 'heart', 'cardiac', 'coronary',
                             'hypertension', 'blood pressure']):
        return 'Cardiovascular'
    if any(k in t for k in ['diabetes', 'obesity', 'thyroid', 'metabolic']):
        return 'Metabolic/Endocrine'
    if any(k in t for k in ['glaucoma', 'macular', 'retinal', 'vision', 'eye']):
        return 'Eye/Vision'
    return 'Other disease'


# ── Step 1: Filter and classify disease associations ──────────────────────────

def run_phewas(args: argparse.Namespace) -> None:
    path = getattr(args, '_cross_trait_path', CROSS_TRAIT)
    print(f'Loading cross-trait data from {path} ...')
    df = pd.read_csv(path)
    print(f'  {len(df):,} total GWAS Catalog associations')

    # Drop brain/imaging traits
    non_brain = df[df['Organ_Category'] != 'Brain/Imaging'].copy()
    print(f'  {len(non_brain):,} non-brain-imaging associations')

    # Filter to disease traits
    non_brain['is_disease'] = non_brain['Trait'].apply(
        lambda x: is_disease(str(x)))
    disease_df = non_brain[non_brain['is_disease']].copy()
    print(f'  {len(disease_df):,} disease-specific associations')

    # Remap categories
    disease_df['category'] = disease_df.apply(
        lambda row: remap_category(str(row['Organ_Category']),
                                   str(row['Trait'])), axis=1)

    # Clean region names
    disease_df['region_display'] = disease_df['Our_Region'].map(
        REGION_DISPLAY).fillna(disease_df['Our_Region'])

    # Save disease associations
    out = os.path.join(RESULTS, 'phewas_disease.csv')
    disease_df.to_csv(out, index=False)
    print(f'  Saved → {out}')

    # Summary by category
    trait_col = 'Trait'
    locus_col = 'GenomicLocus_x'
    summary = (disease_df.groupby('category')
               .agg(n_associations=(trait_col, 'count'),
                    n_diseases=(trait_col, 'nunique'),
                    n_loci=(locus_col, 'nunique'),
                    n_regions=('region_display', 'nunique'))
               .reset_index()
               .sort_values('n_associations', ascending=False))
    summ_out = os.path.join(RESULTS, 'phewas_disease_summary.csv')
    summary.to_csv(summ_out, index=False)
    print(f'\nDisease PheWAS summary:\n{summary.to_string(index=False)}')
    print(f'\nTotal unique disease traits: {disease_df[trait_col].nunique()}')
    print(f'Total unique loci with ≥1 disease hit: {disease_df[locus_col].nunique()}')

    # Region × category matrix
    reg_cat = (disease_df.groupby(['region_display', 'category'])
               .agg(n_diseases=(trait_col, 'nunique'),
                    n_loci=(locus_col, 'nunique'))
               .reset_index())
    rc_out = os.path.join(RESULTS, 'phewas_region_category.csv')
    reg_cat.to_csv(rc_out, index=False)
    print(f'  Region×category matrix → {rc_out}')


# ── Step 2: Plot ──────────────────────────────────────────────────────────────

def plot_phewas(args: argparse.Namespace) -> None:
    apply_mpl_style()

    summ_path    = os.path.join(RESULTS, 'phewas_disease_summary.csv')
    disease_path = os.path.join(RESULTS, 'phewas_disease.csv')
    rc_path      = os.path.join(RESULTS, 'phewas_region_category.csv')

    if not os.path.exists(summ_path):
        print(f'  {summ_path} not found — run --run first.')
        return

    summ     = pd.read_csv(summ_path)
    disease  = pd.read_csv(disease_path)
    reg_cat  = pd.read_csv(rc_path)

    # Sort categories by total n_associations descending (matches right bar)
    summ = summ.sort_values('n_associations', ascending=False).reset_index(drop=True)
    cats_ordered = summ['category'].tolist()

    # Per-disease association counts (for left scatter dots)
    trait_assoc = (disease.groupby(['Trait', 'category'])
                   .size().reset_index(name='n_assoc'))

    # ── Figure A: compound diagram (Xue 2026 Fig 6 style) ────────────────────
    # Left: each individual disease trait as a dot (x=category, y=n_assoc)
    # Right: total n_associations per category as horizontal bar
    # Both panels use the same metric → consistent ordering

    fig, axes = plt.subplots(1, 2, figsize=(MM(180), MM(90)),
                             gridspec_kw={'width_ratios': [3, 1]})

    ax = axes[0]
    cat_x = {cat: i for i, cat in enumerate(cats_ordered)}

    for _, row in trait_assoc.iterrows():
        cat = row['category']
        if cat not in cat_x:
            continue
        xi  = cat_x[cat]
        col = CATEGORY_COLORS.get(cat, '#888888')
        # Jitter x slightly so overlapping dots are visible
        jitter = np.random.uniform(-0.25, 0.25)
        ax.scatter(xi + jitter, row['n_assoc'],
                   s=12, color=col, alpha=0.7,
                   edgecolors='none', zorder=3)

    ax.set_xticks(range(len(cats_ordered)))
    ax.set_xticklabels(cats_ordered, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Associations per disease trait', fontsize=8)
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title('PheWAS: 276 JAGWAS loci × GWAS Catalog disease traits', fontsize=9)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.set_xlim(-0.5, len(cats_ordered) - 0.5)

    # Right: bar chart — total associations per category, same order as left
    ax2 = axes[1]
    colors = [CATEGORY_COLORS.get(c, '#888888') for c in cats_ordered]
    ax2.barh(range(len(summ)), summ['n_associations'].values,
             color=colors, edgecolor='white', linewidth=0.4)
    ax2.set_yticks(range(len(summ)))
    ax2.set_yticklabels(cats_ordered, fontsize=7)
    ax2.set_xlabel('Total associations', fontsize=8)
    ax2.tick_params(labelsize=7)
    # Categories ordered by n_associations descending → top of bar = highest
    ax2.invert_yaxis()

    plt.tight_layout()
    save_mpl(fig, os.path.join(FIGURES, 'phewas_compound'))
    print('  Saved phewas_compound.pdf/.png')

    # ── Figure B: bubble (region × category) ─────────────────────────────────
    regs_present = [r for r in REGION_ORDER if r in reg_cat['region_display'].values]
    cats_present2 = [c for c in CATEGORY_ORDER if c in reg_cat['category'].values]

    fig2, ax3 = plt.subplots(figsize=(MM(160), MM(110)))

    max_n = reg_cat['n_diseases'].max()
    for _, row in reg_cat.iterrows():
        xi = cats_present2.index(row['category']) if row['category'] in cats_present2 else -1
        yi = regs_present.index(row['region_display']) if row['region_display'] in regs_present else -1
        if xi < 0 or yi < 0:
            continue
        col = CATEGORY_COLORS.get(row['category'], '#888888')
        size = (row['n_diseases'] / max_n) * 300 + 10
        ax3.scatter(xi, yi, s=size, color=col,
                    alpha=0.85, edgecolors='white', linewidths=0.5, zorder=3)

    ax3.set_xticks(range(len(cats_present2)))
    ax3.set_xticklabels(cats_present2, rotation=45, ha='right', fontsize=7)
    ax3.set_yticks(range(len(regs_present)))
    ax3.set_yticklabels(regs_present, fontsize=7)
    ax3.set_xlabel('Disease category', fontsize=8)
    ax3.set_ylabel('BRE region', fontsize=8)
    ax3.set_title('Disease associations by region (bubble size = n diseases)',
                  fontsize=9)
    ax3.grid(linestyle='--', linewidth=0.3, alpha=0.4)

    # Legend for bubble size
    for n_ex, label in [(5, '5'), (20, '20'), (50, '50')]:
        s_ex = (n_ex / max_n) * 300 + 10
        ax3.scatter([], [], s=s_ex, color='grey', alpha=0.7,
                    label=f'{label} diseases', edgecolors='white')
    ax3.legend(title='N diseases', fontsize=7, title_fontsize=7,
               loc='lower right', framealpha=0.8)

    plt.tight_layout()
    save_mpl(fig2, os.path.join(FIGURES, 'phewas_bubble'))
    print('  Saved phewas_bubble.pdf/.png')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Script 25: PheWAS from existing GWAS Catalog cross-trait data')
    ap.add_argument('--run',  action='store_true',
                    help='Classify disease associations and write summary CSVs')
    ap.add_argument('--plot', action='store_true',
                    help='Generate compound and bubble figures')
    ap.add_argument('--cross-trait', default=CROSS_TRAIT,
                    help=f'Path to cross_trait_all_categories.csv '
                         f'(default: {CROSS_TRAIT})')
    args = ap.parse_args()

    cross_trait_path = args.cross_trait

    if not any([args.run, args.plot]):
        ap.print_help()
        sys.exit(0)

    args._cross_trait_path = cross_trait_path
    if args.run:
        run_phewas(args)
    if args.plot:
        plot_phewas(args)


if __name__ == '__main__':
    main()
