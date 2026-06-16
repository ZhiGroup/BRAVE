"""
13_loci_replication.py
Loci replication validation — discovery (N≈22,878) vs replication (N≈12,359).

Two levels of validation:

(A) Per-region: for each of 16 JAGWAS regions, look up each discovery lead SNP
    (FUMA leadSNPs.txt) in the matched replication summary statistics.
    Report replication at three thresholds:
      - GW-significant : p < 5e-8
      - Bonferroni     : p < 0.05 / n_lead_SNPs_in_that_region
      - Nominal        : p < 0.05

(B) Aggregate (AL-) level: for each of 276 AL- loci, look up the best_lead_snp
    in the replication file of every region that originally discovered it.
    The locus replicates if its minimum replication p reaches the threshold in
    ANY of those regions (at least-one-region criterion).
    Thresholds: GW (5e-8) / Bonferroni (0.05/276) / Nominal (0.05).

Figures:
  (A1) Per-region replication rate barplot (nominal and Bonferroni)
  (A2) Discovery vs replication -log10(p) scatter (pooled, all lead SNPs)
  (B1) Aggregate-loci replication summary (donut)
  (B2) Replication rate by n_regions (region-sharing level)

Outputs:
  results/replication/per_region_replication.csv
  results/replication/aggregate_replication.csv
  figures/replication/per_region_replication_rates.pdf/.png
  figures/replication/discovery_vs_replication_scatter.pdf/.png
  figures/replication/aggregate_replication_summary.pdf/.png
  figures/replication/replication_by_nregions.pdf/.png

Supports all 3 paper claims:
  - Claim 2: high replication rate demonstrates robust multivariate signal
  - Claim 1: novel loci (vs ENIGMA) replicate at similar rates to known loci
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL, FS_TITLE

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
FUMA_DIR  = Path(cfg.postgwas.fuma_dir)
REP_DIR   = Path("<EXTERNAL: replication-cohort FUMA output dir>")

RES_DIR   = Path(__file__).parents[1] / "results"  / "replication"
FIG_DIR   = Path(__file__).parents[1] / "figures" / "replication"
CR_RES    = Path(__file__).parents[1] / "results"  / "cross_region"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

AGG_LOCI  = CR_RES / "aggregate_loci.csv"
NOVELTY   = CR_RES / "jagwas_novelty_classification.csv"

# ── REGION MAPPING: display name → FUMA folder ───────────────────────────────
DISP_TO_FOLDER = {
    'Brain Stem / 4th V.': 'Brain_Stem_or_4th_Ventricle',
    'CSF':                  'CSF',
    'L. Accumbens':         'Left_Accumbens-area',
    'L. Amygdala':          'Left_Amygdala',
    'L. Caudate':           'Left_Caudate',
    'L. Hippocampus':       'Left_Hippocampus',
    'L. Pallidum':          'Left_Pallidum',
    'L. Putamen':           'Left_Putamen',
    'L. Thalamus':          'Left_Thalamus_Proper',
    'R. Accumbens':         'Right_Accumbens-area',
    'R. Amygdala':          'Right_Amygdala',
    'R. Caudate':           'Right_Caudate',
    'R. Hippocampus':       'Right_Hippocampus',
    'R. Pallidum':          'Right_Pallidum',
    'R. Putamen':           'Right_Putamen',
    'R. Thalamus':          'Right_Thalamus-Proper',
}

# Anatomical sort order for display
REGION_ORDER = [
    'L. Accumbens', 'R. Accumbens',
    'L. Amygdala',  'R. Amygdala',
    'L. Caudate',   'R. Caudate',
    'L. Hippocampus','R. Hippocampus',
    'L. Pallidum',  'R. Pallidum',
    'L. Putamen',   'R. Putamen',
    'L. Thalamus',  'R. Thalamus',
    'Brain Stem / 4th V.', 'CSF',
]

# Colours
COL_GW    = '#CB181D'   # red — genome-wide in replication
COL_BONF  = '#FB6A4A'   # orange-red — Bonferroni
COL_NOM   = '#FDAE6B'   # orange — nominal
COL_FAIL  = '#CCCCCC'   # grey — not replicated

# Replication thresholds
P_GW      = 5e-8
P_NOM     = 0.05

# ── STEP 1: LOAD DISCOVERY LEAD SNPS PER REGION ──────────────────────────────
print("Loading discovery lead SNPs ...")
discovery = {}   # folder → DataFrame with rsID, genomicLocus, discovery_p
for disp, folder in DISP_TO_FOLDER.items():
    f = FUMA_DIR / folder / 'leadSNPs.txt'
    if not f.exists():
        print(f"  WARNING: {f} not found")
        continue
    df = pd.read_csv(f, sep='\t')
    df = df.rename(columns={'rsID': 'rsid', 'p': 'discovery_p', 'GenomicLocus': 'locus_id'})
    df['display'] = disp
    df['folder']  = folder
    discovery[folder] = df
    print(f"  {disp}: {len(df)} lead SNPs")

# ── STEP 2: COLLECT ALL SNPS NEEDED ──────────────────────────────────────────
# Per-region: lead SNPs from each region (look up in same-region replication)
# Aggregate: best_lead_snp from aggregate_loci.csv (look up in each relevant region)
agg_df    = pd.read_csv(AGG_LOCI)
all_snps_needed = set()
for _, row in agg_df.iterrows():
    all_snps_needed.add(row['best_lead_snp'])
for df in discovery.values():
    all_snps_needed.update(df['rsid'].tolist())
print(f"\nTotal unique SNPs to look up: {len(all_snps_needed)}")

# ── STEP 3: LOAD REPLICATION STATS — ONE REGION AT A TIME ────────────────────
print("\nLoading replication summary stats ...")
# rep_lookup[folder][rsid] = replication_p
rep_lookup = {}

for disp, folder in DISP_TO_FOLDER.items():
    rep_file = REP_DIR / folder / f"{folder}_JAGWAS_results.txt"
    if not rep_file.exists():
        print(f"  WARNING: {rep_file} not found")
        continue
    print(f"  Loading {folder} replication ...", end=' ', flush=True)
    rep = pd.read_csv(rep_file, sep='\t', usecols=['SNP', 'P'],
                      dtype={'SNP': str, 'P': float})
    # Filter to only needed SNPs for speed
    rep = rep[rep['SNP'].isin(all_snps_needed)]
    rep_lookup[folder] = rep.set_index('SNP')['P'].to_dict()
    n_rep_sample = pd.read_csv(rep_file, sep='\t', nrows=1)['N'].values[0]
    print(f"{len(rep_lookup[folder])} matched / N_rep={int(n_rep_sample)}")

# ── STEP 4: PER-REGION VALIDATION ────────────────────────────────────────────
print("\nRunning per-region validation ...")
per_region_rows = []

all_scatter = []  # for the pooled scatter plot

for disp in REGION_ORDER:
    folder = DISP_TO_FOLDER[disp]
    if folder not in discovery or folder not in rep_lookup:
        continue
    disc = discovery[folder].copy()
    rep  = rep_lookup[folder]

    n_loci   = len(disc)
    p_bonf   = 0.05 / n_loci

    disc['rep_p']  = disc['rsid'].map(rep)
    disc['found']  = disc['rep_p'].notna()
    disc['rep_p_filled'] = disc['rep_p'].fillna(1.0)  # missing → not replicated

    n_found  = disc['found'].sum()
    n_gw     = (disc['rep_p_filled'] < P_GW).sum()
    n_bonf   = (disc['rep_p_filled'] < p_bonf).sum()
    n_nom    = (disc['rep_p_filled'] < P_NOM).sum()

    per_region_rows.append(dict(
        display=disp, folder=folder,
        n_lead_snps=n_loci, n_found_in_rep=n_found,
        p_bonf_threshold=p_bonf,
        n_gw=n_gw, n_bonf=n_bonf, n_nom=n_nom,
        n_fail=n_loci - n_nom,
        pct_gw=n_gw/n_loci*100, pct_bonf=n_bonf/n_loci*100,
        pct_nom=n_nom/n_loci*100,
    ))
    print(f"  {disp:<22}: {n_loci:3d} loci | "
          f"GW={n_gw} ({n_gw/n_loci*100:.0f}%) | "
          f"Bonf={n_bonf} ({n_bonf/n_loci*100:.0f}%) | "
          f"Nom={n_nom} ({n_nom/n_loci*100:.0f}%)")

    # Collect for scatter
    for _, row in disc.iterrows():
        if pd.notna(row['rep_p']):
            all_scatter.append(dict(
                display=disp,
                disc_p=row['discovery_p'],
                rep_p=row['rep_p'],
                rep_cat=('GW' if row['rep_p'] < P_GW else
                         'Bonf' if row['rep_p'] < p_bonf else
                         'Nom' if row['rep_p'] < P_NOM else 'Fail'),
            ))

pr_df = pd.DataFrame(per_region_rows)
pr_df.to_csv(RES_DIR / 'per_region_replication.csv', index=False)
print(f"\nSaved: {RES_DIR}/per_region_replication.csv")

# ── STEP 5: AGGREGATE (AL-) VALIDATION ───────────────────────────────────────
print("\nRunning aggregate (AL-) loci validation ...")

# Bonferroni for 276 loci
N_AGG  = len(agg_df)
P_BONF_AGG = 0.05 / N_AGG

agg_rows = []
for _, row in agg_df.iterrows():
    snp     = row['best_lead_snp']
    regions = [r.strip() for r in str(row['regions']).split(';')]

    # Look up this SNP in each relevant region's replication file
    rep_ps = []
    for reg_disp in regions:
        folder = DISP_TO_FOLDER.get(reg_disp)
        if folder and folder in rep_lookup:
            p = rep_lookup[folder].get(snp, np.nan)
            if pd.notna(p):
                rep_ps.append(p)

    min_rep_p = min(rep_ps) if rep_ps else np.nan
    n_rep_files_hit = sum(1 for p in rep_ps if p < P_NOM)

    if pd.isna(min_rep_p):
        cat = 'Not found'
    elif min_rep_p < P_GW:
        cat = 'GW'
    elif min_rep_p < P_BONF_AGG:
        cat = 'Bonferroni'
    elif min_rep_p < P_NOM:
        cat = 'Nominal'
    else:
        cat = 'Not replicated'

    agg_rows.append(dict(
        al_id=row['al_id'], chr=row['chr'],
        n_regions=row['n_regions'],
        best_lead_snp=snp, discovery_p=row['best_p'],
        min_rep_p=min_rep_p, n_rep_files_nominal=n_rep_files_hit,
        rep_category=cat,
        novel_vs_enigma=row.get('novel_vs_all', np.nan),
    ))

# Merge novelty classification
novelty_df = pd.read_csv(NOVELTY)[['al_id', 'novel_vs_all', 'n_regions']]
agg_rep_df = pd.DataFrame(agg_rows).merge(
    novelty_df[['al_id', 'novel_vs_all']], on='al_id', how='left')
agg_rep_df.to_csv(RES_DIR / 'aggregate_replication.csv', index=False)
print(f"Saved: {RES_DIR}/aggregate_replication.csv")

# Summary
cats = ['GW', 'Bonferroni', 'Nominal', 'Not replicated', 'Not found']
cat_counts = agg_rep_df['rep_category'].value_counts().reindex(cats, fill_value=0)
n_rep_any = cat_counts[['GW', 'Bonferroni', 'Nominal']].sum()
print(f"\n{'='*55}")
print(f"  AL- loci total:                  {N_AGG}")
print(f"  Replicated (nominal, p<0.05):    {n_rep_any}  ({n_rep_any/N_AGG*100:.1f}%)")
print(f"  Replicated (Bonferroni p<{P_BONF_AGG:.1e}): "
      f"{cat_counts['GW'] + cat_counts['Bonferroni']}  "
      f"({(cat_counts['GW']+cat_counts['Bonferroni'])/N_AGG*100:.1f}%)")
print(f"  GW-significant in replication:   {cat_counts['GW']}  ({cat_counts['GW']/N_AGG*100:.1f}%)")
for c in cats:
    print(f"    {c:<25} {cat_counts[c]:4d}  ({cat_counts[c]/N_AGG*100:.1f}%)")
print(f"{'='*55}")

# ── STEP 6: REPLICATION RATE OF NOVEL vs KNOWN LOCI ──────────────────────────
novel_mask = agg_rep_df['novel_vs_all'].fillna(False).astype(bool)
known_mask = ~novel_mask
for label, mask in [('Novel vs ENIGMA', novel_mask), ('ENIGMA-known', known_mask)]:
    sub = agg_rep_df[mask]
    n_rep = (sub['rep_category'].isin(['GW', 'Bonferroni', 'Nominal'])).sum()
    print(f"  {label}: {n_rep}/{len(sub)} ({n_rep/len(sub)*100:.1f}%) replicated nominally")

# ── FIGURE A1: PER-REGION REPLICATION RATES ───────────────────────────────────
print("\nGenerating Figure A1: per-region replication rates ...")

pr_plot = pr_df.set_index('display').reindex(REGION_ORDER).dropna(subset=['n_lead_snps'])
y = np.arange(len(pr_plot))

fig, axes = plt.subplots(1, 2, figsize=(MM(170), MM(105)))

for ax_idx, (metric_col, metric_label, ref_col, title) in enumerate([
    ('n_nom',  'Nominal (p<0.05)',         'n_lead_snps', 'Nominal replication (p < 0.05)'),
    ('n_bonf', f'Bonferroni (p<0.05/n)',   'n_lead_snps', 'Bonferroni replication'),
]):
    ax = axes[ax_idx]
    vals  = pr_plot[metric_col].values
    total = pr_plot['n_lead_snps'].values
    pct   = vals / total

    ax.barh(y, pct, color=COL_NOM if 'Nominal' in title else COL_BONF,
            height=0.7, edgecolor='none')
    for i, (v, t, p) in enumerate(zip(vals, total, pct)):
        ax.text(p + 0.01, i, f'{int(v)}/{int(t)}', va='center',
                fontsize=FS_TICK - 1, color='#333333')
    ax.set_yticks(y)
    ax.set_yticklabels(pr_plot.index.tolist() if ax_idx == 0 else [''] * len(y),
                       fontsize=FS_TICK)
    ax.set_xlabel('Fraction replicated', fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_LABEL, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()
    ax.axvline(0.5, color='#888888', lw=0.7, ls='--')

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'per_region_replication_rates')
plt.close()
print("Saved: figures/replication/per_region_replication_rates.pdf/.png")

# ── FIGURE A2: DISCOVERY vs REPLICATION SCATTER ───────────────────────────────
print("Generating Figure A2: discovery vs replication scatter ...")

scat_df = pd.DataFrame(all_scatter)
scat_df['disc_neglog'] = -np.log10(scat_df['disc_p'].clip(1e-300))
scat_df['rep_neglog']  = -np.log10(scat_df['rep_p'].clip(1e-300))

cat_colors = {'GW': COL_GW, 'Bonf': COL_BONF, 'Nom': COL_NOM, 'Fail': COL_FAIL}
cat_zorder = {'GW': 4, 'Bonf': 3, 'Nom': 2, 'Fail': 1}
cat_labels = {'GW': f'GW (p<5e-8)', 'Bonf': 'Bonferroni', 'Nom': 'Nominal (p<0.05)', 'Fail': 'Not replicated'}

fig, ax = plt.subplots(figsize=(MM(100), MM(95)))

for cat in ['Fail', 'Nom', 'Bonf', 'GW']:
    sub = scat_df[scat_df['rep_cat'] == cat]
    ax.scatter(sub['disc_neglog'], sub['rep_neglog'],
               color=cat_colors[cat], s=8, alpha=0.7, linewidths=0,
               zorder=cat_zorder[cat], label=f"{cat_labels[cat]} (n={len(sub)})")

# Spearman correlation
rho, pval = stats.spearmanr(scat_df['disc_neglog'], scat_df['rep_neglog'])
ax.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}', transform=ax.transAxes,
        fontsize=FS_TICK, va='top')

ax.axhline(-np.log10(P_NOM), color='#888888', lw=0.8, ls='--',
           label=f'Nominal threshold')
ax.set_xlabel('Discovery −log₁₀(p)', fontsize=FS_LABEL)
ax.set_ylabel('Replication −log₁₀(p)', fontsize=FS_LABEL)
ax.set_title('Discovery vs replication p-values\n(lead SNPs, all 16 regions)',
             fontsize=FS_LABEL, fontweight='bold')
ax.legend(fontsize=FS_TICK - 0.5, frameon=False, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'discovery_vs_replication_scatter')
plt.close()
print("Saved: figures/replication/discovery_vs_replication_scatter.pdf/.png")

# ── FIGURE B1: AGGREGATE REPLICATION SUMMARY ─────────────────────────────────
print("Generating Figure B1: aggregate replication summary ...")

# Combine GW into Bonferroni tier for cleaner display (3 tiers)
def tier(cat):
    if cat in ('GW', 'Bonferroni'):
        return 'Bonferroni'
    return cat
agg_rep_df['tier'] = agg_rep_df['rep_category'].apply(tier)

tier_order  = ['GW', 'Bonferroni', 'Nominal', 'Not replicated', 'Not found']
tier_colors = [COL_GW, COL_BONF, COL_NOM, COL_FAIL, '#EEEEEE']
tier_disp   = [f'GW (p<5e-8)  n={cat_counts["GW"]}',
               f'Bonferroni (p<{P_BONF_AGG:.1e})  n={cat_counts["Bonferroni"]}',
               f'Nominal (p<0.05)  n={cat_counts["Nominal"]}',
               f'Not replicated  n={cat_counts["Not replicated"]}',
               f'Not found  n={cat_counts["Not found"]}']

sizes  = [cat_counts[t] for t in tier_order]
colors = tier_colors

fig, axes = plt.subplots(1, 2, figsize=(MM(160), MM(80)))

# Left: Donut for all 276 AL- loci
ax0 = axes[0]
wedges, _ = ax0.pie(
    sizes, colors=colors, startangle=90,
    wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 1.5},
)
ax0.text(0, 0.08, f'{N_AGG}', ha='center', va='center',
         fontsize=FS_TITLE+1, fontweight='bold', color='#333333')
ax0.text(0, -0.15, 'AL- loci', ha='center', va='center',
         fontsize=FS_TICK-1, color='#666666')
ax0.set_title('Aggregate loci replication\n(best lead SNP, ≥1 region)',
              fontsize=FS_LABEL, fontweight='bold')
ax0.legend(wedges, tier_disp, loc='lower center', fontsize=FS_TICK - 0.5,
           frameon=False, bbox_to_anchor=(0.5, -0.25), ncol=1)

# Right: Novel vs known replication comparison
ax1 = axes[1]
labels_nk = ['Novel\n(vs ENIGMA)', 'ENIGMA-\nknown']
grp_counts = []
for mask in [novel_mask, known_mask]:
    sub = agg_rep_df[mask]
    grp_counts.append([
        (sub['rep_category'] == 'GW').sum(),
        (sub['rep_category'] == 'Bonferroni').sum(),
        (sub['rep_category'] == 'Nominal').sum(),
        (sub['rep_category'] == 'Not replicated').sum() + (sub['rep_category'] == 'Not found').sum(),
    ])

y2 = np.arange(2)
lefts2 = np.zeros(2)
for i, (color, label) in enumerate(zip(
        [COL_GW, COL_BONF, COL_NOM, COL_FAIL],
        ['GW', 'Bonferroni', 'Nominal', 'Not replicated'])):
    vals2 = np.array([gc[i] for gc in grp_counts])
    totals2 = np.array([novel_mask.sum(), known_mask.sum()])
    fracs2 = vals2 / totals2
    ax1.barh(y2, fracs2, left=lefts2, color=color, height=0.5,
             edgecolor='none', label=label)
    for j, (f, v) in enumerate(zip(fracs2, vals2)):
        if v > 0:
            ax1.text(lefts2[j] + f/2, j, str(v), ha='center', va='center',
                     fontsize=FS_TICK-1, fontweight='bold', color='white')
    lefts2 += fracs2

ax1.set_yticks(y2)
ax1.set_yticklabels(labels_nk, fontsize=FS_TICK)
ax1.set_xlabel('Fraction of loci', fontsize=FS_LABEL)
ax1.set_title('Replication: novel vs\nENIGMA-known loci', fontsize=FS_LABEL, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'aggregate_replication_summary')
plt.close()
print("Saved: figures/replication/aggregate_replication_summary.pdf/.png")

# ── FIGURE B2: REPLICATION BY SHARING LEVEL ──────────────────────────────────
print("Generating Figure B2: replication by region-sharing level ...")

nreg_rep = agg_rep_df.groupby('n_regions').apply(
    lambda g: pd.Series({
        'n_total': len(g),
        'n_rep_nom': (g['rep_category'].isin(['GW','Bonferroni','Nominal'])).sum(),
        'n_rep_bonf': (g['rep_category'].isin(['GW','Bonferroni'])).sum(),
    })
).reset_index()
nreg_rep['pct_nom']  = nreg_rep['n_rep_nom']  / nreg_rep['n_total'] * 100
nreg_rep['pct_bonf'] = nreg_rep['n_rep_bonf'] / nreg_rep['n_total'] * 100

fig, ax = plt.subplots(figsize=(MM(110), MM(75)))
x = nreg_rep['n_regions'].values
ax.bar(x - 0.2, nreg_rep['pct_nom'].values,  width=0.35, color=COL_NOM,  label='Nominal (p<0.05)', edgecolor='none')
ax.bar(x + 0.2, nreg_rep['pct_bonf'].values, width=0.35, color=COL_BONF, label='Bonferroni', edgecolor='none')
ax.axhline(n_rep_any/N_AGG*100, color='#555555', ls='--', lw=1,
           label=f'Overall nominal {n_rep_any/N_AGG*100:.0f}%')
for i, (xi, n) in enumerate(zip(x, nreg_rep['n_total'])):
    ax.text(xi, max(nreg_rep['pct_nom'].iloc[i], nreg_rep['pct_bonf'].iloc[i]) + 2,
            f'n={n}', ha='center', fontsize=FS_TICK-2, color='#444444')
ax.set_xlabel('Number of regions discovering the locus', fontsize=FS_LABEL)
ax.set_ylabel('% Replicated', fontsize=FS_LABEL)
ax.set_title('Replication rate by locus region-sharing level',
             fontsize=FS_LABEL, fontweight='bold')
ax.set_ylim(0, 115)
ax.set_xticks(x)
ax.legend(fontsize=FS_TICK, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_mpl(fig, FIG_DIR / 'replication_by_nregions')
plt.close()
print("Saved: figures/replication/replication_by_nregions.pdf/.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════╗
║  SCRIPT 13 SUMMARY — Replication Validation            ║
╠══════════════════════════════════════════════════════════╣
║  Discovery N ≈ 22,878   |   Replication N ≈ 12,359     ║
╠══════════════════════════════════════════════════════════╣
║  PER-REGION (lead SNPs):                                ║
║  Pooled lead SNPs:          {len(scat_df):5d}                      ║
║  Spearman ρ (disc vs rep):  {rho:.3f}                      ║
╠══════════════════════════════════════════════════════════╣
║  AGGREGATE (AL- loci):                                  ║
║  Total:                     {N_AGG:5d}                      ║
║  Replicated (nominal):      {n_rep_any:5d}  ({n_rep_any/N_AGG*100:.1f}%)            ║
║  Replicated (Bonferroni):   {cat_counts['GW']+cat_counts['Bonferroni']:5d}  ({(cat_counts['GW']+cat_counts['Bonferroni'])/N_AGG*100:.1f}%)            ║
║  GW-significant in rep:     {cat_counts['GW']:5d}  ({cat_counts['GW']/N_AGG*100:.1f}%)            ║
╚══════════════════════════════════════════════════════════╝
""")
