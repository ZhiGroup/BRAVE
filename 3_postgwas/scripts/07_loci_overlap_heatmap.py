"""
07_loci_overlap_heatmap.py
16×16 loci overlap heatmap between all pairs of brain regions.

For each pair of regions (A, B):
  overlap = number of AL- loci shared between A and B
  Jaccard = |A ∩ B| / |A ∪ B|

Two heatmaps:
  1. Raw overlap counts (lower triangle)
  2. Jaccard similarity (upper triangle + diagonal = 1)
  Combined into a single figure.

Also: dendrogram-ordered clustermap showing hierarchical grouping of regions
      by their loci-sharing profile.

Outputs:
  results/cross_region/loci_overlap_matrix.csv   — pairwise raw counts
  results/cross_region/jaccard_matrix.csv        — pairwise Jaccard
  figures/cross_region/loci_overlap_heatmap.pdf/.png
  figures/cross_region/loci_overlap_clustermap.pdf/.png

Supports paper Claims 2 & 3: quantifies region-specificity and shared loci
  structure; clustermap shows anatomically coherent groupings.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import scipy.cluster.hierarchy as sch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL, FS_TITLE

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
RES_DIR = Path(__file__).parents[1] / "results" / "cross_region"
FIG_DIR = Path(__file__).parents[1] / "figures" / "cross_region"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD BINARY MATRIX ────────────────────────────────────────────────────────
matrix_df = pd.read_csv(RES_DIR / 'loci_region_matrix.csv', index_col='al_id')
regions   = matrix_df.columns.tolist()
M         = matrix_df.values   # shape: (n_loci, 16)
print(f"Loaded binary matrix: {M.shape[0]} AL- loci × {M.shape[1]} regions")

# ── COMPUTE PAIRWISE OVERLAP AND JACCARD ──────────────────────────────────────
n = len(regions)
overlap_mat = np.zeros((n, n), dtype=int)
jaccard_mat = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        a = M[:, i].astype(bool)
        b = M[:, j].astype(bool)
        inter = (a & b).sum()
        union = (a | b).sum()
        overlap_mat[i, j] = inter
        jaccard_mat[i, j] = inter / union if union > 0 else 0.0

overlap_df = pd.DataFrame(overlap_mat, index=regions, columns=regions)
jaccard_df = pd.DataFrame(jaccard_mat, index=regions, columns=regions)

overlap_df.to_csv(RES_DIR / 'loci_overlap_matrix.csv')
jaccard_df.to_csv(RES_DIR / 'jaccard_matrix.csv')
print(f"Saved: {RES_DIR}/loci_overlap_matrix.csv")
print(f"Saved: {RES_DIR}/jaccard_matrix.csv")

# ── SHORT LABELS ──────────────────────────────────────────────────────────────
def short_label(r):
    r = r.replace('Brain Stem / 4th V.', 'BrStem')
    r = r.replace('L. ', 'L.').replace('R. ', 'R.')
    r = r.replace(' ', '\n')
    return r

labels = [short_label(r) for r in regions]

# ── FIGURE 1: RAW OVERLAP (lower triangle) + JACCARD (upper triangle) ─────────
print("\nGenerating overlap heatmap ...")

# Build combined matrix: lower = raw counts, upper = Jaccard
combined = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i >= j:
            combined[i, j] = overlap_mat[i, j]   # lower + diagonal
        else:
            combined[i, j] = jaccard_mat[i, j]   # upper

fig, ax = plt.subplots(figsize=(MM(160), MM(145)))

# Use two colormaps blended visually with masking
# Lower: Blues for counts (0 to max overlap)
# Upper: Oranges for Jaccard (0 to 1)
# Strategy: plot full combined matrix with a diverging palette, then overlay text

vmax_count = np.max(overlap_mat[np.tril_indices(n, k=-1)])  # off-diagonal lower max

# Plot as single heatmap with Blues (rescaled so upper Jaccard ∈ [0, vmax_count])
# Scale Jaccard to count range for display uniformity
combined_scaled = combined.copy()
mask_upper = np.triu(np.ones((n, n), bool), k=1)
combined_scaled[mask_upper] = jaccard_mat[mask_upper] * vmax_count

im = ax.imshow(combined_scaled, cmap='YlOrRd', aspect='auto',
               vmin=0, vmax=vmax_count)

# Annotate cells
for i in range(n):
    for j in range(n):
        if i == j:
            txt = f"{overlap_mat[i,j]}"
            ax.text(j, i, txt, ha='center', va='center', fontsize=5,
                    color='white', fontweight='bold')
        elif i > j:   # lower: raw count
            val = overlap_mat[i, j]
            color = 'white' if val > vmax_count * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=4.5,
                    color=color)
        else:          # upper: Jaccard
            val = jaccard_mat[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=4,
                    color=color)

ax.set_xticks(range(n))
ax.set_xticklabels(labels, fontsize=5.5, rotation=90)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=5.5)

# Dividing line between lower/upper triangles
for k in range(n):
    ax.plot([k - 0.5, k + 0.5], [k - 0.5, k + 0.5], color='white', lw=0.5)

# Colourbar
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Shared loci (lower) / Jaccard × max (upper)', fontsize=FS_TICK)
cbar.ax.tick_params(labelsize=FS_TICK - 1)

ax.set_title('Pairwise loci overlap — lower: count, upper: Jaccard similarity',
             fontsize=FS_LABEL)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'loci_overlap_heatmap')
plt.close()
print("Saved: figures/cross_region/loci_overlap_heatmap.pdf/.png")

# ── FIGURE 2: CLUSTERMAP (Jaccard distance dendrogram) ────────────────────────
print("Generating clustermap ...")

# Hierarchical clustering on Jaccard distance
dist = 1 - jaccard_mat
np.fill_diagonal(dist, 0)
linkage = sch.linkage(dist[np.tril_indices(n, k=-1)], method='average')
dendro  = sch.dendrogram(linkage, no_plot=True)
order   = dendro['leaves']

jac_ordered  = jaccard_mat[np.ix_(order, order)]
labels_ord   = [labels[i] for i in order]

fig, ax = plt.subplots(figsize=(MM(155), MM(140)))
cmap = LinearSegmentedColormap.from_list('jac', ['#FFFFFF', '#2166AC'])
im = ax.imshow(jac_ordered, cmap=cmap, aspect='auto', vmin=0, vmax=1)

for i in range(n):
    for j in range(n):
        val = jac_ordered[i, j]
        col = 'white' if val > 0.55 else 'black'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                fontsize=4, color=col)

ax.set_xticks(range(n))
ax.set_xticklabels(labels_ord, fontsize=5.5, rotation=90)
ax.set_yticks(range(n))
ax.set_yticklabels(labels_ord, fontsize=5.5)
ax.set_title('Jaccard loci overlap — hierarchically clustered', fontsize=FS_LABEL)

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Jaccard similarity', fontsize=FS_TICK)
cbar.ax.tick_params(labelsize=FS_TICK - 1)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'loci_overlap_clustermap')
plt.close()
print("Saved: figures/cross_region/loci_overlap_clustermap.pdf/.png")

# ── PRINT TOP OVERLAPPING PAIRS ───────────────────────────────────────────────
print("\n=== Top 10 overlapping region pairs (by Jaccard) ===")
pairs = []
for i in range(n):
    for j in range(i+1, n):
        pairs.append((regions[i], regions[j], overlap_mat[i,j], jaccard_mat[i,j]))
pairs_df = pd.DataFrame(pairs, columns=['Region_A', 'Region_B', 'n_shared', 'Jaccard'])
pairs_df = pairs_df.sort_values('Jaccard', ascending=False)
print(pairs_df.head(10).to_string(index=False))

print("\n=== Lowest overlap region pairs (by Jaccard, top 5) ===")
print(pairs_df.tail(5).to_string(index=False))
