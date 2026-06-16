#!/bin/bash
# Reproduce the post-GWAS analyses in dependency order.
#
#   cd 3_postgwas && bash run_all.sh
#
# Assumes config/paths.local.yaml points at your FUMA outputs and munged
# summary statistics. Results go to results/, figures to figures/.
# Some steps need external resources (LDSC, PLINK2, a BGEN) configured first;
# those are noted inline. Stop on the first error so a missing input is obvious.

set -e
PYTHON=${PYTHON:-python}
S=scripts

run() { echo ">> $1"; $PYTHON "$S/$1"; }

# Per region
run 01_per_region_summary.py
run 02_eqtl_tissue_breakdown.py
run 03_magma_tissue_enrichment.py
run 04_celltype_enrichment.py          # needs FUMA GENE2FUNC results

# Across regions
run 05_cross_trait_sankey.py
run 06_aggregate_loci.py
run 06b_aggregate_loci_sensitivity.py
run 07_loci_overlap_heatmap.py
run 08_upset_plots.py
run 09_structural_vs_neuronal.py
run 09b_structural_neuronal_loo.py
run 10_eqtl_tissue_matrix.py

# Novelty & replication
run 11_ci_mapping_analysis.py
run 12_loci_novelty.py
run 12b_loci_novelty_3way.py
run 12c_gwas_catalog_novelty.py
run 12c_novelty_window_sensitivity.py
run 13_loci_replication.py

# Heritability & genetic correlation  (LDSC must be configured)
run 14_extract_h2_select_dims.py
run 15_ldsc_genetic_correlation.py
run 16_disease_gc.py
run 16_magma_gobp_enrichment.py
run 17_matched_region_celltype.py
run 18_heart_gc.py
run 18b_heart_gc_focused_figure.py
run 19_oct_gc.py
run 19b_oct_gc_focused_figure.py

# PGS  (needs PLINK2 + BGEN; first run builds a pgen subset, ~15-30 min)
run 20_bre_pgs.py
run 20b_pgs_extended.py

# Locus plots, PheWAS, demographics, PoPS, coloc, PHESANT
for s in 21_ablation_loci_barplot 21_oct_locus_plot 22_heart_locus_plot \
         23_syngo_summary 24_canonical_h2_thalamus 25_phewas 26_finngen_phewas \
         27_cohort_flow_diagram 28_compute_demographics 29_training_loss_curve \
         30_pops_per_region 31_focused_coloc 32_coloc_visualisations \
         33_pops_visualisations 34_phesant_per_region 35_phesant_visualisations \
         36_age_sex_scatter_supp 37_assemble_fig2 38_age_sex_predict_with_icv_cov \
         40_h2_violin 41_replot_enigma_gc_heatmap 42_composite_pgs_aggregation \
         42_fig4a_replication_bars 43_render_composite_pgs_sensitivity_fig \
         43_univariate_baseline_novelty 44_render_h2_vs_phesant_hits; do
  run "$s.py"
done

# Assembled main-figure panels (run last; they read figures produced above)
run Fig1_framework.py
run fig5_biology_composite.py
run fig6_crossmodal_composite.py

echo "All post-GWAS scripts finished."
