# Stage 3 — post-GWAS analysis

This is where most of the paper's numbers and figures come from. Each script
reads FUMA outputs and/or munged summary statistics, writes a CSV under
`results/<domain>/`, and a figure under `figures/<domain>/`. Scripts are
numbered in dependency order.

```bash
cd 3_postgwas
cp ../config/paths.yaml ../config/paths.local.yaml   # then edit it
bash run_all.sh
```

Outputs land in `results/` and `figures/` (both git-ignored). Shared figure
formatting (Arial, Nature-Comms sizing) lives in `scripts/fig_style.py`.

## What the scripts do

**Per region (01–04)** — loci/gene summaries (Table 1), eQTL tissue breakdown,
MAGMA tissue enrichment, cell-type enrichment.

**Across regions (05–10)** — cross-trait GWAS-Catalog Sankey, aggregated
non-redundant loci, region-overlap heatmap, UpSet plots, structural-vs-neuronal
cell types, eQTL tissue matrix.

**Novelty & replication (11–13)** — credible-interval mapping, novelty against
ENIGMA volume and a shape GWAS, replication in the held-out fold.

**Heritability & genetic correlation (14–19)** — LDSC h² and top-dim selection,
genetic correlation with ENIGMA/shape, with brain disorders, MAGMA GO-BP
enrichment, matched-region cell types, and the cardiac-MRI / retinal-OCT
cross-modal correlations (18/18b, 19/19b).

**PGS (20, 20b)** — polygenic-score validation in the replication fold.

**Genome-wide PGS — SBayesRC (36, 37, 38, 48)** — a single-region pilot that
confirms the stack end to end, the production sweep across all 16 regions, the
cross-region R² matrix that tests region specificity at the polygenic level, and
the composite figure comparing that matrix with the lead-SNP baseline from 20 and
20b. Needs SBayesRC and its LD reference; the sweep runs for hours.

**Effect-size architecture — GENESIS (39, 44, 40, 41)** — in that order, and the
first two are R, not Python. `39_genesis_polygenicity_pilot.R` fits one trait per
invocation and takes its arguments on the command line, so `run_all.sh` does not
call it: run it once per trait first. `44_dump_genesis_params.R` then collects
every fit into two tidy CSVs, and 40 and 41 render the pooled and per-region
panels.

**Locus plots, PheWAS, demographics, PoPS, colocalisation, PHESANT (21–44)** —
the focused locus figures, FinnGen/OpenGWAS PheWAS, cohort flow and demographics,
PoPS gene prioritisation, colocalisation, phenome-wide PHESANT scans, and the
assembled main-figure panels (`Fig1_framework.py`, `fig5_biology_composite.py`,
`fig6_crossmodal_composite.py`, etc.).

Scripts numbered 19+ take `argparse` options (`--help` lists them); earlier ones
read their paths from the central config. A few one-off reference files are
marked inline as `<EXTERNAL: ...>` — grep for that string and set each.

## A note on the numbering

Ten numbers are used twice, because separate groups of scripts were developed in
parallel: 16, 21, 36, 37, 38, 40, 41, 42, 43 and 44 each name two scripts. For
example `36_age_sex_scatter_supp.py` and `36_sbayesrc_pgs_pilot.py` both exist.
The number is a rough development order, not a unique identifier — `run_all.sh`
is the authoritative run order, and it names every script in full.
