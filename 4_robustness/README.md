# Stage 4 — robustness

Two questions that the discovery pipeline (stages 1–3) cannot answer about itself.

**Does the result depend on who was measured?** Discovery ran in a genetically homogeneous
sample. `generalizability/` recomputes the reported loci in a cohort drawn from outside that
ancestry cluster, and asks both whether the association reappears and whether the allelic
effects point the same way.

**Does the result depend on the particular representation that was learned?** The phenotype
here is not a measurement but the output of a self-supervised encoder, and a different
training run learns a different 128-dimensional basis for the same anatomy.
`reproducibility/` retrains the encoder from independent initialisations, carries each run
through the whole pipeline, and compares the representations, the association statistics and
the called loci.

Both read their paths from `config/paths.yaml` like the rest of the repository. Scripts that
need a resource with no natural home in that file carry an inline `<EXTERNAL: ...>` marker
naming what to supply.

---

## generalizability/

Run in this order. Each step writes what the next one reads.

| script | what it does |
|---|---|
| `generate_nonwb_masks.py` | builds region masks for the new cohort by applying each participant's existing linear registration, rather than re-running segmentation |
| `extract_bre_nonwb.py` | extracts the 128-dimensional embedding per region, using the SAME encoder checkpoint as discovery — a retrained encoder would confound the question |
| `extract_dosages_nonwb.py` | pulls genotype dosages at the reported loci |
| `build_replication_targets.py` | assembles the target locus list and the discovery effect directions to test against |
| `genomewide_nonwb_jagwas.py` | recomputes the joint statistic genome-wide in the new cohort |
| `summarise_genomewide_nonwb.py` | per-region replication rates and the genomic inflation factor |
| `top_hits_nonwb.py` | per-locus replication and agreement of effect direction |
| `build_nonwb_figures_tables.py` | the figures and tables as they appear in the paper |

Two points that matter for interpretation, both of which the code makes explicit:

- The joint statistic tests **magnitude, not direction**, so replication of the statistic alone
  does not show the effect points the same way. `top_hits_nonwb.py` tests direction separately,
  and that test is the more informative one at this sample size because it does not depend on a
  significance threshold.
- The cohort is defined by **exclusion** from one ancestry cluster, not by sampling a named
  ancestry. It is therefore a test of whether the loci survive ancestral heterogeneity, and not
  a test of transferability to any particular ancestry.

## reproducibility/

| script | what it does |
|---|---|
| `run_gwas_seed.py`, `run_gwas_region.py` | drive the full association pipeline for one retrained encoder |
| `extract_bre.py`, `make_phenotypes.py` | embedding extraction and phenotype construction per run |
| `concordance_embeddings.py`, `summarise_concordance.py` | agreement between the representations two runs learn |
| `cka_significance.py` | centered kernel alignment against a cross-region null |
| `ceiling_cka.py` | the upper reference: alignment between two checkpoints of the SAME run one epoch apart, which bounds how much agreement is even attainable |
| `verify_cka.py` | independent recomputation of the alignment statistic |
| `epoch_alignment.py` | alignment across training epochs between runs |
| `effective_rank.py`, `dim_correspondence.py` | how many embedding dimensions carry signal, and whether dimensions match across runs |
| `val_curve.py` | held-out loss per epoch, read from the partition that contributes no gradient |
| `plot_cka_significance.py`, `plot_embedding_concordance.py`, `plot_epoch_structure.py`, `plot_stability.py`, `plot_loss_curves.py` | the figures |

Interpreting the output:

- **Individual embedding dimensions are not comparable across runs.** Nothing anchors dimension
  7 of one run to dimension 7 of another; only the subspace is comparable, which is why the
  comparison is by alignment rather than by correlating matching dimensions.
- **Lead variants are less stable than loci.** Independent runs frequently select a different
  lead variant inside the same linkage-disequilibrium block, so any tabulation keyed on lead
  variant understates agreement. Compare at the locus level.
- **Two retrained encoders are two draws, not a distribution.** The agreement values bound
  seed-to-seed variance only coarsely; a confidence interval over seeds needs more runs.
