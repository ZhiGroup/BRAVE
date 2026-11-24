# Local clumping validated against the published FUMA loci

Reproduction of the 16 per-region genomic risk locus lists from
`JAGWAS/FUMA/<region>/` using `fuma_clump.py` in place of FUMA v1.5.2, against a
locally rebuilt 1000 Genomes Phase 3 EUR panel. Run 2026-08-05.

## Headline

| | FUMA v1.5.2 | local |
|---|---|---|
| Genomic risk loci, 16 regions | 1,034 | 1,036 |
| Lead variants shared (chr:pos) | — | **1,032 (99.8%)** |
| Regions at 100% recall | — | **14 / 16** |
| Lead rsID labels disagreeing | — | **0** |
| Unrecovered leads outside every local locus | — | **0** |

Fourteen regions reproduce exactly. The two that do not (Left_Putamen 91/92,
Right_Accumbens-area 74/75) each miss one lead, and in both cases the signal was
found and merged into a neighbouring block rather than lost — see "Attribution".

Independently confirmed out-of-sample: on a GWAS the tool had never been tested
against (seed 1, Right_Accumbens-area), submitted to the FUMA **web application**
directly, recovery is **100% in both directions** with **99.2% identical lead
SNPs**. See "Second, out-of-sample validation".

## Configuration that reproduces FUMA

Parameters verbatim from the per-region `params.config`:

```
--lead-p 3.125e-9  --gwas-p 0.05  --r2 0.6  --r2-2 0.1
--merge-dist 250000  --maf 0  --no-mhc  --clump-kb 1000
--bfile-pattern <path/to>/g1000_phase3/EUR.chr{}
--plink <path/to>/plink
```

plink 1.9 (`v1.90b6.10`) is required — the code uses 1.9-style `--clump` /
`--r2`, and plink2 cannot substitute. JAGWAS sumstats feed in unmodified:
`read_gwas` needs only `{CHR, SNP, POS, A1, A2, P}`, so the absent `BETA`/`SE`
columns do not matter, clumping being p-value based.

No `--ref-map` harmonization. It existed to reconcile rsIDs against the old
MAGMA panel; the rebuilt panel is labelled directly from the GWAS, so the two
already share identifiers.

## Second, out-of-sample validation (2026-08-09)

The validation above is against the published run — the same data the reference
panel was rebuilt and the parameters were reverse-engineered for. That leaves an
obvious objection: the tool could have been tuned, however unintentionally, to
one dataset.

This addresses it. Seed 1's Right_Accumbens-area summary statistics — a GWAS the
tool had never been tested on — were submitted to the **FUMA web application**
directly (job 761901), with settings matching `params.config` line for line
including `refSNPs = 1` and `MHCopt = annot`. Archive and extracted output:
`reproducibility/seed_1/fuma/`.

| | |
|---|---|
| FUMA web app loci | 121 |
| Local call | 123 |
| **FUMA loci recovered locally (interval overlap)** | **121/121 = 100.0%** |
| FUMA loci recovered (lead within 500 kb) | 121/121 = 100.0% |
| **FUMA loci with identical lead SNP** | **120/121 = 99.2%** |
| **Local loci present in FUMA** | **123/123 = 100.0%** |

Nothing is unmatched in either direction.

The identical-lead-SNP figure is the informative one. Against the published run,
seed 1 shares only 31.7% of lead SNPs (§ "Locus recovery" in
REPLICATION_FINDINGS §4.6) — but that compares two *different* GWAS. Here, with
the same summary statistics passed through two independent implementations, the
lead SNP agrees on 120 of 121 loci. Lead-SNP instability is therefore a property
of comparing different GWAS, not of the locus caller.

The 123-vs-121 difference is two loci the local tool splits and FUMA merges,
which is the expected direction of the single known mismatch: `refSNPs = 1` gives
FUMA wider LD blocks, and wider blocks merge more readily under the 250 kb rule.
It costs nothing here — both split loci still overlap FUMA's merged one, so
recovery is 100% in both directions.

Comparison script: `local_clumping/locus_overlap.py`; result
`reproducibility/seed_1/fuma/overlap_fuma_vs_local.csv`.

## What FUMA actually does (from its docs and source)

From the [FUMA tutorial](https://fuma.ctglab.nl/tutorial) and
[`getLD.py`](https://github.com/Kyoko-wtnb/FUMA-webapp/blob/master/storage/scripts/getLD.py).

FUMA does not run plink at analysis time; it reads pre-computed per-chromosome
tabix files built once per panel:

> "MAF and pairwise LD were computed by PLINK (`--r2 --ld-window 99999
> --ld-window-r2 0.05`) ... and SNPs with MAF=0 were excluded for each
> population."

No `--ld-window-kb` is given, so plink's **1000 kb default** applies: FUMA never
links two SNPs more than 1 Mb apart.

`getLD.py` hardcodes the MHC exactly as our copy does, and applies it by
filtering the GWAS input before clumping — which is what `--no-mhc` does:

```python
MHCstart = 29614758 # hg19
MHCend   = 33170276 # hg19
```

Locus boundaries are `min`/`max` position over LD partners; merging is
`start <= previous_end or start - previous_end < mergeDist`. Both match.

## Reference panel (rebuilt)

`build_fuma_panel.py` rebuilds the panel from the 1000G Phase 3 VCFs following
FUMA's stated recipe:

> "Multi allelic SNPs were first split into separate columns"
> "VCF files were then converted to PLINK bfile (PLINK v1.9). Any CNVs were
> removed, while any indels were kept."
> "SNPs with MAF=0 were excluded for each population."

| | variants | samples |
|---|---|---|
| FUMA 1000G Phase3 EUR | 25,063,419 | 503 |
| **rebuilt** (`ref_panels/g1000_phase3/EUR.chr{}`) | **24,131,852** | 503 |
| previous (MAGMA `g1000_eur.zip`) | 8,550,156 | 503 |

96.3% of FUMA's variant count, versus 34% before. Output is one bfile per
chromosome, which is also far cheaper to clump against than a merged file.

Three things the recipe required that were not obvious:

- **plink2 does not drop CNVs on import.** 1000G carries them as symbolic ALT
  alleles (`<CN0>`, `<INS:ME:ALU>`); they are imported verbatim and later
  collide on duplicate IDs. They are excluded explicitly from the pvar.
- **`--rm-dup` must run after `--set-missing-var-ids`.** In the split step it
  runs first and cannot see collisions among generated IDs — two distinct indels
  at one position can reduce to the same `chr:pos:ref:alt` string. plink 1.9
  aborts the entire chromosome on the first duplicate.
- **The v5b VCFs contain no rsIDs at all** — every ID field is `.`, so all IDs
  had to be synthesised. `assign_panel_rsids.py` relabels the panel from the
  GWAS by CHR:POS (8,069,128 of 24,131,852 rows). The dbSNP copy on disk is the
  common-only subset and covers just 47.9%, whereas the GWAS covers every
  variant that can ever be reported as a lead — and is the naming FUMA's own
  output uses. Multi-allelic positions keep synthetic IDs, since several panel
  rows there cannot share one rsID.

## Three defects fixed in our copy of `fuma_clump.py`

1. **`--maf 0` kills every plink call.** FUMA's `MAF = 0` means "no filter";
   plink 1.9 rejects it ("must be > 0"), and the flag cannot simply be dropped
   because plink's own default is 0.01. `_plink_maf_arg()` substitutes 1e-6.

2. **plink failures were silent.** Both plink wrappers sent stdout *and* stderr
   to `DEVNULL` and returned an empty frame when the output file was absent, so
   22 consecutive crashes were reported as "No significant SNPs found" — an
   empty but entirely plausible-looking result. `_run_plink()` now prints
   plink's error on non-zero exit. Every subsequent defect here was found
   through that message.

3. **`CLUMP_KB` was a hardcoded 50 Mb constant** with no CLI flag, so clumping
   could link SNPs 50 Mb apart where FUMA stops at 1 Mb. That merged blocks FUMA
   keeps separate across long-range LD — most visibly the 8p23.1 inversion
   (~4 Mb), and it cost 6 of 16 regions their exact agreement. Because there was
   no override, no run log recorded which window had been used.
   `--clump-kb` is now exposed **and the default changed to 1000**, so
   reproducing FUMA is the behaviour you get without asking; widening the window
   now requires an explicit flag. Drivers pass `--clump-kb 1000` anyway, so runs
   remain self-documenting.

A fourth change is efficiency only: the skip for chromosomes with no
genome-wide significant SNP was gated on `and not bfile`, disabling it whenever
a reference BED was supplied, so plink ran on all 22 chromosomes every time.

## Attribution of the 2 unrecovered leads

|  | in panel | absent from panel |
|---|---|---|
| absorbed into an adjacent local locus | **2** | 0 |
| outside every local locus | **0** | 0 |

Both are present in the panel and both fall inside a local locus: the signal was
detected and merged into a neighbouring block. Neither sits at a position the
pipeline failed to see. Both are at multi-allelic positions that retain
synthetic IDs by design, so the GWAS rsID could not bind to a panel row.

## Locus boundaries

568 of the 1,032 shared loci have bit-identical start and end; 399 are narrower
and 65 wider. Narrower is expected and systematic: FUMA's `refSNPs=1` extends an
LD block using reference-panel variants absent from the GWAS, which this tool
does not do (Known Limitation #1 upstream). Implementing that is the remaining
gap — it is a code addition, not a data one.

## Progression

| configuration | shared / FUMA | recall | regions at 100% |
|---|---|---|---|
| MAGMA panel, 50 Mb window | 1,007 / 1,034 | 97.4% | 6 |
| MAGMA panel, 1 Mb window | 1,016 / 1,034 | 98.3% | 10 |
| **rebuilt 1000G panel, 1 Mb window** | **1,032 / 1,034** | **99.8%** | **14** |

Superseded outputs are retained at `validation_clumpkb50Mb/`,
`validation_g1000magma_1Mb/` and `validation_newpanel_synthids/`.

## Files

| path | role |
|---|---|
| `build_fuma_panel.py` | rebuild the 1000G Phase 3 EUR panel from VCFs |
| `assign_panel_rsids.py` | label the panel with rsIDs |
| `validate_vs_fuma.py` | run + compare one region |
| `sweep_all_regions.py` | all 16 regions, writes the summary CSV |
| `<path/to>/g1000_phase3/` | the rebuilt 1000G Phase 3 EUR reference panel |
