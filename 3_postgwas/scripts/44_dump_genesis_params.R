#!/usr/bin/env Rscript
# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.
# 44_dump_genesis_params.R
#
# Extract the fitted GENESIS effect-size distribution parameters and the
# forward projections from every per-trait fit, into two tidy CSVs that can be
# shipped as supplementary tables.
#
# Why this exists: the fitted parameters live only inside 27 fit_2comp.rds
# objects, and the projections live in 27 projection.csv files whose column
# HEADERS ARE MISASSIGNED at source (see the rename below). Every published
# number is correct because the figure renderer applies that rename, but no
# shipped table carries pi_c, so Supplementary Note 3's "12 of 16 regions"
# claim has no display item. This produces one.
#
# The projection.csv "model" column reads "3comp"; the fit is TWO-component
# (39_genesis_polygenicity_pilot.R passes modelcomponents = 2 and saves
# fit_2comp.rds). The label string was never updated. We write the true value.
#
# Usage:
#   Rscript 44_dump_genesis_params.R --genesis-dir <dir> --out-dir <dir>

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[i + 1]
}

GENESIS_DIR <- get_arg("--genesis-dir",
  "<EXTERNAL: genesis>")
OUT_DIR <- get_arg("--out-dir", GENESIS_DIR)

dirs <- list.dirs(GENESIS_DIR, recursive = FALSE, full.names = TRUE)
dirs <- dirs[grepl("^(BRE_|ENIGMA_|height_)", basename(dirs))]
dirs <- dirs[file.exists(file.path(dirs, "fit_2comp.rds"))]
cat(sprintf("found %d fits\n", length(dirs)))

par_rows <- list()
proj_rows <- list()

for (d in dirs) {
  trait <- basename(d)
  fit <- readRDS(file.path(d, "fit_2comp.rds"))
  est <- fit$estimates
  v <- est[["Parameter (pic, sigmasq, a) estimates"]]
  s <- est[["S.D. of parameter estimates"]]

  par_rows[[length(par_rows) + 1]] <- data.frame(
    trait        = trait,
    pi_c         = v[1],
    pi_c_sd      = s[1],
    sigma_sq     = v[2],
    sigma_sq_sd  = s[2],
    intercept    = v[3],
    intercept_sd = s[3],
    n_sSNPs      = as.character(est[["Number of sSNPs (sd)"]]),
    heritability = as.character(est[["Total heritability in log-odds-ratio scale (sd)"]]),
    M_after_QC   = est[["Total number of SNPs in the GWAS study after quality control"]],
    model        = "2-component",
    stringsAsFactors = FALSE)

  pf <- file.path(d, "projection.csv")
  if (file.exists(pf)) {
    p <- read.csv(pf, stringsAsFactors = FALSE)
    # The source headers are misassigned; this is the same rename the figure
    # renderer applies (41_render_genesis_fig5_per_region.py, load_projection).
    proj_rows[[length(proj_rows) + 1]] <- data.frame(
      trait                  = trait,
      N                      = p$N,
      h2_total               = p$n_discoveries,
      n_loci_gws             = p$GV_pct,
      frac_h2_captured       = p$herit_explained,
      stringsAsFactors = FALSE)
  }
}

par_df <- do.call(rbind, par_rows)
proj_df <- do.call(rbind, proj_rows)
par_df <- par_df[order(par_df$trait), ]
proj_df <- proj_df[order(proj_df$trait, proj_df$N), ]

write.csv(par_df, file.path(OUT_DIR, "genesis_params_all.csv"), row.names = FALSE)
write.csv(proj_df, file.path(OUT_DIR, "genesis_projections_all.csv"), row.names = FALSE)
cat(sprintf("wrote %d parameter rows and %d projection rows to %s\n",
            nrow(par_df), nrow(proj_df), OUT_DIR))
