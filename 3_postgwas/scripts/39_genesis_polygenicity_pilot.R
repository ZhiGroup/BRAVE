#!/usr/bin/env Rscript
# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.
# 39_genesis_polygenicity_pilot.R
# Pilot — Zhang 2018 GENESIS effect-size distribution + sample-size
# projection for one trait at a time. Wen et al. Fig 5 equivalent.
#
# Usage:
#   Rscript 39_genesis_polygenicity_pilot.R <trait_name> <sumstat_path> [cores]
#
# Inputs:
#   trait_name    short label for output dir (e.g., "BRE_R.Caudate_QT81")
#   sumstat_path  path to summary statistics file
#                   * BRE FastGWA: tab-separated CHR SNP POS A1 A2 N AF1 BETA SE P INFO
#                   * ENIGMA:      tab-separated SNP CHR BP A1 A2 FREQ BETA SE Z P N (gzipped OK)
#                   * height:      special token "BUNDLED_HEIGHT" → load GENESIS heightGWAS data
#   cores         optional CPU threads (default 4)
#
# Outputs (under results/genesis/<trait_name>/):
#   est_2comp.rds, est_3comp.rds   — GENESIS fit objects
#   projection.csv                  — projected # discoveries + h² explained at N grid
#   trait_meta.json                 — trait input summary (N obs, % filtered, etc.)
#   qqplot_3comp.pdf                — QQ plot for 3-component fit

suppressMessages({
  .libPaths(c(
    "<EXTERNAL: library_zhang_genesis>",
    .libPaths()
  ))
  library(GENESIS)   # Zhang 2018 polygenicity tool
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript 39_genesis_polygenicity_pilot.R <trait_name> <sumstat_path> [cores]")
}
trait_name   <- args[1]
sumstat_path <- args[2]
cores        <- if (length(args) >= 3) as.integer(args[3]) else 4L

stopifnot(packageDescription("GENESIS")$Title ==
          "GENetic Effect-Size distribution Inference from Summary-level data")

cat(sprintf("=== GENESIS pilot — %s ===\n", trait_name))
cat(sprintf("  sumstat:      %s\n", sumstat_path))
cat(sprintf("  cores:        %d\n", cores))
cat(sprintf("  GENESIS:      %s v%s\n",
            find.package("GENESIS"), packageVersion("GENESIS")))

# ── Load sumstat in (SNP, Z, N) GENESIS format ──────────────────────────────
load_sumstat <- function(path) {
  if (path == "BUNDLED_HEIGHT") {
    cat("  Loading bundled heightGWAS reference dataset ...\n")
    data(heightGWAS, package = "GENESIS")
    return(heightGWAS)
  }
  cat("  Reading summary stats ...\n")
  is_gz <- grepl("\\.gz$", path)
  df <- read.table(path, header = TRUE, sep = "\t",
                   stringsAsFactors = FALSE,
                   nrows = -1, check.names = FALSE)
  cat(sprintf("  Raw rows: %d, cols: %s\n", nrow(df),
              paste(colnames(df), collapse = ", ")))
  cols <- toupper(colnames(df))
  # Detect format
  if ("BETA" %in% cols && "SE" %in% cols && "N" %in% cols && "SNP" %in% cols) {
    snp_col  <- colnames(df)[which(cols == "SNP")[1]]
    beta_col <- colnames(df)[which(cols == "BETA")[1]]
    se_col   <- colnames(df)[which(cols == "SE")[1]]
    n_col    <- colnames(df)[which(cols == "N")[1]]
    df$BETA <- as.numeric(df[[beta_col]])
    df$SE   <- as.numeric(df[[se_col]])
    df$N    <- as.numeric(df[[n_col]])
    out <- data.frame(
      SNP = df[[snp_col]],
      Z   = df$BETA / df$SE,
      N   = df$N,
      stringsAsFactors = FALSE
    )
  } else if ("Z" %in% cols && "N" %in% cols && "SNP" %in% cols) {
    snp_col <- colnames(df)[which(cols == "SNP")[1]]
    z_col   <- colnames(df)[which(cols == "Z")[1]]
    n_col   <- colnames(df)[which(cols == "N")[1]]
    out <- data.frame(
      SNP = df[[snp_col]],
      Z   = as.numeric(df[[z_col]]),
      N   = as.numeric(df[[n_col]]),
      stringsAsFactors = FALSE
    )
  } else {
    stop("Sumstat must have SNP + (BETA + SE + N) or (Z + N) columns")
  }
  out <- out[complete.cases(out) & is.finite(out$Z) & is.finite(out$N), ]
  cat(sprintf("  After NA filter: %d rows\n", nrow(out)))
  return(out)
}

t0 <- Sys.time()
sumstat <- load_sumstat(sumstat_path)
cat(sprintf("  Load time: %.1f sec\n", as.numeric(difftime(Sys.time(), t0, units="secs"))))

# ── Set up output directory ─────────────────────────────────────────────────
out_dir <- file.path(
  "<EXTERNAL: jagwas_paper>/",
  "post_gwas_analysis/results/genesis",
  trait_name
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
setwd(out_dir)
cat(sprintf("  Output dir:   %s\n", out_dir))

# ── Run genesis() — 2-component model, fast mode (no QQ plot, verbose) ──────
# 2-comp is much faster than 3-comp; covers polygenicity + h² + effect-size variance
# (3-comp adds a second Gaussian for large-effect variants; valuable but slower).
cat("\n--- genesis(): 2-component EM fit (fast mode) ---\n")
flush.console()
t0 <- Sys.time()
set.seed(42)
fit3 <- tryCatch({
  genesis(
    summarydata     = sumstat,
    filter          = TRUE,
    modelcomponents = 2,
    cores           = cores,
    LDcutoff        = 0.1,
    LDwindow        = 1,
    print           = TRUE,
    printfreq       = 5,
    qqplot          = FALSE,
    stratification  = TRUE,
    seeds           = 42
  )
}, error = function(e) {
  cat(sprintf("  2-comp FAILED: %s\n", conditionMessage(e)))
  return(NULL)
})
cat(sprintf("  2-comp time: %.1f min\n",
            as.numeric(difftime(Sys.time(), t0, units="mins"))))
if (!is.null(fit3)) {
  saveRDS(fit3, file = "fit_2comp.rds")
  cat("  saved fit_2comp.rds\n")
  est <- fit3$estimates
  param_vec <- est[["Parameter (pic, sigmasq, a) estimates"]]
  param_sd  <- est[["S.D. of parameter estimates"]]
  cat(sprintf("  pic (frac causal):      %.4g  (SD %.4g)\n", param_vec[1], param_sd[1]))
  cat(sprintf("  sigmasq (effect var):   %.4g  (SD %.4g)\n", param_vec[2], param_sd[2]))
  cat(sprintf("  a (intercept):          %.4g  (SD %.4g)\n", param_vec[3], param_sd[3]))
  cat(sprintf("  # susceptibility SNPs:  %s\n", est[["Number of sSNPs (sd)"]]))
  cat(sprintf("  heritability (LOR sd):  %s\n", est[["Total heritability in log-odds-ratio scale (sd)"]]))
  cat(sprintf("  M (HM3 SNPs after QC):  %d\n", est[["Total number of SNPs in the GWAS study after quality control"]]))
}

# ── Run projection() — # GWS loci + h² explained at sample-size grid ────────
cat("\n--- projection(): sample-size grid ---\n")
N_grid <- c(22878, 50000, 100000, 200000, 500000, 1000000)
proj_rows <- list()
for (N_proj in N_grid) {
  for (label in c("3comp")) {
    fit <- if (label == "3comp") fit3 else NULL
    if (is.null(fit)) next
    est_vec <- fit$estimates[["Parameter (pic, sigmasq, a) estimates"]]
    cov_mat <- fit$estimates[["Covariance matrix of parameter estimates"]]
    p <- tryCatch({
      projection(est = est_vec, v = cov_mat, n = N_proj,
                 gwas.significance = 5e-8, M = 1070777,
                 CI = FALSE, seeds = 42)
    }, error = function(e) {
      cat(sprintf("  projection N=%d failed: %s\n", N_proj, conditionMessage(e)))
      return(NULL)
    })
    if (!is.null(p)) {
      proj_rows[[length(proj_rows) + 1]] <- data.frame(
        N           = N_proj,
        n_discoveries = if (length(p) >= 1) p[[1]] else NA,
        GV_pct      = if (length(p) >= 2) p[[2]] else NA,
        herit_explained = if (length(p) >= 3) p[[3]] else NA,
        model       = label
      )
    }
  }
}
if (length(proj_rows) > 0) {
  proj <- do.call(rbind, proj_rows)
  write.csv(proj, "projection.csv", row.names = FALSE)
  cat("  saved projection.csv\n")
  print(proj)
}

cat(sprintf("\n=== DONE %s ===\n", trait_name))
