#!/usr/bin/env Rscript
# 45_genesis_refit_startingpic.R
#
# Refit one trait's 2-component GENESIS model from given starting values of pi_c.
# Why: four BRE fits made by 39_genesis_polygenicity_pilot.R
# (R.Hippocampus QT35, L.Pallidum QT11, R.Pallidum QT20, R.Putamen QT118) stopped after
# 5-10 EM iterations at GENESIS's default starting pi_c = 0.01, because pi_c moved less
# than GENESIS's pi_c tolerance (1e-6) in the first iterations.
#
# Everything else is identical to Script 39: the same loader (SNP, Z = BETA/SE, N), filter
# = TRUE, modelcomponents = 2, LDcutoff = 0.1, LDwindow = 1, stratification = TRUE,
# seeds = 42, the same projection() grid and arguments, and the same projection.csv headers
# (shifted at source, as in Script 39) so that Scripts 41 and 44 read the output unchanged.
# The sumstat is loaded ONCE and fitted once per starting value.
#
# Usage:
#   Rscript 45_genesis_refit_startingpic.R --trait <name> --sumstat <path> \
#       --startingpic 0.005,0.001 --cores 8 --out-root <dir>

suppressMessages({
  .libPaths(c(
    "<EXTERNAL: library_zhang_genesis>",
    .libPaths()))
  library(GENESIS)
})
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[i + 1]
}
trait   <- get_arg("--trait")
sspath  <- get_arg("--sumstat")
sps     <- as.numeric(strsplit(get_arg("--startingpic", "0.005,0.001"), ",")[[1]])
cores   <- as.integer(get_arg("--cores", "8"))
outroot <- get_arg("--out-root",
  "<EXTERNAL: jagwas_paper>/post_gwas_analysis/results/genesis_refit")
stopifnot(!is.null(trait), !is.null(sspath), all(is.finite(sps)))
stopifnot(packageVersion("GENESIS") == "1.0")

# Same loader as Script 39 (BETA/SE/N branch, which is the FastGWA format).
cat(sprintf("Reading %s ...\n", sspath)); t0 <- Sys.time()
df <- read.table(sspath, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
cols <- toupper(colnames(df))
stopifnot(all(c("SNP", "BETA", "SE", "N") %in% cols))
sumstat <- data.frame(SNP = df[[which(cols == "SNP")[1]]],
                      Z = as.numeric(df[[which(cols == "BETA")[1]]]) / as.numeric(df[[which(cols == "SE")[1]]]),
                      N = as.numeric(df[[which(cols == "N")[1]]]), stringsAsFactors = FALSE)
rm(df)
sumstat <- sumstat[complete.cases(sumstat) & is.finite(sumstat$Z) & is.finite(sumstat$N), ]
cat(sprintf("  rows after NA filter: %d (load %.0f s)\n", nrow(sumstat),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

for (sp in sps) {
  tag <- sprintf("%s_sp%s", trait, format(sp, scientific = FALSE))
  od <- file.path(outroot, trait, sprintf("sp%s", format(sp, scientific = FALSE)))
  dir.create(od, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("\n=== GENESIS refit — %s ===\n", tag)); flush.console()
  t0 <- Sys.time(); set.seed(42)
  fit <- tryCatch(genesis(summarydata = sumstat, filter = TRUE, modelcomponents = 2, cores = cores,
                          LDcutoff = 0.1, LDwindow = 1, print = TRUE, printfreq = 5, qqplot = FALSE,
                          stratification = TRUE, seeds = 42, startingpic = sp),
                  error = function(e) { cat("  FIT FAILED:", conditionMessage(e), "\n"); NULL })
  cat(sprintf("  fit time: %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  if (is.null(fit)) next
  saveRDS(fit, file.path(od, "fit_2comp.rds"))
  est <- fit$estimates
  v <- est[["Parameter (pic, sigmasq, a) estimates"]]; s <- est[["S.D. of parameter estimates"]]
  cat(sprintf("  RESULT %s pic=%.6g sd=%.4g sigmasq=%.4g llk=%.6f h2=%s\n", tag, v[1], s[1], v[2],
              est[["Composite log-likelihood of fitted model"]],
              est[["Total heritability in log-odds-ratio scale (sd)"]]))
  rows <- list()
  for (N_proj in c(22878, 50000, 100000, 200000, 500000, 1000000)) {
    p <- tryCatch(projection(est = v, v = est[["Covariance matrix of parameter estimates"]], n = N_proj,
                             gwas.significance = 5e-8, M = 1070777, CI = FALSE, seeds = 42),
                  error = function(e) NULL)
    if (!is.null(p)) rows[[length(rows) + 1]] <- data.frame(N = N_proj, n_discoveries = p[[1]],
      GV_pct = p[[2]], herit_explained = p[[3]], model = "3comp")
  }
  write.csv(do.call(rbind, rows), file.path(od, "projection.csv"), row.names = FALSE)
  cat(sprintf("=== DONE %s ===\n", tag))
}
