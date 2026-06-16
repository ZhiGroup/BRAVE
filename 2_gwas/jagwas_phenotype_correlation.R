library(Matrix)
library(GMMAT)
library(data.table)
library(dplyr)
library(glue)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Set these to match config/paths.yaml
JAGWAS_OUT  <- "<set to cfg.gwas.jagwas_out in config/paths.yaml>"
COVAR_DIR   <- "<set to cfg.gwas.covar_dir in config/paths.yaml>"
PHENO_DIR   <- "<set to cfg.gwas.pheno_dir in config/paths.yaml>"
COHORTS_DIR <- "<set to cfg.cohorts.dir in config/paths.yaml>"
DISCOVERY_COHORT_CSV <- "<set to cfg.cohorts.discovery in config/paths.yaml>"
SPARSE_GRM  <- "<set to cfg.gwas.sparse_grm in config/paths.yaml>"
# ──────────────────────────────────────────────────────────────────────────────

# --- 1. Configuration & Variables ---
region_name  <- 'Right_Caudate'
cohort_type  <- 'replication'

# Define paths
base_out_dir <- glue("{JAGWAS_OUT}/output_{cohort_type}/")
region_out_dir <- file.path(base_out_dir, region_name)

# Create directories if they don't exist
if (!dir.exists(region_out_dir)) {
    dir.create(region_out_dir, recursive = TRUE)
}

cc_discovery <- file.path(COVAR_DIR, "T1_ccovar_discovery_v2")
qc_discovery <- file.path(COVAR_DIR, "T1_qcovar_discovery_v2")
discovery_sample_list <- file.path(COHORTS_DIR, DISCOVERY_COHORT_CSV)
root_dir <- PHENO_DIR

# --- 2. Load and Align Master Data ---
sample_select <- fread(discovery_sample_list, data.table = FALSE)

# Load GRM
grm_sparse <- fread(paste0(SPARSE_GRM, ".grm.sp"), data.table = FALSE)
grm_id <- fread(paste0(SPARSE_GRM, ".grm.id"), data.table = FALSE)

# Load Covariates
cor_variable <- fread(cc_discovery, data.table = FALSE)
qcor_variable <- fread(qc_discovery, data.table = FALSE)

# Force IDs to character for robust merging
cor_variable$IID <- as.character(cor_variable$IID)
qcor_variable$IID <- as.character(qcor_variable$IID)

# Safe merge of covariates
combine_variable <- merge(qcor_variable, cor_variable, by = c("FID", "IID"))
rownames(combine_variable) <- as.character(combine_variable$IID)

# Construct GRM Matrix
grm_matrix <- sparseMatrix(
    i = grm_sparse$V1 + 1, 
    j = grm_sparse$V2 + 1,
    x = grm_sparse$V3,
    symmetric = TRUE
)
rownames(grm_matrix) <- as.character(grm_id$V2)
colnames(grm_matrix) <- as.character(grm_id$V2)

# Find true intersection across all master files
intersample <- Reduce(intersect, list(as.character(sample_select$IID), 
                                      as.character(combine_variable$IID), 
                                      as.character(rownames(grm_matrix))))

message(paste("Final overlapping samples across master files:", length(intersample)))

# --- 3. Process Phenotype Files ---
all_files <- list.files(root_dir)
all_files <- all_files[grep(region_name, all_files)]
all_files <- all_files[order(as.numeric(gsub(".*QT", "", all_files)))] # Corrected regex

save_df <- NULL  

for (file_name in all_files) {
    message(paste("Processing:", file_name))
    temp_path <- file.path(root_dir, file_name)
    
    # Read phenotype and fix headers/types
    select_phenotype <- fread(temp_path, data.table = FALSE)
    colnames(select_phenotype)[1:2] <- c('FID', 'IID') # Standardize names
    select_phenotype$IID <- as.character(select_phenotype$IID) # Match intersample type
    rownames(select_phenotype) <- select_phenotype$IID 
    
    # Subset data
    select_phenotype_sample <- select_phenotype[intersample, , drop = FALSE]
    select_combine_variable <- combine_variable[intersample, , drop = FALSE]
    
    if (nrow(select_phenotype_sample) == 0) next
    
    # Merge for LMM and remove missing values
    LMM_df <- cbind(select_combine_variable, 
                    pheno_type = select_phenotype_sample[,3])
    LMM_df <- na.omit(LMM_df)
    
    if (nrow(LMM_df) == 0) {
        warning(paste("Zero samples after na.omit for", file_name))
        next
    }
    
    # Align GRM with current samples
    grm_matrix_select <- grm_matrix[as.character(LMM_df$IID), as.character(LMM_df$IID), drop = FALSE]

    # Fit Mixed Model
    fit <- glmmkin(
        pheno_type ~ PC0 + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + 
                     AGE + `AGE^2` + SEXxAGE + `SEXxAGE^2` + SEX + 
                     `25735` + `25000` + `25756` + `25757` + `25758` + `25759` + 
                     `53` + `53^2` + `54`,
        data = LMM_df,
        kins = grm_matrix_select,
        id = "IID",
        family = gaussian(link = "identity")
    )
    
    # Store residuals
    residuals_pheno <- fit$residuals
    if (length(residuals_pheno) > 0) {
        if (is.null(save_df)) {
            save_df <- data.frame(residuals_pheno, row.names = names(residuals_pheno))
        } else {
            save_df <- cbind(save_df, residuals_pheno)
        }
    }
}

# --- 4. Compute & Save Output ---
if (!is.null(save_df) && ncol(save_df) > 1) {
    save_df_cor <- cor(save_df, use = "pairwise.complete.obs")
    
    cor_file <- file.path(region_out_dir, paste0(region_name, "_residuals_cor.txt"))
    resid_file <- file.path(region_out_dir, paste0(region_name, "_residuals_original.txt"))
    
    write.table(save_df_cor, cor_file, row.names = FALSE, col.names = FALSE, quote = FALSE)
    write.table(save_df, resid_file, row.names = FALSE, col.names = FALSE, quote = FALSE)
    
    message("Files saved successfully in: ", region_out_dir)
}