library(Matrix)
library(GMMAT)
library(data.table)
library(dplyr)
library(glue)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Set these to match config/paths.yaml
COVAR_DIR    <- "<set to cfg.gwas.covar_dir in config/paths.yaml>"
PHENO_DIR    <- "<set to cfg.gwas.pheno_dir in config/paths.yaml>"
FASTGWA_OUT  <- "<set to cfg.gwas.fastgwa_out in config/paths.yaml>"
JAGWAS_OUT   <- "<set to cfg.gwas.jagwas_out in config/paths.yaml>"
SPARSE_GRM   <- "<set to cfg.gwas.sparse_grm in config/paths.yaml>"
COHORTS_DIR  <- "<set to cfg.cohorts.dir in config/paths.yaml>"
DISCOVERY_COHORT_CSV   <- "<set to cfg.cohorts.discovery in config/paths.yaml>"
REPLICATION_COHORT_CSV <- "<set to cfg.cohorts.replication in config/paths.yaml>"
# ──────────────────────────────────────────────────────────────────────────────

#--1. HANDLE COMMAND LINE ARGUMENTS--------------

args <- commandArgs(trailingOnly = TRUE)
if ( length(args) <2 ){

    stop("Usage: Rscript jagwas_phenotype_correlation_cli.R <region_name> <cohort_type>")
}

region_name <- args[1]
cohort_type <- args[2]


# --- 2. DYNAMIC PATH SELECTION ---
if (cohort_type == "discovery") {


    cc_path <- file.path(COVAR_DIR, "T1_ccovar_discovery_v2")

    qc_path <- file.path(COVAR_DIR, "T1_qcovar_discovery_v2")

    sample_list_path <- file.path(COHORTS_DIR, DISCOVERY_COHORT_CSV)
    
    root_dir <- PHENO_DIR
    fastgwa_output_dir <- FASTGWA_OUT

} else {

    cc_path <- file.path(COVAR_DIR, "T1_ccovar_replication_v2")

    qc_path <- file.path(COVAR_DIR, "T1_qcovar_replication_v2")

    sample_list_path <- file.path(COHORTS_DIR, REPLICATION_COHORT_CSV)

    root_dir <- file.path(PHENO_DIR, "replication", "input")
    fastgwa_output_dir <- file.path(FASTGWA_OUT, "replication", "output")
}



# --- 1. Configuration & Variables ---
#region_name  <- 'Brain_Stem_or_4th_Ventricle'
#cohort_type  <- 'discovery'

# Define paths
base_out_dir <- glue("{JAGWAS_OUT}/output_{cohort_type}/")
region_out_dir <- file.path(base_out_dir, region_name)

# Create directories if they don't exist
if (!dir.exists(region_out_dir)) {
    dir.create( region_out_dir,  recursive = TRUE)
}


# --- 2. Load and Align Master Data ---
sample_select <- fread(sample_list_path, data.table = FALSE)

# Load GRM
grm_sparse <- fread(paste0(SPARSE_GRM, ".grm.sp"), data.table = FALSE)
grm_id <- fread(paste0(SPARSE_GRM, ".grm.id"), data.table = FALSE)

# Load Covariates
cor_variable <- fread(cc_path, data.table = FALSE)
qcor_variable <- fread(qc_path, data.table = FALSE)

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

message(paste0("Final overlapping samples across master files:", length(intersample)))

# --- 3. Process Phenotype Files ---
all_input_files <- list.files(root_dir)


all_input_files <- all_input_files[grep(region_name, all_input_files)]
all_input_files <- all_input_files[grep(cohort_type, all_input_files)]
#all_files <- all_files[grep(paste0("^", cohort_type, ".*", region_name, ".*\\.fastGWA$"), all_files)]
#all_files <- all_files[order(as.numeric(gsub(".*QT", "", all_files)))] # Corrected regex
all_input_files <- all_input_files[order(as.numeric(gsub(".*QT(\\d+).*", "\\1", all_input_files)))]
# 4. Print verification to console

# 2. Filter input files based on the existence of successful .fastGWA.fastGWA outputs
valid_input_files <- c()

for (input_f in all_input_files ){
    # Extract the QT identifier (e.g., QT85)
    qt_id <- gsub(".*(QT\\d+).*", "\\1", input_f)
    # Construct the expected output filename template
    # template: discovery_WM_QT85.fastGWA.fastGWA
    expected_output <- paste0(cohort_type, "_", 
                              region_name, "_", qt_id, ".fastGWA.fastGWA")
    
    output_path <- file.path(fastgwa_output_dir, expected_output)

    # Only analyze phenotypes that successfully finished GWAS
if (file.exists(output_path)) {
        valid_input_files <- c(valid_input_files, input_f)
    } else {
        message(paste("SKIPPING input file:", input_f, "- No successful fastGWA output found."))
    }
}




message(paste0("Region: ", region_name, " | Cohort: ", cohort_type))
message(paste0("Got ", length(valid_input_files), " valid .fastGWA files. They are:"))
print(valid_input_files)
print("")
print("")
save_df <- NULL  
successful_qts <- c()
for (file_name in valid_input_files) {

    
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

    message(paste("Got LMM_df:", nrow(LMM_df)))
    if (nrow(LMM_df) == 0) {
        warning(paste("Zero samples after na.omit for", file_name))
        next
    }
    
    # Align GRM with current samples
    grm_matrix_select <- grm_matrix[as.character(LMM_df$IID), as.character(LMM_df$IID), drop = FALSE]

    # Fit Mixed Model
    fit <- tryCatch({
        glmmkin(
            pheno_type ~ PC0 + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + 
                         AGE + `AGE^2` + SEXxAGE + `SEXxAGE^2` + SEX + 
                         `25735` + `25000` + `25756` + `25757` + `25758` + `25759` + 
                         `53` + `53^2` + `54`,
            data = LMM_df, kins = grm_matrix[as.character(LMM_df$IID), as.character(LMM_df$IID)],
            id = "IID", family = gaussian(link = "identity")
        )
    }, error = function(e) {
        message(paste("SKIPPING", file_name, ":", e$message))
        return(NULL)
    })

    if (!is.null(fit) && length(fit$residuals) > 0) {
        if (sd(fit$residuals, na.rm = TRUE) > 0) {
            
            # 1. Get the current QT label
            qt_label <- gsub(".*(QT\\d+).*", "\\1", file_name)
            
            # 2. Append to successful list
            successful_qts <- c(successful_qts, qt_label)
            
            # 3. Add to save_df and label the column IMMEDIATELY
            new_resids <- data.frame(fit$residuals)
            colnames(new_resids) <- qt_label # This ensures the name is tied to the data
            
            if (is.null(save_df)) {
                save_df <- new_resids
            } else {
                # Use cbind with the named dataframe to keep dimensions aligned
                save_df <- cbind(save_df, new_resids)
            }
        } else {
            message(paste("SKIPPING", file_name, ": Zero standard deviation."))
        }
    }
}

# --- 4. Compute & Save Output ---
if (!is.null(save_df) && ncol(save_df) > 1) {
    # Assign headers so Bash can find "successful_files"
    colnames(save_df) <- successful_qts
    
    save_df_cor <- cor(save_df, use = "pairwise.complete.obs")
    
    cor_file <- file.path(region_out_dir, paste0(region_name, "_residuals_cor.txt"))
    resid_file <- file.path(region_out_dir, paste0(region_name, "_residuals_original.txt"))
    
    # Save correlation matrix (JAGWAS input)
    write.table(save_df_cor, cor_file, row.names = FALSE, col.names = FALSE, quote = FALSE)
    # Save residuals with headers for Bash parsing
    write.table(save_df, resid_file, row.names = FALSE, col.names = TRUE, quote = FALSE)
    
    message("Process complete. Files saved in: ", region_out_dir)
} else {
    message("Error: Not enough phenotypes passed model fitting and variance checks.")
}






