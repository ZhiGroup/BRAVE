#!/bin/bash

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Set these to match config/paths.yaml
JAGWAS_EXE="<set to cfg.tools.jagwas in config/paths.yaml>"
FASTGWA_OUT="<set to cfg.gwas.fastgwa_out in config/paths.yaml>"
JAGWAS_OUT="<set to cfg.gwas.jagwas_out in config/paths.yaml>"
# ──────────────────────────────────────────────────────────────────────────────

# #!/bin/bash

# # Define Region and Cohort from arguments
# REGION_NAME=$1   
# COHORT_NAME=$2   

# # Check if arguments are provided
# if [ -z "$REGION_NAME" ] || [ -z "$COHORT_NAME" ]; then
#     echo "Usage: ./run_jagwas.sh <REGION_NAME> <COHORT_NAME>"
#     exit 1
# fi

# # Set Executable Path
# JAGWAS_EXE="${JAGWAS_EXE}"

# # Set Paths based on Cohort
# if [ "$COHORT_NAME" == "discovery" ]; then
#     FASTGWA_DIR="${FASTGWA_OUT}/"
#     COR_DIR="${JAGWAS_OUT}/output_discovery/${REGION_NAME}"
# else
#     FASTGWA_DIR="${FASTGWA_OUT}/replication/output/"
#     COR_DIR="${JAGWAS_OUT}/output_replication/${REGION_NAME}"
# fi

# COR_MATRIX="${COR_DIR}/${REGION_NAME}_residuals_cor.txt"
# RESIDUALS_FILE="${COR_DIR}/${REGION_NAME}_residuals_original.txt"
# OUTPUT_FILE="${COR_DIR}/${REGION_NAME}_JAGWAS_results.txt"

# # --- 1. PRE-EXECUTION CHECKS ---
# if [ ! -f "$RESIDUALS_FILE" ]; then
#     echo "ERROR: Residuals file missing. Run R script first."
#     exit 1
# fi

# # --- 2. EXTRACT SUCCESSFUL DIMENSIONS ---
# # Extract QT identifiers (e.g., QT0, QT1) from the R-generated headers
# SUCCESSFUL_QTS=$(head -n 1 "$RESIDUALS_FILE" | tr ' ' '\n' | grep -o "QT[0-9]*")
# VALID_DIM_COUNT=$(echo "$SUCCESSFUL_QTS" | wc -w)

# echo "----------------------------------------------------"
# echo "DEBUG: SUCCESSFUL DIMENSIONS"
# echo "Count: $VALID_DIM_COUNT"
# echo "Example QTs: $(echo $SUCCESSFUL_QTS | head -n 5)"
# echo "----------------------------------------------------"

# # --- 3. CONSTRUCT VALID FILE PATHS ---
# # 1. Get all regional files for this cohort
# # 2. Use grep -v "\.log$" to strictly exclude all log files
# # 3. Use grep "\.fastGWA$" to ensure we only get the data files
# ALL_REGIONAL_FILES=$(ls ${FASTGWA_DIR} | grep "^${COHORT_NAME}" | grep "${REGION_NAME}" | grep -v "\.log$" | grep "\.fastGWA$")

# VALID_FILE_PATHS=""
# for qt in $SUCCESSFUL_QTS; do
#     # Anchoring with "_${qt}." ensures we don't match QT1 with QT11
#     # We look for the exact data file, handling the double extension seen in your ls output
#     match=$(echo "$ALL_REGIONAL_FILES" | grep "_${qt}\.fastGWA")
    
#     if [ ! -z "$match" ]; then
#         VALID_FILE_PATHS="${VALID_FILE_PATHS} ${FASTGWA_DIR}${match}"
#     else
#         echo "DEBUG WARNING: No match found for $qt in $FASTGWA_DIR"
#     fi
# done

# echo "----------------------------------------------------"
# echo "DEBUG: VALID_FILE_PATHS"
# # Print the first two files to verify the full path and double extension handling
# echo "First file: $(echo $VALID_FILE_PATHS | awk '{print $1}')"
# echo "Second file: $(echo $VALID_FILE_PATHS | awk '{print $2}')"
# echo "Total files in path: $(echo $VALID_FILE_PATHS | wc -w)"
# echo "----------------------------------------------------"

# # --- 4. EXECUTION ---
# if [ $(echo $VALID_FILE_PATHS | wc -w) -eq 0 ]; then
#     echo "ERROR: No valid file paths constructed. Check your FASTGWA_DIR or REGION_NAME."
#     exit 1
# fi

# echo "Starting JAGWAS..."
# $JAGWAS_EXE --outputFilePath "$OUTPUT_FILE" \
#             --cor_matrix "$COR_MATRIX" \
#             --nrow 10000 \
#             --MAF 0.01 \
#             --score_test 0 \
#             --beta_se 1 \
#             --logP 0 \
#             --delim "\t" \
#             --fileNames $VALID_FILE_PATHS

# echo "Process completed. Output: $OUTPUT_FILE"




#!/bin/bash

# Define Region and Cohort from arguments
REGION_NAME=$1   
COHORT_NAME=$2   

# Check if arguments are provided
if [ -z "$REGION_NAME" ] || [ -z "$COHORT_NAME" ]; then
    echo "Usage: ./run_jagwas.sh <REGION_NAME> <COHORT_NAME>"
    exit 1
fi

# Set Executable Path (JAGWAS_EXE set in CONFIG block above)

# Set Paths based on Cohort
if [ "$COHORT_NAME" == "discovery" ]; then
    FASTGWA_DIR="${FASTGWA_OUT}/"
    COR_DIR="${JAGWAS_OUT}/output_discovery/${REGION_NAME}"
else
    FASTGWA_DIR="${FASTGWA_OUT}/replication/output/"
    COR_DIR="${JAGWAS_OUT}/output_replication/${REGION_NAME}"
fi

COR_MATRIX="${COR_DIR}/${REGION_NAME}_residuals_cor.txt"
RESIDUALS_FILE="${COR_DIR}/${REGION_NAME}_residuals_original.txt"
OUTPUT_FILE="${COR_DIR}/${REGION_NAME}_JAGWAS_results.txt"

# --- 1. PRE-EXECUTION CHECKS ---
if [ ! -f "$RESIDUALS_FILE" ]; then
    echo "ERROR: Residuals file missing. Run R script first."
    exit 1
fi

# --- 2. DETECT HEADER TYPE AND EXTRACT SUCCESSFUL DIMENSIONS ---
# Check the first line for the string "QT"
FIRST_LINE=$(head -n 1 "$RESIDUALS_FILE")

if [[ "$FIRST_LINE" == *"QT"* ]]; then
    # PATH A: Header exists - use successful dimensions from file
    echo "STATUS: 'QT' detected in header. Using successful dimensions from residuals file."
    SUCCESSFUL_QTS=$(echo "$FIRST_LINE" | tr ' ' '\n' | grep -o "QT[0-9]*")
    echo "successfull QTS"
    echo $SUCCESSFUL_QTS    
else
    # PATH B: Fallback - generate a sequence of 0-127 if headers are missing
    echo "STATUS: No 'QT' detected in header. Generating sequence for 128 dimensions."
    SUCCESSFUL_QTS=$(seq 0 127 | sed 's/^/QT/')

fi

VALID_FILE_PATHS=""
FOUND_COUNT=0
echo "Searching for files in: $FASTGWA_DIR"

for qt in $SUCCESSFUL_QTS; do
    # Exact filename construction using your template
    FILE_NAME="${COHORT_NAME}_${REGION_NAME}_${qt}.fastGWA.fastGWA"
    FULL_PATH="${FASTGWA_DIR}${FILE_NAME}"


    if [ -f "$FULL_PATH" ]; then
        VALID_FILE_PATHS="${VALID_FILE_PATHS} ${FULL_PATH}"
        ((FOUND_COUNT++))
    else
        # If the file is missing, we MUST alert the user because JAGWAS will crash
        echo "!!! CRITICAL MISSING FILE: $qt -> $FILE_NAME"
    fi
done


# --- 4. DIMENSION VALIDATION ---
MATRIX_DIM=$(head -n 1 "$COR_MATRIX" | wc -w)

echo "----------------------------------------------------"
echo "DEBUG: JAGWAS PREPARATION"
echo "Matrix Dimensions (Expected): $MATRIX_DIM"
echo "Physical Files Found: $FOUND_COUNT"
echo "----------------------------------------------------"


# --- 5. EXECUTION ---
# JAGWAS requires exact 1-to-1 mapping
if [ "$FOUND_COUNT" -ne "$MATRIX_DIM" ]; then
    echo "ERROR: Dimension mismatch! Matrix has $MATRIX_DIM columns but we only found $FOUND_COUNT files."
    echo "JAGWAS would crash with 'incompatible matrix dimensions'. Check the missing files above."
    exit 1
fi

echo "Starting JAGWAS..."
$JAGWAS_EXE --outputFilePath "$OUTPUT_FILE" \
            --cor_matrix "$COR_MATRIX" \
            --nrow 10000 \
            --MAF 0.01 \
            --score_test 0 \
            --beta_se 1 \
            --logP 0 \
            --delim "\t" \
            --fileNames $VALID_FILE_PATHS

echo "Process completed successfully."
















# -----------old -------------

# if [[ "$FIRST_LINE" == *"QT"* ]]; then
#     # PATH A: Header exists - use successful dimensions from file
#     echo "STATUS: 'QT' detected in header. Using successful dimensions from residuals file."
#     SUCCESSFUL_QTS=$(echo "$FIRST_LINE" | tr ' ' '\n' | grep -o "QT[0-9]*")
#     echo "successfull QTS"
#     echo $SUCCESSFUL_QTS    
#     # Pre-filter regional files for matching
#     ALL_REGIONAL_FILES=$(ls ${FASTGWA_DIR} | grep "^${COHORT_NAME}" | grep "${REGION_NAME}" | grep -v "\.log$" | grep "\.fastGWA$")

#     VALID_FILE_PATHS=""
#     for qt in $SUCCESSFUL_QTS; do
#         match=$(echo "$ALL_REGIONAL_FILES" | grep "_${qt}\.fastGWA")
#         if [ ! -z "$match" ]; then
#             VALID_FILE_PATHS="${VALID_FILE_PATHS} ${FASTGWA_DIR}${match}"
#         fi
#     done
# else
#     # PATH B: No header - assume 128 dimensions and use numeric sorting
#     echo "STATUS: No 'QT' detected in header. Falling back to numeric sort of all dimensions."
    
#     FILES=$(ls ${FASTGWA_DIR} | \
#             grep "^${COHORT_NAME}" | \
#             grep "${REGION_NAME}" | \
#             grep -v "\.log$" | \
#             grep "\.fastGWA$" | \
#             perl -pe 's/.*QT(\d+).*/$1\t$_/' | sort -n | cut -f2)

#     VALID_FILE_PATHS=""
#     for f in $FILES; do
#         VALID_FILE_PATHS="${VALID_FILE_PATHS} ${FASTGWA_DIR}${f}"
#     done
# fi

# # --- 3. DEBUG & VERIFICATION ---
# VALID_COUNT=$(echo $VALID_FILE_PATHS | wc -w)
# echo $VALID_FILE_PATHS
# echo "----------------------------------------------------"
# echo "DEBUG: JAGWAS PREPARATION"
# echo "Region: $REGION_NAME | Cohort: $COHORT_NAME"
# echo "Valid Dimensions Found: $VALID_COUNT"
# echo "First file: $(echo $VALID_FILE_PATHS | awk '{print $1}')"
# echo "----------------------------------------------------"

# # --- 4. EXECUTION ---
# if [ "$VALID_COUNT" -eq 0 ]; then
#     echo "ERROR: No valid file paths constructed."
#     exit 1
# fi

# echo "Starting JAGWAS..."
# $JAGWAS_EXE --outputFilePath "$OUTPUT_FILE" \
#             --cor_matrix "$COR_MATRIX" \
#             --nrow 10000 \
#             --MAF 0.01 \
#             --score_test 0 \
#             --beta_se 1 \
#             --logP 0 \
#             --delim "\t" \
#             --fileNames $VALID_FILE_PATHS

# echo "Process completed. Output: $OUTPUT_FILE"