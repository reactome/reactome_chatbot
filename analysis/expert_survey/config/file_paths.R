# File path configuration for expert survey analysis

ANALYSIS_ROOT <- "analysis/expert_survey"

DATA_PATH <- file.path(ANALYSIS_ROOT, "survey_results.csv")

RESULTS_DIR <- file.path(ANALYSIS_ROOT, "results")
TABLES_DIR <- file.path(RESULTS_DIR, "tables")
RAW_DATA_DIR <- file.path(RESULTS_DIR, "raw_data")
SENSITIVITY_DIR <- file.path(RESULTS_DIR, "sensitivity")
STRATIFIED_DIR <- file.path(RESULTS_DIR, "stratified")

PROCESSED_LONG_PATH <- file.path(RAW_DATA_DIR, "survey_long.csv")
PROCESSED_PAIRED_PATH <- file.path(RAW_DATA_DIR, "paired_differences.csv")

