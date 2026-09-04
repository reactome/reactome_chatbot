#!/usr/bin/env Rscript

# Setup script: load data, ensure output directories, and write processed datasets.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})

source("analysis/expert_survey/config/file_paths.R")
source("analysis/expert_survey/config/analysis_parameters.R")
source("analysis/expert_survey/utils/data_preparation.R")

# Create output directories -----------------------------------------------------
dirs_to_create <- c(
  RESULTS_DIR,
  TABLES_DIR,
  RAW_DATA_DIR,
  SENSITIVITY_DIR,
  STRATIFIED_DIR
)

invisible(lapply(dirs_to_create, dir.create, recursive = TRUE, showWarnings = FALSE))

cat("Output directories verified.\n")

# Load and prepare data ---------------------------------------------------------
cat("Loading survey data from:", DATA_PATH, "\n")
survey_wide <- load_and_validate_data(DATA_PATH)

cat("Converting to long format...\n")
survey_long <- to_long_format(survey_wide)

cat("Creating paired dataset...\n")
survey_paired <- create_paired_dataset(survey_long)

# Write processed datasets ------------------------------------------------------
write_csv(survey_long, PROCESSED_LONG_PATH)
write_csv(survey_paired, PROCESSED_PAIRED_PATH)

cat("Processed datasets saved:\n")
cat(" - Long format:", PROCESSED_LONG_PATH, "\n")
cat(" - Paired differences:", PROCESSED_PAIRED_PATH, "\n")

cat("Setup complete.\n")

