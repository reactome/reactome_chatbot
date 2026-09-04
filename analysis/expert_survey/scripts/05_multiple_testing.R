#!/usr/bin/env Rscript

# Multiple testing correction using Holm method across families.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
})

source("analysis/expert_survey/config/file_paths.R")
source("analysis/expert_survey/config/analysis_parameters.R")

ordinal_results <- read_csv(
  file.path(TABLES_DIR, "ordinal_models_all_stratifications.csv"),
  show_col_types = FALSE
) %>%
  mutate(interaction = as.logical(interaction))

primary_strata <- PRIMARY_METRICS
secondary_strata <- QUESTION_TYPES

ordinal_filtered <- ordinal_results %>%
  filter(str_detect(parameter, "systemReact-to-Me"))

p_values_primary <- ordinal_filtered %>%
  filter(stratum %in% primary_strata, !interaction) %>%
  transmute(
    stratum,
    p_raw = p_value,
    test_family = "Primary"
  ) %>%
  mutate(
    p_holm = p.adjust(p_raw, method = "holm"),
    significant_raw = p_raw < 0.05,
    significant_holm = p_holm < 0.05
  )

p_values_secondary <- ordinal_filtered %>%
  filter(stratum %in% secondary_strata) %>%
  transmute(
    stratum,
    p_raw = p_value,
    test_family = "Secondary"
  ) %>%
  mutate(
    p_holm = p.adjust(p_raw, method = "holm"),
    significant_raw = p_raw < 0.05,
    significant_holm = p_holm < 0.05
  )

exploratory_strata <- setdiff(unique(ordinal_filtered$stratum), c(primary_strata, secondary_strata))

p_values_exploratory <- ordinal_filtered %>%
  filter(stratum %in% exploratory_strata) %>%
  transmute(
    stratum,
    p_raw = p_value,
    test_family = "Exploratory"
  ) %>%
  mutate(
    p_holm = p_raw,
    significant_raw = p_raw < 0.05,
    significant_holm = significant_raw
  )

all_corrections <- bind_rows(
  p_values_primary,
  p_values_secondary,
  p_values_exploratory
)

write_csv(all_corrections, file.path(TABLES_DIR, "multiple_testing_correction_comprehensive.csv"))

cat("Multiple testing corrections saved to multiple_testing_correction_comprehensive.csv\n")

