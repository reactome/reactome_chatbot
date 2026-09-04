#!/usr/bin/env Rscript

# Merge ordinal, nonparametric, Holm corrections, power, and sign tests.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
})

source("analysis/expert_survey/config/file_paths.R")
source("analysis/expert_survey/config/analysis_parameters.R")

ordinal_results <- read_csv(
  file.path(TABLES_DIR, "ordinal_models_all_stratifications.csv"),
  show_col_types = FALSE
)

nonparam_results <- read_csv(
  file.path(TABLES_DIR, "nonparametric_tests_all_stratifications.csv"),
  show_col_types = FALSE
)

holm_results <- read_csv(
  file.path(TABLES_DIR, "multiple_testing_correction_comprehensive.csv"),
  show_col_types = FALSE
)

power_results <- read_csv(
  file.path(TABLES_DIR, "power_analysis_all_stratifications.csv"),
  show_col_types = FALSE
)

sign_question <- read_csv(
  file.path(TABLES_DIR, "sign_test_question_level.csv"),
  show_col_types = FALSE
)

sign_participant <- read_csv(
  file.path(TABLES_DIR, "sign_test_participant_level.csv"),
  show_col_types = FALSE
)

ordinal_filtered <- ordinal_results %>%
  filter(str_detect(parameter, "systemReact-to-Me")) %>%
  select(
    stratum,
    interaction,
    OR,
    OR_lower_95,
    OR_upper_95,
    p_ordinal = p_value
  )

merge_table <- ordinal_filtered %>%
  left_join(
    nonparam_results %>%
      select(
        stratum,
        n_pairs,
        n_improve,
        pct_improve,
        cles_strict,
        cles_strict_ci_lower,
        cles_strict_ci_upper,
        cles_inclusive,
        cles_inclusive_ci_lower,
        cles_inclusive_ci_upper,
        cliff_delta,
        cohens_d,
        wilcox_p,
        sign_p
      ),
    by = "stratum"
  ) %>%
  left_join(
    holm_results %>%
      select(stratum, test_family, p_holm, significant_holm),
    by = "stratum"
  ) %>%
  left_join(
    power_results %>%
      rename(
        cohens_d_power = cohens_d,
        n_pairs_power = n_pairs
      ),
    by = "stratum"
  ) %>%
  left_join(
    sign_question %>%
      select(
        stratum,
        n_positive,
        n_negative,
        n_zero,
        n_nonzero,
        sign_test_p_two_sided,
        sign_test_p_one_sided,
        median_diff,
        mean_diff
      ),
    by = "stratum"
  ) %>%
  mutate(
    stratum_type = case_when(
      stratum %in% PRIMARY_METRICS ~ "1. Primary (By Metric)",
      stratum %in% QUESTION_TYPES ~ "2. Secondary (By Question Type)",
      stratum == "Overall" ~ "3. Exploratory (Overall)",
      TRUE ~ "4. Exploratory (Metric × Question Type)"
    )
  ) %>%
  arrange(stratum_type, stratum)

write_csv(merge_table, file.path(TABLES_DIR, "COMPREHENSIVE_RESULTS_ALL_STRATIFICATIONS.csv"))

cat("Comprehensive results saved to COMPREHENSIVE_RESULTS_ALL_STRATIFICATIONS.csv\n")

