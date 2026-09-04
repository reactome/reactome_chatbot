#!/usr/bin/env Rscript

# Publication-ready tables and key text snippets (no figures).

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
})

source("analysis/expert_survey/config/file_paths.R")
source("analysis/expert_survey/config/analysis_parameters.R")

comprehensive_results <- read_csv(
  file.path(TABLES_DIR, "COMPREHENSIVE_RESULTS_ALL_STRATIFICATIONS.csv"),
  show_col_types = FALSE
)

question_strata <- filter(comprehensive_results, stratum %in% QUESTION_TYPES)

main_results_formatted <- comprehensive_results %>%
  filter(stratum_type == "1. Primary (By Metric)") %>%
  mutate(
    p_onesided = p_ordinal / 2,
    p_holm_onesided = if_else(!is.na(p_holm), p_holm / 2, NA_real_),
    `Odds Ratio (95% CI)` = sprintf("%.2f (%.2f-%.2f)", OR, OR_lower_95, OR_upper_95),
    `P-value (one-sided)‡` = format.pval(p_onesided, digits = 3, eps = 0.001),
    `Adjusted P†` = if_else(
      !is.na(p_holm_onesided),
      format.pval(p_holm_onesided, digits = 3, eps = 0.001),
      "\u2014"
    ),
    `Strict Win Rate (95% CI)` = sprintf(
      "%.2f (%.2f-%.2f)",
      cles_strict,
      cles_strict_ci_lower,
      cles_strict_ci_upper
    ),
    `CLES with Ties (95% CI)` = sprintf(
      "%.2f (%.2f-%.2f)",
      cles_inclusive,
      cles_inclusive_ci_lower,
      cles_inclusive_ci_upper
    ),
    `% Improved` = sprintf("%.1f%%", pct_improve),
    `Superiority Claim` = if_else(p_onesided < ALPHA_ONESIDED & OR > 1, "Supported", "Not Supported")
  ) %>%
  select(
    Stratum = stratum,
    `N Pairs` = n_pairs,
    `% Improved`,
    `Odds Ratio (95% CI)`,
    `P-value (one-sided)‡`,
    `Adjusted P†`,
    `Strict Win Rate (95% CI)`,
    `CLES with Ties (95% CI)`,
    `Superiority Claim`
  )

write_csv(main_results_formatted, file.path(TABLES_DIR, "TABLE_MAIN_RESULTS_FORMATTED.csv"))

stratified_results_formatted <- comprehensive_results %>%
  filter(stratum_type %in% c("2. Secondary (By Question Type)", "3. Exploratory (Overall)", "4. Exploratory (Metric × Question Type)")) %>%
  mutate(
    `Odds Ratio (95% CI)` = sprintf("%.2f (%.2f-%.2f)", OR, OR_lower_95, OR_upper_95),
    `P-value` = format.pval(p_ordinal, digits = 3, eps = 0.001),
    `Strict Win Rate (95% CI)` = sprintf(
      "%.2f (%.2f-%.2f)",
      cles_strict,
      cles_strict_ci_lower,
      cles_strict_ci_upper
    ),
    `CLES with Ties (95% CI)` = sprintf(
      "%.2f (%.2f-%.2f)",
      cles_inclusive,
      cles_inclusive_ci_lower,
      cles_inclusive_ci_upper
    ),
    `% Improved` = sprintf("%.1f%%", pct_improve)
  ) %>%
  select(
    Category = stratum_type,
    Stratum = stratum,
    `N Pairs` = n_pairs,
    `% Improved`,
    `Odds Ratio (95% CI)`,
    `P-value`,
    `Strict Win Rate (95% CI)`,
    `CLES with Ties (95% CI)`,
    `Analysis Type` = test_family
  )

write_csv(
  stratified_results_formatted,
  file.path(TABLES_DIR, "TABLE_STRATIFIED_RESULTS_FORMATTED.csv")
)

key_stats <- comprehensive_results %>%
  mutate(
    p_onesided = p_ordinal / 2,
    summary_line = case_when(
      stratum %in% PRIMARY_METRICS ~ sprintf(
        "%s: OR = %.2f [%.2f-%.2f], p(one-sided) = %s (Holm adj: %s), CLES = %.2f, Improvement = %.1f%%",
        stratum,
        OR,
        OR_lower_95,
        OR_upper_95,
        format.pval(p_onesided, digits = 3),
        format.pval(p_holm / 2, digits = 3),
        cles_strict,
        pct_improve
      ),
      stratum %in% QUESTION_TYPES ~ sprintf(
        "%s questions: OR = %.2f [%.2f-%.2f], p = %s, CLES = %.2f, Improvement = %.1f%%",
        stratum,
        OR,
        OR_lower_95,
        OR_upper_95,
        format.pval(p_ordinal, digits = 3),
        cles_strict,
        pct_improve
      ),
      stratum == "Overall" ~ sprintf(
        "Overall: OR = %.2f [%.2f-%.2f], p = %s, CLES = %.2f, Improvement = %.1f%%",
        OR,
        OR_lower_95,
        OR_upper_95,
        format.pval(p_ordinal, digits = 3),
        cles_strict,
        pct_improve
      ),
      TRUE ~ sprintf(
        "%s: OR = %.2f [%.2f-%.2f], p = %s, CLES = %.2f, Improvement = %.1f%%",
        stratum,
        OR,
        OR_lower_95,
        OR_upper_95,
        format.pval(p_ordinal, digits = 3),
        cles_strict,
        pct_improve
      )
    )
  ) %>%
  pull(summary_line)

writeLines(key_stats, file.path(TABLES_DIR, "KEY_STATISTICS_FOR_TEXT.txt"))

cat("Publication tables saved:\n")
cat(" - TABLE_MAIN_RESULTS_FORMATTED.csv\n")
cat(" - TABLE_STRATIFIED_RESULTS_FORMATTED.csv\n")
cat(" - KEY_STATISTICS_FOR_TEXT.txt\n")

