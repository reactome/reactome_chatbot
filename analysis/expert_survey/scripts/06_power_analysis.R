#!/usr/bin/env Rscript

# Post-hoc power analysis for each stratum using paired t-test approximation.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(pwr)
})

source("analysis/expert_survey/config/file_paths.R")
source("analysis/expert_survey/config/analysis_parameters.R")

nonparam_results <- read_csv(
  file.path(TABLES_DIR, "nonparametric_tests_all_stratifications.csv"),
  show_col_types = FALSE
)

compute_power <- function(n, d) {
  if (is.na(d) || n <= 2 || d == 0) {
    return(NA_real_)
  }

  res <- tryCatch(
    pwr.t.test(
      n = n,
      d = abs(d),
      sig.level = ALPHA_ONESIDED,
      type = "paired",
      alternative = "greater"
    ),
    error = function(e) NULL
  )

  if (is.null(res)) {
    return(NA_real_)
  }

  res$power
}

power_results <- nonparam_results %>%
  mutate(
    achieved_power = map2_dbl(n_pairs, cohens_d, compute_power)
  ) %>%
  select(stratum, n_pairs, cohens_d, achieved_power)

write_csv(power_results, file.path(TABLES_DIR, "power_analysis_all_stratifications.csv"))

cat("Power analysis results saved to power_analysis_all_stratifications.csv\n")

