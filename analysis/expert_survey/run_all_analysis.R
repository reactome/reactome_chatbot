#!/usr/bin/env Rscript

# Orchestrate expert survey analysis pipeline (scripts 00–08).

scripts <- sprintf(
  "analysis/expert_survey/scripts/%02d_%s.R",
  0:8,
  c(
    "setup",
    "descriptives",
    "ordinal_models",      # note: placeholder, actual 02 skipped (reliability)
    "ordinal_models",
    "nonparametrics",
    "multiple_testing",
    "power_analysis",
    "results_compilation",
    "tables_and_text"
  )
)

# Adjust script names to match existing files
scripts <- c(
  "analysis/expert_survey/scripts/00_setup.R",
  "analysis/expert_survey/scripts/01_descriptives.R",
  "analysis/expert_survey/scripts/03_ordinal_models.R",
  "analysis/expert_survey/scripts/04_nonparametrics.R",
  "analysis/expert_survey/scripts/05_multiple_testing.R",
  "analysis/expert_survey/scripts/06_power_analysis.R",
  "analysis/expert_survey/scripts/07_results_compilation.R",
  "analysis/expert_survey/scripts/08_tables_and_text.R"
)

for (script in scripts) {
  message("\n=== Running ", script, " ===")
  tryCatch(
    source(script, echo = TRUE, max.deparse.length = Inf),
    error = function(e) {
      stop(sprintf("Error running %s: %s", script, e$message), call. = FALSE)
    }
  )
}

message("\nAll analysis steps completed successfully.")

