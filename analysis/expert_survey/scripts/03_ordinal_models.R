#!/usr/bin/env Rscript

# Ordinal mixed-effects models across stratifications.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(ordinal)
})

source("analysis/expert_survey/config/file_paths.R")
source("analysis/expert_survey/config/analysis_parameters.R")

# -------------------------------------------------------------------------
# Helper to fit CLMM and extract odds ratios
# -------------------------------------------------------------------------
fit_and_extract_ordinal <- function(data, stratum_name, test_interaction = FALSE) {
  message("\n--- Fitting model for: ", stratum_name, " ---")

  data <- droplevels(data)
  data$system <- relevel(data$system, ref = "ChatGPT")

  formula_str <- if (test_interaction) {
    "score ~ system * question_type + (1 + system|participant_id) + (1 + system|question_id)"
  } else {
    "score ~ system + (1 + system|participant_id) + (1 + system|question_id)"
  }

  message("Formula: ", formula_str)

  model <- tryCatch(
    clmm(
      as.formula(formula_str),
      data = data,
      link = "logit",
      threshold = "flexible"
    ),
    error = function(e) {
      message("Model failed for ", stratum_name, ": ", e$message)
      return(NULL)
    }
  )

  if (is.null(model)) {
    return(NULL)
  }

  message("Convergence: ", model$convergence$code == 0)

  coef_df <- tryCatch(
    {
      tmp <- as.data.frame(summary(model)$coefficients)
      tmp$parameter <- rownames(tmp)
      tmp
    },
    error = function(e) {
      message("Coefficient extraction failed for ", stratum_name, ": ", e$message)
      return(NULL)
    }
  )

  if (is.null(coef_df)) {
    return(NULL)
  }

  results <- coef_df %>%
    as_tibble() %>%
    filter(grepl("system", parameter)) %>%
    mutate(
      OR = exp(Estimate),
      OR_lower_95 = exp(Estimate - 1.96 * `Std. Error`),
      OR_upper_95 = exp(Estimate + 1.96 * `Std. Error`),
      stratum = stratum_name,
      interaction = test_interaction
    ) %>%
    select(
      stratum,
      interaction,
      parameter,
      Estimate,
      `Std. Error`,
      `z value`,
      p_value = `Pr(>|z|)`,
      OR,
      OR_lower_95,
      OR_upper_95
    )

  list(model = model, results = results)
}

# -------------------------------------------------------------------------
# Load processed data
# -------------------------------------------------------------------------

cat("Loading long-format dataset...\n")
survey_long <- read_csv(PROCESSED_LONG_PATH, show_col_types = FALSE) %>%
  mutate(
    system = factor(system, levels = c("ChatGPT", "React-to-Me")),
    metric = factor(metric, levels = PRIMARY_METRICS),
    question_type = factor(question_type, levels = QUESTION_TYPES),
    participant_id = factor(participant_id),
    question_id = factor(question_id),
    score = factor(score, levels = 1:4, ordered = TRUE)
  )

# -------------------------------------------------------------------------
# Overall model
# -------------------------------------------------------------------------

ordinal_results <- list()

overall_model <- fit_and_extract_ordinal(survey_long, "Overall", test_interaction = FALSE)
ordinal_results[["Overall"]] <- overall_model$results

# -------------------------------------------------------------------------
# By metric (with interaction tests)
# -------------------------------------------------------------------------

metric_models <- lapply(PRIMARY_METRICS, function(metric_name) {
  data_met <- filter(survey_long, metric == metric_name)

  main_model <- fit_and_extract_ordinal(data_met, metric_name, test_interaction = FALSE)
  if (is.null(main_model)) {
    message("Main model unavailable for ", metric_name, "; skipping metric.")
    return(list(metric = metric_name, main = NULL, interaction = NULL))
  }

  int_model <- fit_and_extract_ordinal(
    data_met,
    paste0(metric_name, " (interaction)"),
    test_interaction = TRUE
  )

  interaction_sig <- FALSE
  if (!is.null(int_model)) {
    lr_result <- tryCatch(
      anova(main_model$model, int_model$model),
      error = function(e) {
        message("LR test failed for ", metric_name, ": ", e$message)
        NULL
      }
    )
    if (!is.null(lr_result) && nrow(lr_result) >= 2) {
      interaction_sig <- lr_result$`Pr(>Chisq)`[2] < 0.05
    }
  }

  list(
    metric = metric_name,
    main = main_model$results,
    interaction = if (interaction_sig && !is.null(int_model)) int_model$results else NULL
  )
})

for (mm in metric_models) {
  if (is.null(mm$main)) next
  ordinal_results[[mm$metric]] <- mm$main
  if (!is.null(mm$interaction)) {
    ordinal_results[[paste0(mm$metric, " (interaction)")]] <- mm$interaction
  }
}

# -------------------------------------------------------------------------
# By question type
# -------------------------------------------------------------------------

for (qtype in QUESTION_TYPES) {
  data_qtype <- filter(survey_long, question_type == qtype)
  q_model <- fit_and_extract_ordinal(data_qtype, qtype, test_interaction = FALSE)
  if (!is.null(q_model)) {
    ordinal_results[[qtype]] <- q_model$results
  }
}

# -------------------------------------------------------------------------
# Metric × question type combinations
# -------------------------------------------------------------------------

for (metric_name in PRIMARY_METRICS) {
  for (qtype in QUESTION_TYPES) {
    data_strat <- filter(survey_long, metric == metric_name, question_type == qtype)

    if (nrow(data_strat) < 10) {
      message("Skipping ", metric_name, " - ", qtype, ": insufficient data (n = ", nrow(data_strat), ")")
      next
    }

    stratum_name <- paste0(metric_name, " - ", qtype)
    strat_model <- fit_and_extract_ordinal(data_strat, stratum_name, test_interaction = FALSE)
    if (!is.null(strat_model)) {
      ordinal_results[[stratum_name]] <- strat_model$results
    }
  }
}

# -------------------------------------------------------------------------
# Combine and write results
# -------------------------------------------------------------------------

ordinal_results_table <- bind_rows(ordinal_results)
write_csv(ordinal_results_table, file.path(TABLES_DIR, "ordinal_models_all_stratifications.csv"))

cat("Ordinal model results written to ordinal_models_all_stratifications.csv\n")

