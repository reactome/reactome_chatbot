#!/usr/bin/env Rscript

# Nonparametric analyses: Wilcoxon, sign tests, and effect sizes.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(effsize)
  library(boot)
})

source("analysis/expert_survey/config/file_paths.R")
source("analysis/expert_survey/config/analysis_parameters.R")

PROCESSED_PAIRED_PATH

cat("Loading paired dataset...\n")
survey_paired <- read_csv(PROCESSED_PAIRED_PATH, show_col_types = FALSE) %>%
  mutate(
    metric = factor(metric, levels = PRIMARY_METRICS),
    question_type = factor(question_type, levels = QUESTION_TYPES),
    participant_id = factor(participant_id),
    question_id = factor(question_id)
  )

# -------------------------------------------------------------------------
# Wilcoxon + effect sizes helper
# -------------------------------------------------------------------------
run_nonparametric_comprehensive <- function(data, stratum_name) {
  cat("\n", strrep("-", 70), "\n")
  cat("Stratum:", stratum_name, "\n")
  cat(strrep("-", 70), "\n")

  wilcox_result <- wilcox.test(
    data$diff_numeric,
    alternative = "greater",
    conf.int = TRUE,
    conf.level = 0.95,
    exact = FALSE
  )

  n_pos <- sum(data$diff_numeric > 0)
  n_neg <- sum(data$diff_numeric < 0)
  n_ties <- sum(data$diff_numeric == 0)
  n_total <- nrow(data)
  n_nonzero <- n_pos + n_neg

  if (n_nonzero > 0) {
    sign_result <- binom.test(n_pos, n_nonzero, p = 0.5, alternative = "greater")
    sign_p <- sign_result$p.value
  } else {
    sign_p <- NA_real_
  }

  cliff_result <- cliff.delta(data$react_score_num, data$base_score_num, paired = TRUE)
  cohens_d <- mean(data$diff_numeric) / sd(data$diff_numeric)

  cles_strict <- n_pos / n_total
  cles_inclusive <- n_pos / n_total + 0.5 * (n_ties / n_total)

  boot_effect_sizes <- function(df, indices) {
    d <- df[indices, ]
    n_wins <- sum(d$diff_numeric > 0)
    n_ties <- sum(d$diff_numeric == 0)
    n_tot <- nrow(d)
    c(
      strict = n_wins / n_tot,
      inclusive = n_wins / n_tot + 0.5 * (n_ties / n_tot)
    )
  }

  boot_result <- boot(
    data,
    statistic = boot_effect_sizes,
    R = 10000,
    strata = data$participant_id
  )

  boot_ci_strict <- tryCatch({
    ci <- boot.ci(boot_result, index = 1, type = "perc", conf = 0.95)
    c(ci$percent[4], ci$percent[5])
  }, error = function(e) {
    quantile(boot_result$t[, 1], c(0.025, 0.975), na.rm = TRUE)
  })

  boot_ci_inclusive <- tryCatch({
    ci <- boot.ci(boot_result, index = 2, type = "perc", conf = 0.95)
    c(ci$percent[4], ci$percent[5])
  }, error = function(e) {
    quantile(boot_result$t[, 2], c(0.025, 0.975), na.rm = TRUE)
  })

  tibble(
    stratum = stratum_name,
    n_pairs = n_total,
    n_improve = n_pos,
    n_worsen = n_neg,
    n_ties = n_ties,
    pct_improve = 100 * n_pos / n_total,
    wilcox_V = as.numeric(wilcox_result$statistic),
    wilcox_p = wilcox_result$p.value,
    hl_estimate = as.numeric(wilcox_result$estimate),
    hl_ci_lower = wilcox_result$conf.int[1],
    hl_ci_upper = wilcox_result$conf.int[2],
    sign_p = sign_p,
    cohens_d = cohens_d,
    cliff_delta = cliff_result$estimate,
    cliff_magnitude = cliff_result$magnitude,
    cles_strict = cles_strict,
    cles_strict_ci_lower = boot_ci_strict[1],
    cles_strict_ci_upper = boot_ci_strict[2],
    cles_inclusive = cles_inclusive,
    cles_inclusive_ci_lower = boot_ci_inclusive[1],
    cles_inclusive_ci_upper = boot_ci_inclusive[2]
  )
}

# -------------------------------------------------------------------------
# Stratifications to evaluate
# -------------------------------------------------------------------------

strata_list <- list(
  "Overall" = survey_paired,
  "Factual accuracy" = filter(survey_paired, metric == "Factual accuracy"),
  "Level of Granularity" = filter(survey_paired, metric == "Level of Granularity"),
  "Relational Depth" = filter(survey_paired, metric == "Relational Depth"),
  "Query" = filter(survey_paired, question_type == "Query"),
  "Reasoning" = filter(survey_paired, question_type == "Reasoning")
)

for (metric_name in PRIMARY_METRICS) {
  for (qtype in QUESTION_TYPES) {
    data_strat <- filter(survey_paired, metric == metric_name, question_type == qtype)
    if (nrow(data_strat) >= 10) {
      stratum_name <- paste0(metric_name, " - ", qtype)
      strata_list[[stratum_name]] <- data_strat
    } else {
      message("Skipping ", metric_name, " - ", qtype, ": insufficient data (n = ", nrow(data_strat), ")")
    }
  }
}

# -------------------------------------------------------------------------
# Run analyses for each stratum
# -------------------------------------------------------------------------

nonparam_results <- map2_dfr(
  strata_list,
  names(strata_list),
  ~ run_nonparametric_comprehensive(.x, .y)
)

write_csv(nonparam_results, file.path(TABLES_DIR, "nonparametric_tests_all_stratifications.csv"))

# -------------------------------------------------------------------------
# Detailed sign-test outputs
# -------------------------------------------------------------------------

run_sign_test_question_level <- function(data, stratum_name) {
  df_wide <- data %>%
    select(participant_id, question_id, metric, question_type, `React-to-Me`, ChatGPT) %>%
    pivot_longer(
      cols = c(`React-to-Me`, ChatGPT),
      names_to = "system",
      values_to = "score"
    ) %>%
    mutate(system = factor(system, levels = c("ChatGPT", "React-to-Me"))) %>%
    pivot_wider(
      id_cols = c(participant_id, question_id, metric, question_type),
      names_from = system,
      values_from = score,
      names_prefix = "score_"
    ) %>%
    filter(!is.na(score_ChatGPT) & !is.na(`score_React-to-Me`)) %>%
    rename(
      score_react = `score_React-to-Me`,
      score_chatgpt = score_ChatGPT
    )

  if (nrow(df_wide) == 0) {
    return(tibble(
      stratum = stratum_name,
      analysis_type = "question_level",
      n_pairs = 0,
      n_positive = 0,
      n_negative = 0,
      n_zero = 0,
      n_nonzero = 0,
      sign_test_statistic = NA_real_,
      sign_test_p_two_sided = NA_real_,
      sign_test_p_one_sided = NA_real_,
      median_diff = NA_real_,
      mean_diff = NA_real_
    ))
  }

  df_wide <- df_wide %>%
    mutate(
      diff = score_react - score_chatgpt,
      diff_sign = case_when(
        diff > 0 ~ "positive",
        diff < 0 ~ "negative",
        TRUE ~ "zero"
      )
    )

  sign_counts <- df_wide %>%
    count(diff_sign) %>%
    pivot_wider(names_from = diff_sign, values_from = n, values_fill = 0)

  n_positive <- sign_counts$positive %||% 0
  n_negative <- if ("negative" %in% names(sign_counts)) sign_counts$negative else 0
  n_zero <- if ("zero" %in% names(sign_counts)) sign_counts$zero else 0
  n_nonzero <- n_positive + n_negative

  median_diff <- median(df_wide$diff)
  mean_diff <- mean(df_wide$diff)

  if (n_nonzero > 0) {
    sign_test_p_two_sided <- binom.test(
      min(n_positive, n_negative),
      n_nonzero,
      p = 0.5,
      alternative = "two.sided"
    )$p.value

    sign_test_p_one_sided <- binom.test(
      n_positive,
      n_nonzero,
      p = 0.5,
      alternative = "greater"
    )$p.value

    sign_test_statistic <- min(n_positive, n_negative)
  } else {
    sign_test_p_two_sided <- NA_real_
    sign_test_p_one_sided <- NA_real_
    sign_test_statistic <- NA_real_
  }

  tibble(
    stratum = stratum_name,
    analysis_type = "question_level",
    n_pairs = nrow(df_wide),
    n_positive = n_positive,
    n_negative = n_negative,
    n_zero = n_zero,
    n_nonzero = n_nonzero,
    sign_test_statistic = sign_test_statistic,
    sign_test_p_two_sided = sign_test_p_two_sided,
    sign_test_p_one_sided = sign_test_p_one_sided,
    median_diff = median_diff,
    mean_diff = mean_diff
  )
}

run_sign_test_participant_level <- function(data, stratum_name) {
  participant_preferences <- data %>%
    mutate(diff = `React-to-Me` - ChatGPT) %>%
    group_by(participant_id) %>%
    summarise(
      n_obs = n(),
      sum_diff = sum(diff),
      mean_diff = mean(diff),
      n_positive = sum(diff > 0),
      n_negative = sum(diff < 0),
      n_zero = sum(diff == 0),
      participant_preference = case_when(
        sum_diff > 0 ~ "positive",
        sum_diff < 0 ~ "negative",
        TRUE ~ "zero"
      ),
      .groups = "drop"
    )

  if (nrow(participant_preferences) == 0) {
    return(tibble(
      stratum = stratum_name,
      analysis_type = "participant_level",
      n_participants = 0,
      n_positive = 0,
      n_negative = 0,
      n_zero = 0,
      sign_test_statistic = NA_real_,
      sign_test_p_two_sided = NA_real_,
      sign_test_p_one_sided = NA_real_
    ))
  }

  pref_counts <- participant_preferences %>%
    count(participant_preference) %>%
    pivot_wider(names_from = participant_preference, values_from = n, values_fill = 0)

  n_positive <- pref_counts$positive %||% 0
  n_negative <- if ("negative" %in% names(pref_counts)) pref_counts$negative else 0
  n_zero <- if ("zero" %in% names(pref_counts)) pref_counts$zero else 0
  n_nonzero <- n_positive + n_negative
  n_participants <- nrow(participant_preferences)

  if (n_nonzero > 0) {
    sign_test_p_two_sided <- binom.test(
      min(n_positive, n_negative),
      n_nonzero,
      p = 0.5,
      alternative = "two.sided"
    )$p.value

    sign_test_p_one_sided <- binom.test(
      n_positive,
      n_nonzero,
      p = 0.5,
      alternative = "greater"
    )$p.value

    sign_test_statistic <- min(n_positive, n_negative)
  } else {
    sign_test_p_two_sided <- NA_real_
    sign_test_p_one_sided <- NA_real_
    sign_test_statistic <- NA_real_
  }

  tibble(
    stratum = stratum_name,
    analysis_type = "participant_level",
    n_participants = n_participants,
    n_positive = n_positive,
    n_negative = n_negative,
    n_zero = n_zero,
    sign_test_statistic = sign_test_statistic,
    sign_test_p_two_sided = sign_test_p_two_sided,
    sign_test_p_one_sided = sign_test_p_one_sided
  )
}

sign_results <- map2(
  strata_list,
  names(strata_list),
  function(data, stratum_name) {
    list(
      question_level = run_sign_test_question_level(data, stratum_name),
      participant_level = run_sign_test_participant_level(data, stratum_name)
    )
  }
)

sign_question_level <- map_dfr(sign_results, ~ .x$question_level, .id = "stratum_id") %>%
  select(-stratum_id)
sign_participant_level <- map_dfr(sign_results, ~ .x$participant_level, .id = "stratum_id") %>%
  select(-stratum_id)

write_csv(sign_question_level, file.path(TABLES_DIR, "sign_test_question_level.csv"))
write_csv(sign_participant_level, file.path(TABLES_DIR, "sign_test_participant_level.csv"))

cat("Nonparametric analysis complete.\n")
cat("Results written to:\n")
cat(" - nonparametric_tests_all_stratifications.csv\n")
cat(" - sign_test_question_level.csv\n")
cat(" - sign_test_participant_level.csv\n")

