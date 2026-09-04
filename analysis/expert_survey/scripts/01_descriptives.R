#!/usr/bin/env Rscript

# Descriptive statistics and summary tables.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
})

source("analysis/expert_survey/config/file_paths.R")
source("analysis/expert_survey/config/analysis_parameters.R")

cat("Loading processed datasets...\n")
survey_long <- read_csv(PROCESSED_LONG_PATH, show_col_types = FALSE) %>%
  mutate(
    system = factor(system, levels = c("ChatGPT", "React-to-Me")),
    metric = factor(metric, levels = PRIMARY_METRICS),
    question_type = factor(question_type, levels = QUESTION_TYPES),
    participant_id = factor(participant_id),
    question_id = factor(question_id)
  )

survey_paired <- read_csv(PROCESSED_PAIRED_PATH, show_col_types = FALSE) %>%
  mutate(
    metric = factor(metric, levels = PRIMARY_METRICS),
    question_type = factor(question_type, levels = QUESTION_TYPES),
    participant_id = factor(participant_id),
    question_id = factor(question_id)
  )

# Score distributions -----------------------------------------------------------
cat("Generating score distribution tables...\n")
score_dist <- survey_long %>%
  group_by(system, metric, score) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(system, metric) %>%
  mutate(
    total = sum(n),
    percent = 100 * n / total
  ) %>%
  arrange(system, metric, score)

write_csv(score_dist, file.path(TABLES_DIR, "score_distributions.csv"))

# Summary statistics by system and metric ---------------------------------------
summary_stats <- survey_long %>%
  mutate(score_num = as.numeric(score)) %>%
  group_by(system, metric) %>%
  summarise(
    n = n(),
    mean = mean(score_num),
    sd = sd(score_num),
    median = median(score_num),
    q25 = quantile(score_num, 0.25),
    q75 = quantile(score_num, 0.75),
    pct_ge3 = 100 * mean(score_num >= 3),
    .groups = "drop"
  ) %>%
  arrange(metric, system)

write_csv(summary_stats, file.path(TABLES_DIR, "summary_statistics_by_system.csv"))

# Stratified summary by question type -------------------------------------------
summary_stats_stratified <- survey_long %>%
  mutate(score_num = as.numeric(score)) %>%
  group_by(system, metric, question_type) %>%
  summarise(
    n = n(),
    mean = mean(score_num),
    sd = sd(score_num),
    median = median(score_num),
    pct_ge3 = 100 * mean(score_num >= 3),
    .groups = "drop"
  ) %>%
  arrange(metric, question_type, system)

write_csv(
  summary_stats_stratified,
  file.path(TABLES_DIR, "summary_statistics_stratified.csv")
)

# Paired differences summary ----------------------------------------------------
diff_summary <- survey_paired %>%
  group_by(metric) %>%
  summarise(
    n_pairs = n(),
    n_improve = sum(diff_numeric > 0),
    n_worsen = sum(diff_numeric < 0),
    n_ties = sum(diff_numeric == 0),
    pct_improve = 100 * n_improve / n_pairs,
    pct_worsen = 100 * n_worsen / n_pairs,
    pct_ties = 100 * n_ties / n_pairs,
    mean_diff = mean(diff_numeric),
    sd_diff = sd(diff_numeric),
    median_diff = median(diff_numeric),
    .groups = "drop"
  ) %>%
  arrange(metric)

write_csv(diff_summary, file.path(TABLES_DIR, "paired_differences_summary.csv"))

# Paired differences by question type ------------------------------------------
diff_summary_stratified <- survey_paired %>%
  group_by(metric, question_type) %>%
  summarise(
    n_pairs = n(),
    n_improve = sum(diff_numeric > 0),
    n_worsen = sum(diff_numeric < 0),
    n_ties = sum(diff_numeric == 0),
    pct_improve = 100 * n_improve / n_pairs,
    mean_diff = mean(diff_numeric),
    median_diff = median(diff_numeric),
    .groups = "drop"
  ) %>%
  arrange(metric, question_type)

write_csv(
  diff_summary_stratified,
  file.path(STRATIFIED_DIR, "paired_differences_by_question_type.csv")
)

cat("Descriptive tables created:\n")
cat(" - score_distributions.csv\n")
cat(" - summary_statistics_by_system.csv\n")
cat(" - summary_statistics_stratified.csv\n")
cat(" - paired_differences_summary.csv\n")
cat(" - paired_differences_by_question_type.csv\n")
cat("Descriptive analysis complete.\n")

