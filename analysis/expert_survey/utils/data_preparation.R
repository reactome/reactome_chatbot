# Data Preparation Utilities for Expert Survey Analysis

library(readr)
library(dplyr)
library(tidyr)
library(stringr)

#' Load and validate expert survey data
#'
#' @param file_path Path to the survey CSV.
#' @return A tibble with cleaned column names and string values trimmed.
load_and_validate_data <- function(file_path) {
  df <- read_csv(
    file_path,
    col_types = cols(
      participant_id = col_character(),
      question_id = col_double(),
      question_type = col_character(),
      metric = col_character(),
      `React-to-Me score` = col_double(),
      `GPT-baseline score` = col_double()
    )
  )

  df <- df %>%
    mutate(
      participant_id = str_trim(participant_id),
      question_type = str_trim(question_type),
      metric = str_trim(metric)
    ) %>%
    rename(
      react_score = `React-to-Me score`,
      baseline_score = `GPT-baseline score`
    )

  expected_participants <- 10
  expected_questions <- 15
  expected_metrics <- 3
  expected_systems <- 2

  n_participants <- n_distinct(df$participant_id)
  n_questions <- n_distinct(df$question_id)
  n_metrics <- n_distinct(df$metric)

  if (n_participants != expected_participants) {
    stop(
      sprintf(
        "Unexpected number of participants: %s (expected %s)",
        n_participants,
        expected_participants
      )
    )
  }

  if (n_questions != expected_questions) {
    stop(
      sprintf(
        "Unexpected number of questions: %s (expected %s)",
        n_questions,
        expected_questions
      )
    )
  }

  if (n_metrics != expected_metrics) {
    stop(
      sprintf(
        "Unexpected number of metrics: %s (expected %s)",
        n_metrics,
        expected_metrics
      )
    )
  }

  if (any(df$react_score < 1 | df$react_score > 4, na.rm = TRUE) ||
      any(df$baseline_score < 1 | df$baseline_score > 4, na.rm = TRUE)) {
    stop("Scores must be within the 1-4 scale.")
  }

  df
}

#' Convert cleaned data to long format for modeling
#'
#' @param df Cleaned wide-format tibble.
#' @return Long-format tibble with system and score columns.
to_long_format <- function(df) {
  df_long <- df %>%
    pivot_longer(
      cols = c(react_score, baseline_score),
      names_to = "system",
      values_to = "score"
    ) %>%
    mutate(
      system = recode(
        system,
        react_score = "React-to-Me",
        baseline_score = "ChatGPT"
      ),
      system = factor(system, levels = c("ChatGPT", "React-to-Me")),
      metric = factor(
        metric,
        levels = c("Factual accuracy", "Level of Granularity", "Relational Depth")
      ),
      question_type = factor(question_type, levels = c("Query", "Reasoning")),
      participant_id = factor(participant_id),
      question_id = factor(question_id)
    )

  df_long
}

#' Create paired dataset for nonparametric analyses
#'
#' @param df_long Long-format tibble produced by to_long_format().
#' @return Tibble with paired scores and difference columns.
create_paired_dataset <- function(df_long) {
  df_wide <- df_long %>%
    select(participant_id, question_id, question_type, metric, system, score) %>%
    pivot_wider(
      names_from = system,
      values_from = score
    ) %>%
    mutate(
      diff_numeric = `React-to-Me` - ChatGPT,
      react_score_num = `React-to-Me`,
      base_score_num = ChatGPT,
      participant_num = as.numeric(participant_id),
      question_num = as.numeric(question_id)
    )

  df_wide
}

