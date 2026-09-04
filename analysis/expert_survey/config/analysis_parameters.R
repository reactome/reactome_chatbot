# Analysis constants for expert survey workflow

ALPHA_ONESIDED <- 0.05

PRIMARY_METRICS <- c(
  "Factual accuracy",
  "Level of Granularity",
  "Relational Depth"
)

QUESTION_TYPES <- c("Query", "Reasoning")

NI_MARGINS <- list(
  OR_margin = 1.10,
  win_prob_margin = 0.55,
  median_diff_margin = 0.25
)

