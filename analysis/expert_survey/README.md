# Expert Survey Analysis Pipeline

This directory contains a modular R workflow that reproduces the expert survey evaluation used to compare ChatGPT (baseline) vs React-to-Me (intervention) across three expert-rated metrics:

- Factual accuracy  
- Level of Granularity  
- Relational Depth  

Each participant rated 15 questions (9 Query, 6 Reasoning) on a 4-point ordinal scale, yielding 450 paired comparisons (900 total observations). Place the evaluation data at `analysis/expert_survey/survey_results.csv` with the following columns:

| Column               | Description                                                      |
|----------------------|------------------------------------------------------------------|
| `participant_id`     | Participant identifier (string)                                  |
| `question_id`        | Question identifier (numeric)                                    |
| `question_type`      | `Query` or `Reasoning`                                           |
| `metric`             | `Factual accuracy`, `Level of Granularity`, `Relational Depth`   |
| `React-to-Me score`  | React-to-Me rating (integer 1–4)                                 |
| `GPT-baseline score` | ChatGPT rating (integer 1–4)                                     |

## Running the Pipeline

```bash
poetry run Rscript analysis/expert_survey/run_all_analysis.R
```

This orchestrates all numbered scripts (`00_setup.R` → `08_tables_and_text.R`) and writes outputs under `analysis/expert_survey/results/`. Install dependencies first:

```r
install.packages(c(
  "tidyverse",
  "ordinal",
  "lme4",
  "lmerTest",
  "effsize",
  "irr",
  "boot",
  "broom",
  "broom.mixed",
  "pwr"
))
```

The workflow has been developed and tested with **R 4.3.x** (latest 4.x release); earlier versions may fail to install one or more packages. If you prefer to run outside Poetry, call the scripts with `Rscript` after installing these packages.

## Processing & Stratifications

- **00_setup.R**: Loads the reference dataset (`survey_results.csv`), trims whitespace, validates counts, converts to long format, and creates a paired wide dataset with score differences.
- **01_descriptives.R**: Generates descriptive tables:
  - Score distributions by system × metric × score.
  - Summary statistics by system, by metric, and stratified by question type.
  - Paired-difference summaries overall and by question type.
- **03_ordinal_models.R**: Fits cumulative link mixed models (CLMM) with logit link, flexible thresholds, and random intercept/slope per participant and question.
  - Stratifications: overall, by metric, by question type, and metric × question type.
  - Interaction models (`system * question_type`) are attempted for each metric; likelihood-ratio tests are logged but skipped if they cannot be evaluated.
- **04_nonparametrics.R**:
  - Wilcoxon signed-rank tests (one-sided, React-to-Me > ChatGPT) with Hodges–Lehmann estimates and confidence intervals.
  - Binomial sign tests at question-level and participant-level.
  - Effect sizes: Cliff’s Delta, Cohen’s d, strict/common-language effect size (CLES) with bootstrap CIs.
  - Stratifications mirror the CLMM step (overall, metrics, question types, metric × question type).

## Inferential Adjustments

- **05_multiple_testing.R**: Holm corrections applied separately to:
  - Primary family: three per-metric comparisons.
  - Secondary family: two question-type comparisons.
  - Exploratory family: remaining strata (metric × question type, overall) reported without adjustment.
- **06_power_analysis.R**: Post-hoc power estimates using `pwr.t.test` for paired differences (Cohen’s d, α = 0.05 one-sided).

## Results Assembly & Publication Outputs

- **07_results_compilation.R**: Merges CLMM odds ratios, nonparametric results, Holm adjustments, power, and sign-test summaries into `COMPREHENSIVE_RESULTS_ALL_STRATIFICATIONS.csv`. Each row is labeled as:
  - `1. Primary (By Metric)`
  - `2. Secondary (By Question Type)`
  - `3. Exploratory (Overall)`
  - `4. Exploratory (Metric × Question Type)`
- **08_tables_and_text.R**: Produces publication-ready assets:
  - `TABLE_MAIN_RESULTS_FORMATTED.csv` (primary metrics with one-sided p-values and Holm-adjusted p-values).
  - `TABLE_STRATIFIED_RESULTS_FORMATTED.csv` (secondary and exploratory strata).
  - `KEY_STATISTICS_FOR_TEXT.txt` (concise summary lines per stratum).

## Statistical Methods Summary

- **Ordinal Mixed Models (CLMM)**: `ordinal::clmm`, logit link, flexible thresholds, random intercepts and system-specific slopes at participant and question levels.
- **Nonparametric Tests**: One-sided Wilcoxon signed-rank, sign tests (question-level and participant-level), bootstrap effect-size CIs.
- **Effect Sizes**: Odds ratios with Wald CIs, Cliff’s Delta, Cohen’s d, strict/inclusive common language effect sizes.
- **Multiple Testing**: Holm correction by analysis family.
- **Power Analysis**: Paired t-test approximation via Cohen’s d.

All outputs are written under `analysis/expert_survey/results/` with separate folders for tables, stratified summaries, raw processed data, and sensitivity analyses. The pipeline is self-contained; rerunning it will regenerate every table from scratch given the reference dataset.

## Key Outputs

| File/Folder                                                           | Description                                                       |
|-----------------------------------------------------------------------|-------------------------------------------------------------------|
| `results/raw_data/survey_long.csv` / `paired_differences.csv`         | Processed datasets (long-form and paired differences)             |
| `results/tables/summary_statistics_*.csv`                             | Descriptive summaries                                             |
| `results/tables/ordinal_models_all_stratifications.csv`               | CLMM odds ratios and p-values                                     |
| `results/tables/nonparametric_tests_all_stratifications.csv`          | Wilcoxon results, effect sizes, sign-test p-values                |
| `results/tables/multiple_testing_correction_comprehensive.csv`        | Holm-adjusted p-values by family                                  |
| `results/tables/power_analysis_all_stratifications.csv`               | Post-hoc power estimates                                          |
| `results/tables/sign_test_question_level.csv` / `_participant_level.csv` | Sign-test summaries                                             |
| `results/tables/COMPREHENSIVE_RESULTS_ALL_STRATIFICATIONS.csv`        | Unified table combining all statistics                            |
| `results/tables/TABLE_MAIN_RESULTS_FORMATTED.csv`                     | Publication-ready table for primary metrics                       |
| `results/tables/TABLE_STRATIFIED_RESULTS_FORMATTED.csv`               | Secondary and exploratory formatted table                         |
| `results/tables/KEY_STATISTICS_FOR_TEXT.txt`                          | Key statistics lines for manuscript text                          |

