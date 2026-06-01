
# ============================================================
# Match Outcome Variable Significance Test
# Multinomial Logistic Regression for H / D / A
# File: match_outcome_variable_significance.R
# ============================================================

# -----------------------------
# 1. Install/load packages
# -----------------------------

packages <- c("tidyverse", "nnet", "broom")

installed <- rownames(installed.packages())

for (pkg in packages) {
  if (!(pkg %in% installed)) {
    install.packages(pkg)
  }
}

library(tidyverse)
library(nnet)
library(broom)

# -----------------------------
# 2. Load data
# -----------------------------
# Choose train_model_data.csv when the file picker opens.

df <- read.csv(file.choose())

cat("All columns:\n")
print(colnames(df))

# -----------------------------
# 3. Prepare outcome variable
# -----------------------------

if (!("FTR" %in% colnames(df))) {
  stop("Column FTR is missing. Please choose train_model_data.csv, not test_predictions.csv.")
}

df$FTR <- as.factor(df$FTR)

# Set Draw as baseline if Draw exists.
# This makes coefficients compare H vs D and A vs D.
if ("D" %in% levels(df$FTR)) {
  df$FTR <- relevel(df$FTR, ref = "D")
} else {
  warning("No Draw class found in FTR. Using default baseline class.")
}

# -----------------------------
# 4. Automatically choose feature columns
# -----------------------------

exclude_cols <- c(
  "Date",
  "HomeTeam",
  "AwayTeam",
  "FTHG",
  "FTAG",
  "FTR",
  "PredictedResult",
  "PredictedResult_Original",
  "PredictedResult_Threshold",
  "P_HomeWin",
  "P_Draw",
  "P_AwayWin",
  "ProbHomeWin",
  "ProbDraw",
  "ProbAwayWin",
  "lambda_home",
  "lambda_away",
  "LambdaHome",
  "LambdaAway",
  "draw_margin_used"
)

# Use numeric model features automatically.
numeric_features <- df %>%
  select(where(is.numeric)) %>%
  select(-any_of(exclude_cols)) %>%
  colnames()

features <- numeric_features

# Add League as categorical variable if it exists.
if ("League" %in% colnames(df)) {
  df$League <- as.factor(df$League)
  features <- c(features, "League")
}

cat("\nFeatures used:\n")
print(features)

cat("\nNumber of features:", length(features), "\n")

if (length(features) == 0) {
  stop("No usable feature columns found. Check that you selected train_model_data.csv.")
}

# -----------------------------
# 5. Build modelling dataframe
# -----------------------------

model_df <- df %>%
  select(FTR, all_of(features)) %>%
  drop_na()

cat("\nRows used:", nrow(model_df), "\n")
cat("\nOutcome distribution:\n")
print(table(model_df$FTR))

if (nrow(model_df) < 30) {
  warning("Very few rows available after drop_na(). Results may be unstable.")
}

# -----------------------------
# 6. Fit multinomial logistic regression
# -----------------------------

formula_outcome <- as.formula(
  paste("FTR ~", paste(features, collapse = " + "))
)

outcome_model <- multinom(
  formula_outcome,
  data = model_df,
  trace = FALSE,
  maxit = 1000
)

cat("\n===== MULTINOMIAL LOGISTIC REGRESSION SUMMARY =====\n")
print(summary(outcome_model))

# -----------------------------
# 7. Extract coefficients, z-scores, p-values
# -----------------------------

model_summary <- summary(outcome_model)

coefs <- model_summary$coefficients
std_errors <- model_summary$standard.errors

# If there are only two classes, nnet::multinom may return a vector instead of matrix.
# Convert to matrix so the rest of the code still works.
if (is.null(dim(coefs))) {
  coefs <- matrix(coefs, nrow = 1)
  rownames(coefs) <- levels(model_df$FTR)[levels(model_df$FTR) != levels(model_df$FTR)[1]][1]
}

if (is.null(dim(std_errors))) {
  std_errors <- matrix(std_errors, nrow = 1)
  rownames(std_errors) <- rownames(coefs)
}

z_values <- coefs / std_errors
p_values <- 2 * (1 - pnorm(abs(z_values)))

sig_table <- data.frame()

baseline_class <- levels(model_df$FTR)[1]

for (outcome_class in rownames(coefs)) {
  temp <- data.frame(
    OutcomeComparison = paste(outcome_class, "vs", baseline_class),
    Variable = colnames(coefs),
    Coefficient = as.numeric(coefs[outcome_class, ]),
    StdError = as.numeric(std_errors[outcome_class, ]),
    ZValue = as.numeric(z_values[outcome_class, ]),
    PValue = as.numeric(p_values[outcome_class, ]),
    OddsRatio = exp(as.numeric(coefs[outcome_class, ]))
  )

  sig_table <- bind_rows(sig_table, temp)
}

sig_table <- sig_table %>%
  arrange(PValue) %>%
  mutate(
    Significant_1pct = ifelse(PValue < 0.01, "Yes", "No"),
    Significant_5pct = ifelse(PValue < 0.05, "Yes", "No"),
    Significant_10pct = ifelse(PValue < 0.10, "Yes", "No")
  )

cat("\n===== MATCH OUTCOME VARIABLE SIGNIFICANCE =====\n")
print(as_tibble(sig_table), n = Inf)

# -----------------------------
# 8. Show significant variables only
# -----------------------------

top_sig <- sig_table %>%
  filter(PValue < 0.10) %>%
  arrange(PValue)

cat("\n===== TOP SIGNIFICANT VARIABLES, P < 0.10 =====\n")

if (nrow(top_sig) == 0) {
  cat("No variables are significant at the 10% level.\n")
} else {
  print(as_tibble(top_sig), n = Inf)
}

# -----------------------------
# 9. Save results
# -----------------------------

write.csv(sig_table, "match_outcome_variable_significance.csv", row.names = FALSE)
write.csv(top_sig, "top_match_outcome_significant_variables.csv", row.names = FALSE)

cat("\nSaved:\n")
cat("match_outcome_variable_significance.csv\n")
cat("top_match_outcome_significant_variables.csv\n")
