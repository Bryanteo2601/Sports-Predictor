
# ============================================================
# Robustness Test for Supervised Expected Goals Model
# File: robustness_test_expected_goals.R
# Purpose:
#   Read Python model outputs from test_predictions.csv and test robustness.
# ============================================================

# -----------------------------
# 1. Install/load packages
# -----------------------------
packages <- c("tidyverse", "ggplot2")

installed <- rownames(installed.packages())

for (pkg in packages) {
  if (!(pkg %in% installed)) {
    install.packages(pkg)
  }
}

library(tidyverse)
library(ggplot2)

# -----------------------------
# 2. Load Python test predictions
# -----------------------------
# Put this R script in the same folder as test_predictions.csv
df <- read.csv("test_predictions.csv")

cat("Columns in test_predictions.csv:\n")
print(colnames(df))

# -----------------------------
# 3. Standardise column names
# -----------------------------
# This makes the script work even if your Python file used slightly different names.

rename_if_exists <- function(data, old_name, new_name) {
  if (old_name %in% names(data) && !(new_name %in% names(data))) {
    names(data)[names(data) == old_name] <- new_name
  }
  return(data)
}

df <- rename_if_exists(df, "lambda_home", "LambdaHome")
df <- rename_if_exists(df, "lambda_away", "LambdaAway")
df <- rename_if_exists(df, "PredictedResult", "PredictedResult")
df <- rename_if_exists(df, "predicted_result", "PredictedResult")
df <- rename_if_exists(df, "P_HomeWin", "ProbHomeWin")
df <- rename_if_exists(df, "P_Draw", "ProbDraw")
df <- rename_if_exists(df, "P_AwayWin", "ProbAwayWin")
df <- rename_if_exists(df, "prob_home_win", "ProbHomeWin")
df <- rename_if_exists(df, "prob_draw", "ProbDraw")
df <- rename_if_exists(df, "prob_away_win", "ProbAwayWin")

required_cols <- c(
  "FTHG", "FTAG", "FTR",
  "LambdaHome", "LambdaAway",
  "ProbHomeWin", "ProbDraw", "ProbAwayWin",
  "PredictedResult"
)

missing_cols <- setdiff(required_cols, names(df))

if (length(missing_cols) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_cols, collapse = ", "),
      "\nCheck your test_predictions.csv column names."
    )
  )
}

# -----------------------------
# 4. Prepare data
# -----------------------------
df <- df %>%
  mutate(
    FTR = as.character(FTR),
    PredictedResult = as.character(PredictedResult),
    TotalGoalsActual = FTHG + FTAG,
    TotalGoalsPredicted = LambdaHome + LambdaAway,
    HomeResidual = FTHG - LambdaHome,
    AwayResidual = FTAG - LambdaAway,
    TotalResidual = TotalGoalsActual - TotalGoalsPredicted,
    MaxProb = pmax(ProbHomeWin, ProbDraw, ProbAwayWin)
  )

# -----------------------------
# 5. Helper metric functions
# -----------------------------
rmse_manual <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

accuracy_manual <- function(actual, predicted) {
  mean(actual == predicted, na.rm = TRUE)
}

multiclass_log_loss <- function(actual, p_home, p_draw, p_away) {
  eps <- 1e-15

  p_home <- pmin(pmax(p_home, eps), 1 - eps)
  p_draw <- pmin(pmax(p_draw, eps), 1 - eps)
  p_away <- pmin(pmax(p_away, eps), 1 - eps)

  actual_prob <- ifelse(
    actual == "H", p_home,
    ifelse(actual == "D", p_draw, p_away)
  )

  -mean(log(actual_prob), na.rm = TRUE)
}

brier_score_multiclass <- function(actual, p_home, p_draw, p_away) {
  y_home <- ifelse(actual == "H", 1, 0)
  y_draw <- ifelse(actual == "D", 1, 0)
  y_away <- ifelse(actual == "A", 1, 0)

  mean(
    (p_home - y_home)^2 +
      (p_draw - y_draw)^2 +
      (p_away - y_away)^2,
    na.rm = TRUE
  )
}

# -----------------------------
# 6. Basic evaluation
# -----------------------------
home_rmse <- rmse_manual(df$FTHG, df$LambdaHome)
away_rmse <- rmse_manual(df$FTAG, df$LambdaAway)
total_rmse <- rmse_manual(df$TotalGoalsActual, df$TotalGoalsPredicted)

model_accuracy <- accuracy_manual(df$FTR, df$PredictedResult)

model_log_loss <- multiclass_log_loss(
  df$FTR,
  df$ProbHomeWin,
  df$ProbDraw,
  df$ProbAwayWin
)

model_brier <- brier_score_multiclass(
  df$FTR,
  df$ProbHomeWin,
  df$ProbDraw,
  df$ProbAwayWin
)

cat("\n===== BASIC MODEL EVALUATION =====\n")
cat("Home goals RMSE:", round(home_rmse, 4), "\n")
cat("Away goals RMSE:", round(away_rmse, 4), "\n")
cat("Total goals RMSE:", round(total_rmse, 4), "\n")
cat("Match result accuracy:", round(model_accuracy, 4), "\n")
cat("Multiclass log loss:", round(model_log_loss, 4), "\n")
cat("Multiclass Brier score:", round(model_brier, 4), "\n")

# -----------------------------
# 7. Confusion matrix
# -----------------------------
cat("\n===== CONFUSION MATRIX =====\n")
conf_matrix <- table(
  Actual = df$FTR,
  Predicted = df$PredictedResult
)
print(conf_matrix)

# -----------------------------
# 8. Baseline comparison
# -----------------------------
baseline_home <- rep("H", nrow(df))
baseline_home_accuracy <- accuracy_manual(df$FTR, baseline_home)

most_common_result <- names(sort(table(df$FTR), decreasing = TRUE))[1]
baseline_common <- rep(most_common_result, nrow(df))
baseline_common_accuracy <- accuracy_manual(df$FTR, baseline_common)

# Simple baseline log loss using observed class frequencies
class_freq <- prop.table(table(df$FTR))
pH <- ifelse("H" %in% names(class_freq), as.numeric(class_freq["H"]), 1e-15)
pD <- ifelse("D" %in% names(class_freq), as.numeric(class_freq["D"]), 1e-15)
pA <- ifelse("A" %in% names(class_freq), as.numeric(class_freq["A"]), 1e-15)

baseline_frequency_log_loss <- multiclass_log_loss(
  df$FTR,
  rep(pH, nrow(df)),
  rep(pD, nrow(df)),
  rep(pA, nrow(df))
)

cat("\n===== BASELINE COMPARISON =====\n")
cat("Model accuracy:", round(model_accuracy, 4), "\n")
cat("Always home win accuracy:", round(baseline_home_accuracy, 4), "\n")
cat("Most common result baseline:", most_common_result, "\n")
cat("Most common result accuracy:", round(baseline_common_accuracy, 4), "\n")
cat("Model log loss:", round(model_log_loss, 4), "\n")
cat("Frequency baseline log loss:", round(baseline_frequency_log_loss, 4), "\n")

# -----------------------------
# 9. Subgroup robustness
# -----------------------------
evaluate_group <- function(data, group_name) {
  tibble(
    Group = group_name,
    N = nrow(data),
    Accuracy = accuracy_manual(data$FTR, data$PredictedResult),
    HomeRMSE = rmse_manual(data$FTHG, data$LambdaHome),
    AwayRMSE = rmse_manual(data$FTAG, data$LambdaAway),
    TotalRMSE = rmse_manual(data$TotalGoalsActual, data$TotalGoalsPredicted),
    LogLoss = multiclass_log_loss(
      data$FTR,
      data$ProbHomeWin,
      data$ProbDraw,
      data$ProbAwayWin
    ),
    BrierScore = brier_score_multiclass(
      data$FTR,
      data$ProbHomeWin,
      data$ProbDraw,
      data$ProbAwayWin
    )
  )
}

subgroup_results <- list()
subgroup_results[[1]] <- evaluate_group(df, "Overall")

if ("League" %in% names(df)) {
  leagues <- unique(df$League)
  for (lg in leagues) {
    temp <- df %>% filter(League == lg)
    subgroup_results[[length(subgroup_results) + 1]] <-
      evaluate_group(temp, paste("League:", lg))
  }
}

high_conf <- df %>% filter(MaxProb >= 0.50)
if (nrow(high_conf) > 0) {
  subgroup_results[[length(subgroup_results) + 1]] <-
    evaluate_group(high_conf, "High confidence: max prob >= 0.50")
}

close_games <- df %>% filter(MaxProb < 0.45)
if (nrow(close_games) > 0) {
  subgroup_results[[length(subgroup_results) + 1]] <-
    evaluate_group(close_games, "Close games: max prob < 0.45")
}

subgroup_metrics <- bind_rows(subgroup_results)

cat("\n===== SUBGROUP ROBUSTNESS =====\n")
print(subgroup_metrics)

# -----------------------------
# 10. Calibration checks
# -----------------------------
make_calibration_data <- function(data, prob_col, outcome_label) {
  data %>%
    mutate(
      PredictedProb = .data[[prob_col]],
      ActualBinary = ifelse(FTR == outcome_label, 1, 0),
      ProbBucket = cut(
        PredictedProb,
        breaks = seq(0, 1, by = 0.1),
        include.lowest = TRUE
      )
    ) %>%
    group_by(ProbBucket) %>%
    summarise(
      N = n(),
      AvgPredictedProb = mean(PredictedProb, na.rm = TRUE),
      ObservedFrequency = mean(ActualBinary, na.rm = TRUE),
      .groups = "drop"
    )
}

cal_home <- make_calibration_data(df, "ProbHomeWin", "H")
cal_draw <- make_calibration_data(df, "ProbDraw", "D")
cal_away <- make_calibration_data(df, "ProbAwayWin", "A")

plot_calibration <- function(cal_data, title, filename) {
  p <- ggplot(cal_data, aes(x = AvgPredictedProb, y = ObservedFrequency)) +
    geom_point(size = 3) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(
      title = title,
      x = "Average predicted probability",
      y = "Observed frequency"
    ) +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_minimal()

  ggsave(filename, p, width = 7, height = 5)
}

plot_calibration(cal_home, "Calibration: Home Win", "calibration_home_win.png")
plot_calibration(cal_draw, "Calibration: Draw", "calibration_draw.png")
plot_calibration(cal_away, "Calibration: Away Win", "calibration_away_win.png")

# -----------------------------
# 11. Residual diagnostics
# -----------------------------
p1 <- ggplot(df, aes(x = TotalResidual)) +
  geom_histogram(bins = 20) +
  labs(
    title = "Distribution of Total Goals Residuals",
    x = "Actual total goals - predicted total goals",
    y = "Count"
  ) +
  theme_minimal()

ggsave("residual_histogram.png", p1, width = 7, height = 5)

p2 <- ggplot(df, aes(x = TotalGoalsPredicted, y = TotalResidual)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Residuals vs Fitted Total Goals",
    x = "Predicted total goals",
    y = "Residual"
  ) +
  theme_minimal()

ggsave("residuals_vs_fitted.png", p2, width = 7, height = 5)

p3 <- ggplot(df, aes(x = LambdaHome, y = FTHG)) +
  geom_point(alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Home Goals",
    x = "Predicted home goals",
    y = "Actual home goals"
  ) +
  theme_minimal()

ggsave("actual_vs_predicted_home_goals.png", p3, width = 7, height = 5)

p4 <- ggplot(df, aes(x = LambdaAway, y = FTAG)) +
  geom_point(alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Away Goals",
    x = "Predicted away goals",
    y = "Actual away goals"
  ) +
  theme_minimal()

ggsave("actual_vs_predicted_away_goals.png", p4, width = 7, height = 5)

# -----------------------------
# 12. Bootstrap confidence intervals
# -----------------------------
set.seed(42)

B <- 1000

bootstrap_results <- tibble(
  Accuracy = numeric(B),
  HomeRMSE = numeric(B),
  AwayRMSE = numeric(B),
  TotalRMSE = numeric(B),
  LogLoss = numeric(B),
  BrierScore = numeric(B)
)

for (b in 1:B) {
  boot_df <- df %>% slice_sample(n = nrow(df), replace = TRUE)

  bootstrap_results$Accuracy[b] <- accuracy_manual(
    boot_df$FTR,
    boot_df$PredictedResult
  )

  bootstrap_results$HomeRMSE[b] <- rmse_manual(
    boot_df$FTHG,
    boot_df$LambdaHome
  )

  bootstrap_results$AwayRMSE[b] <- rmse_manual(
    boot_df$FTAG,
    boot_df$LambdaAway
  )

  bootstrap_results$TotalRMSE[b] <- rmse_manual(
    boot_df$TotalGoalsActual,
    boot_df$TotalGoalsPredicted
  )

  bootstrap_results$LogLoss[b] <- multiclass_log_loss(
    boot_df$FTR,
    boot_df$ProbHomeWin,
    boot_df$ProbDraw,
    boot_df$ProbAwayWin
  )

  bootstrap_results$BrierScore[b] <- brier_score_multiclass(
    boot_df$FTR,
    boot_df$ProbHomeWin,
    boot_df$ProbDraw,
    boot_df$ProbAwayWin
  )
}

bootstrap_ci <- bootstrap_results %>%
  pivot_longer(cols = everything(), names_to = "Metric", values_to = "Value") %>%
  group_by(Metric) %>%
  summarise(
    Mean = mean(Value, na.rm = TRUE),
    Lower95 = quantile(Value, 0.025, na.rm = TRUE),
    Upper95 = quantile(Value, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

cat("\n===== BOOTSTRAP CONFIDENCE INTERVALS =====\n")
print(bootstrap_ci)

# -----------------------------
# 13. Save outputs
# -----------------------------
basic_metrics <- tibble(
  Metric = c(
    "Home goals RMSE",
    "Away goals RMSE",
    "Total goals RMSE",
    "Match result accuracy",
    "Multiclass log loss",
    "Multiclass Brier score",
    "Always home win accuracy",
    "Most common result accuracy",
    "Frequency baseline log loss"
  ),
  Value = c(
    home_rmse,
    away_rmse,
    total_rmse,
    model_accuracy,
    model_log_loss,
    model_brier,
    baseline_home_accuracy,
    baseline_common_accuracy,
    baseline_frequency_log_loss
  )
)

write.csv(basic_metrics, "robustness_metrics.csv", row.names = FALSE)
write.csv(subgroup_metrics, "subgroup_metrics.csv", row.names = FALSE)
write.csv(bootstrap_ci, "bootstrap_confidence_intervals.csv", row.names = FALSE)
write.csv(cal_home, "calibration_home_win.csv", row.names = FALSE)
write.csv(cal_draw, "calibration_draw.csv", row.names = FALSE)
write.csv(cal_away, "calibration_away_win.csv", row.names = FALSE)

cat("\nSaved files:\n")
cat("robustness_metrics.csv\n")
cat("subgroup_metrics.csv\n")
cat("bootstrap_confidence_intervals.csv\n")
cat("calibration_home_win.csv\n")
cat("calibration_draw.csv\n")
cat("calibration_away_win.csv\n")
cat("calibration_home_win.png\n")
cat("calibration_draw.png\n")
cat("calibration_away_win.png\n")
cat("residual_histogram.png\n")
cat("residuals_vs_fitted.png\n")
cat("actual_vs_predicted_home_goals.png\n")
cat("actual_vs_predicted_away_goals.png\n")
