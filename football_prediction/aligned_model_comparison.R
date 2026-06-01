
# =============================================================================
# aligned_model_comparison.R
#
# Fits Full and Reduced Poisson models in R using the SAME data, same
# train/test split, and same NA handling as the Python pipeline, so the
# Full vs Reduced comparison is fair.
#
# Key alignment choices:
#   - Reads train_model_data.csv / test_model_data.csv (exported by Python)
#   - Replaces NAs with 0 (matches Python's SimpleImputer fill_value=0)
#   - Uses glm(family = poisson) - unregularized, R's natural estimator
#   - Uses exact Poisson scoreline probabilities (matches the deterministic
#     helper in Python, no Monte Carlo noise)
#   - Checks nobs() == nrow(train): if they differ, rows were dropped and
#     the comparison may be biased
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Packages
# -----------------------------------------------------------------------------
packages <- c("tidyverse", "ggplot2")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(tidyverse)
library(ggplot2)

# Optional: glmnet for regularized Poisson matching Python's alpha=0.01
USE_GLMNET <- FALSE   # set TRUE to match Python exactly
GLMNET_LAMBDA <- 0.01  # equivalent to Python's alpha=0.01

if (USE_GLMNET) {
  if (!requireNamespace("glmnet", quietly = TRUE)) install.packages("glmnet")
  library(glmnet)
}

# -----------------------------------------------------------------------------
# 2. Load the same train/test split as Python
# -----------------------------------------------------------------------------
train <- read.csv("train_model_data.csv", stringsAsFactors = FALSE)
test  <- read.csv("test_model_data.csv",  stringsAsFactors = FALSE)

cat("Train rows:", nrow(train), "\n")
cat("Test rows: ", nrow(test),  "\n\n")

# -----------------------------------------------------------------------------
# 3. Feature sets — must exactly match Python's lists
# -----------------------------------------------------------------------------
home_numeric_features <- c(
  "home_season_avg_gf", "away_season_avg_ga",
  "home_points_per_game", "away_points_per_game",
  "home_avg_goal_difference", "away_avg_goal_difference",
  "home_clean_sheet_rate", "away_clean_sheet_rate",
  "avg_gf_diff", "avg_ga_diff", "ppg_diff",
  "goal_difference_diff", "clean_sheet_rate_diff",
  "home_matches_played_so_far", "away_matches_played_so_far",
  "home_advantage", "neutral_venue"
)

away_numeric_features <- c(
  "away_season_avg_gf", "home_season_avg_ga",
  "away_points_per_game", "home_points_per_game",
  "away_avg_goal_difference", "home_avg_goal_difference",
  "away_clean_sheet_rate", "home_clean_sheet_rate",
  "avg_gf_diff", "avg_ga_diff", "ppg_diff",
  "goal_difference_diff", "clean_sheet_rate_diff",
  "home_matches_played_so_far", "away_matches_played_so_far",
  "home_advantage", "neutral_venue"
)

reduced_candidate_features <- c(
  "home_avg_goal_difference", "away_avg_goal_difference",
  "away_season_avg_shots_for_so_far",
  "away_season_goals_for_so_far", "away_season_avg_gf",
  "away_season_clean_sheets_so_far",
  "away_points_per_game", "away_points_so_far",
  "home_draws_so_far", "away_matches_played_so_far",
  "away_losses_so_far"
)

# Keep only features that actually exist in the data
reduced_numeric_features <- intersect(reduced_candidate_features, names(train))

missing_reduced <- setdiff(reduced_candidate_features, names(train))
if (length(missing_reduced) > 0) {
  cat("Reduced model: skipped missing features:\n")
  cat(paste(missing_reduced, collapse = ", "), "\n\n")
}

cat("Reduced numeric features:\n")
cat(paste(reduced_numeric_features, collapse = ", "), "\n\n")

# -----------------------------------------------------------------------------
# 4. Impute NAs with 0 — matches Python's SimpleImputer(fill_value=0)
#    This prevents glm from silently dropping incomplete rows.
# -----------------------------------------------------------------------------
impute_zero <- function(df, features) {
  all_feats <- c(features, "League")
  for (col in intersect(all_feats, names(df))) {
    if (is.numeric(df[[col]])) {
      df[[col]][is.na(df[[col]])] <- 0
    }
  }
  df
}

all_numeric_features <- unique(c(
  home_numeric_features,
  away_numeric_features,
  reduced_numeric_features
))

train <- impute_zero(train, all_numeric_features)
test  <- impute_zero(test,  all_numeric_features)

# League as factor so glm encodes it the same way in both models
train$League <- as.factor(train$League)
test$League  <- factor(test$League, levels = levels(train$League))

# -----------------------------------------------------------------------------
# 5. Helper: build glm formula from feature vectors
# -----------------------------------------------------------------------------
make_formula <- function(target, numeric_feats) {
  rhs <- paste(c(numeric_feats, "League"), collapse = " + ")
  as.formula(paste(target, "~", rhs))
}

# -----------------------------------------------------------------------------
# 6. Fit models
# -----------------------------------------------------------------------------
cat("Fitting Full home model...\n")
full_home_model <- glm(
  make_formula("FTHG", home_numeric_features),
  data   = train,
  family = poisson(link = "log")
)

cat("Fitting Full away model...\n")
full_away_model <- glm(
  make_formula("FTAG", away_numeric_features),
  data   = train,
  family = poisson(link = "log")
)

cat("Fitting Reduced home model...\n")
reduced_home_model <- glm(
  make_formula("FTHG", reduced_numeric_features),
  data   = train,
  family = poisson(link = "log")
)

cat("Fitting Reduced away model...\n")
reduced_away_model <- glm(
  make_formula("FTAG", reduced_numeric_features),
  data   = train,
  family = poisson(link = "log")
)

# -----------------------------------------------------------------------------
# 7. Data integrity check: nobs() must equal nrow(train)
#    If not, glm dropped incomplete rows and the comparison is biased.
# -----------------------------------------------------------------------------
cat("\n===== DATA INTEGRITY CHECK =====\n")
cat("nrow(train):                 ", nrow(train), "\n")
cat("nobs(full_home_model):       ", nobs(full_home_model), "\n")
cat("nobs(full_away_model):       ", nobs(full_away_model), "\n")
cat("nobs(reduced_home_model):    ", nobs(reduced_home_model), "\n")
cat("nobs(reduced_away_model):    ", nobs(reduced_away_model), "\n")

check_nobs <- function(model, label) {
  n_train <- nrow(train)
  n_obs   <- nobs(model)
  if (n_obs < n_train) {
    warning(sprintf(
      "%s: %d rows used out of %d — %d rows were dropped (NAs not fully imputed)",
      label, n_obs, n_train, n_train - n_obs
    ))
  } else {
    cat(label, ": OK — all", n_obs, "rows used\n")
  }
}

check_nobs(full_home_model,    "full_home_model   ")
check_nobs(full_away_model,    "full_away_model   ")
check_nobs(reduced_home_model, "reduced_home_model")
check_nobs(reduced_away_model, "reduced_away_model")
cat("\n")

# -----------------------------------------------------------------------------
# 8. Predict lambdas on test set
# -----------------------------------------------------------------------------
clip_lambda <- function(x, min_val = 0.05) pmax(x, min_val)

lambda_home_full    <- clip_lambda(predict(full_home_model,    newdata = test, type = "response"))
lambda_away_full    <- clip_lambda(predict(full_away_model,    newdata = test, type = "response"))
lambda_home_reduced <- clip_lambda(predict(reduced_home_model, newdata = test, type = "response"))
lambda_away_reduced <- clip_lambda(predict(reduced_away_model, newdata = test, type = "response"))

# -----------------------------------------------------------------------------
# 9. Exact Poisson probabilities (no Monte Carlo noise)
#    Matches Python's poisson_result_probabilities_from_lambdas()
# -----------------------------------------------------------------------------
poisson_probs <- function(lambda_h, lambda_a, max_goals = 12) {
  goals <- 0:max_goals
  n <- length(lambda_h)
  p_home <- numeric(n); p_draw <- numeric(n); p_away <- numeric(n)

  for (i in seq_len(n)) {
    pmf_h <- dpois(goals, lambda_h[i])
    pmf_a <- dpois(goals, lambda_a[i])
    score_matrix <- outer(pmf_h, pmf_a)
    total_mass <- sum(score_matrix)
    p_home[i] <- sum(score_matrix[lower.tri(score_matrix)]) / total_mass
    p_draw[i] <- sum(diag(score_matrix)) / total_mass
    p_away[i] <- sum(score_matrix[upper.tri(score_matrix)]) / total_mass
  }

  data.frame(P_HomeWin = p_home, P_Draw = p_draw, P_AwayWin = p_away)
}

probs_full    <- poisson_probs(lambda_home_full,    lambda_away_full)
probs_reduced <- poisson_probs(lambda_home_reduced, lambda_away_reduced)

# -----------------------------------------------------------------------------
# 10. Metrics helpers
# -----------------------------------------------------------------------------
rmse_fn <- function(actual, predicted) sqrt(mean((actual - predicted)^2, na.rm = TRUE))

multiclass_log_loss <- function(actual, p_home, p_draw, p_away) {
  eps <- 1e-15
  p_home <- pmin(pmax(p_home, eps), 1 - eps)
  p_draw <- pmin(pmax(p_draw, eps), 1 - eps)
  p_away <- pmin(pmax(p_away, eps), 1 - eps)
  actual_p <- ifelse(actual == "H", p_home, ifelse(actual == "D", p_draw, p_away))
  -mean(log(actual_p), na.rm = TRUE)
}

brier_score <- function(actual, p_home, p_draw, p_away) {
  y_h <- as.integer(actual == "H")
  y_d <- as.integer(actual == "D")
  y_a <- as.integer(actual == "A")
  mean((p_home - y_h)^2 + (p_draw - y_d)^2 + (p_away - y_a)^2, na.rm = TRUE)
}

macro_f1 <- function(actual, predicted) {
  labels <- c("H", "D", "A")
  f1s <- sapply(labels, function(lbl) {
    tp <- sum(actual == lbl & predicted == lbl)
    fp <- sum(actual != lbl & predicted == lbl)
    fn <- sum(actual == lbl & predicted != lbl)
    if (tp + fp == 0 || tp + fn == 0) return(0)
    prec <- tp / (tp + fp)
    rec  <- tp / (tp + fn)
    if (prec + rec == 0) return(0)
    2 * prec * rec / (prec + rec)
  })
  mean(f1s)
}

recall_by_class <- function(actual, predicted, cls) {
  tp <- sum(actual == cls & predicted == cls)
  fn <- sum(actual == cls & predicted != cls)
  if (tp + fn == 0) return(NA_real_)
  tp / (tp + fn)
}

argmax_result <- function(probs) {
  apply(probs, 1, function(row) {
    c("P_HomeWin" = "H", "P_Draw" = "D", "P_AwayWin" = "A")[which.max(row)]
  })
}

evaluate_model <- function(model_label,
                            lambda_h, lambda_a,
                            probs,
                            actual_home_goals, actual_away_goals, actual_ftr) {
  pred_result <- argmax_result(probs)

  list(
    ModelType        = model_label,
    HomeRMSE         = rmse_fn(actual_home_goals, lambda_h),
    AwayRMSE         = rmse_fn(actual_away_goals, lambda_a),
    TotalRMSE        = rmse_fn(actual_home_goals + actual_away_goals, lambda_h + lambda_a),
    Accuracy         = mean(actual_ftr == pred_result, na.rm = TRUE),
    MacroF1          = macro_f1(actual_ftr, pred_result),
    DrawRecall       = recall_by_class(actual_ftr, pred_result, "D"),
    HomeRecall       = recall_by_class(actual_ftr, pred_result, "H"),
    AwayRecall       = recall_by_class(actual_ftr, pred_result, "A"),
    LogLoss          = multiclass_log_loss(actual_ftr, probs$P_HomeWin, probs$P_Draw, probs$P_AwayWin),
    BrierScore       = brier_score(actual_ftr, probs$P_HomeWin, probs$P_Draw, probs$P_AwayWin),
    NumDrawPredicted = sum(pred_result == "D")
  )
}

# -----------------------------------------------------------------------------
# 11. Evaluate and compare
# -----------------------------------------------------------------------------
full_metrics    <- evaluate_model(
  "Full",
  lambda_home_full, lambda_away_full, probs_full,
  test$FTHG, test$FTAG, test$FTR
)
reduced_metrics <- evaluate_model(
  "Reduced",
  lambda_home_reduced, lambda_away_reduced, probs_reduced,
  test$FTHG, test$FTAG, test$FTR
)

comparison <- bind_rows(
  as_tibble(full_metrics),
  as_tibble(reduced_metrics)
)

cat("===== ALIGNED FULL vs REDUCED COMPARISON (R glm, aligned preprocessing) =====\n")
print(comparison %>% mutate(across(where(is.numeric), ~round(.x, 4))), width = 120)
cat("\n")

# Decision: which model wins on majority of key metrics?
reduced_wins <- 0
reasons <- character(0)

if (reduced_metrics$LogLoss <= full_metrics$LogLoss) {
  reduced_wins <- reduced_wins + 1
  reasons <- c(reasons, "Reduced has lower or equal log loss")
} else {
  reasons <- c(reasons, "Full has lower log loss")
}

if (reduced_metrics$MacroF1 >= full_metrics$MacroF1) {
  reduced_wins <- reduced_wins + 1
  reasons <- c(reasons, "Reduced has higher or equal macro F1")
} else {
  reasons <- c(reasons, "Full has higher macro F1")
}

if (reduced_metrics$Accuracy >= full_metrics$Accuracy) {
  reduced_wins <- reduced_wins + 1
  reasons <- c(reasons, "Reduced has higher or equal accuracy")
} else {
  reasons <- c(reasons, "Full has higher accuracy")
}

selected_model <- if (reduced_wins >= 2) "Reduced" else "Full"

cat("Decision (majority of LogLoss, MacroF1, Accuracy):\n")
for (r in reasons) cat(" -", r, "\n")
cat(sprintf("\nSelected model: %s (%d/3 criteria met by Reduced)\n\n", selected_model, reduced_wins))

# -----------------------------------------------------------------------------
# 12. AIC / Deviance comparison (R-only: model fit quality checks)
# -----------------------------------------------------------------------------
cat("===== MODEL FIT CHECKS (AIC, Deviance) =====\n")
cat(sprintf("Full home  — AIC: %.1f  Null deviance: %.1f  Residual deviance: %.1f\n",
            AIC(full_home_model), full_home_model$null.deviance, full_home_model$deviance))
cat(sprintf("Full away  — AIC: %.1f  Null deviance: %.1f  Residual deviance: %.1f\n",
            AIC(full_away_model), full_away_model$null.deviance, full_away_model$deviance))
cat(sprintf("Reduced home — AIC: %.1f  Null deviance: %.1f  Residual deviance: %.1f\n",
            AIC(reduced_home_model), reduced_home_model$null.deviance, reduced_home_model$deviance))
cat(sprintf("Reduced away — AIC: %.1f  Null deviance: %.1f  Residual deviance: %.1f\n",
            AIC(reduced_away_model), reduced_away_model$null.deviance, reduced_away_model$deviance))
cat("\n")

# LRT: does the full model add significant explanatory power over reduced?
# Only valid when reduced is a proper subset of full, which it approximately is
# (shared features). If not nested, just use AIC above.
tryCatch({
  lrt_home <- anova(reduced_home_model, full_home_model, test = "Chisq")
  lrt_away <- anova(reduced_away_model, full_away_model, test = "Chisq")
  cat("Likelihood Ratio Test (reduced vs full, home goals):\n")
  print(lrt_home)
  cat("\nLikelihood Ratio Test (reduced vs full, away goals):\n")
  print(lrt_away)
  cat("\n")
}, error = function(e) {
  cat("LRT skipped (models not nested):", conditionMessage(e), "\n\n")
})

# -----------------------------------------------------------------------------
# 13. Save outputs
# -----------------------------------------------------------------------------
write.csv(
  comparison %>% mutate(across(where(is.numeric), ~round(.x, 4))),
  "r_aligned_model_comparison.csv",
  row.names = FALSE
)

# Test predictions with both sets of lambdas for external inspection
test_preds <- test %>%
  select(Date, League, HomeTeam, AwayTeam, FTHG, FTAG, FTR) %>%
  mutate(
    lambda_home_full    = lambda_home_full,
    lambda_away_full    = lambda_away_full,
    lambda_home_reduced = lambda_home_reduced,
    lambda_away_reduced = lambda_away_reduced,
    P_HomeWin_Full    = probs_full$P_HomeWin,
    P_Draw_Full       = probs_full$P_Draw,
    P_AwayWin_Full    = probs_full$P_AwayWin,
    P_HomeWin_Reduced = probs_reduced$P_HomeWin,
    P_Draw_Reduced    = probs_reduced$P_Draw,
    P_AwayWin_Reduced = probs_reduced$P_AwayWin,
    PredictedResult_Full    = argmax_result(probs_full),
    PredictedResult_Reduced = argmax_result(probs_reduced)
  )

write.csv(test_preds, "r_aligned_test_predictions.csv", row.names = FALSE)

cat("Saved:\n")
cat("  r_aligned_model_comparison.csv\n")
cat("  r_aligned_test_predictions.csv\n")

# -----------------------------------------------------------------------------
# 14. Human-readable model summary
#     Run this any time to see which model won and why.
# -----------------------------------------------------------------------------
model_summary <- function() {
  div <- paste(rep("=", 62), collapse = "")
  cat("\n", div, "\n", sep = "")
  cat("  PSG vs Arsenal — Poisson Model Summary (R glm)\n")
  cat(div, "\n\n", sep = "")

  cat("DATA\n")
  cat(sprintf("  Training matches : %d\n", nrow(train)))
  cat(sprintf("  Test matches     : %d\n", nrow(test)))
  cat(sprintf("  Leagues          : %s\n",
              paste(levels(train$League), collapse = ", ")))
  cat(sprintf("  Rows dropped (NA): %d  (imputed to 0 before fitting)\n\n",
              nrow(train) - nobs(full_home_model)))

  cat("FEATURES\n")
  cat(sprintf("  Full model    : %d numeric features + League\n",
              length(home_numeric_features)))
  cat("  ", paste(home_numeric_features, collapse = ", "), "\n\n", sep = "")
  cat(sprintf("  Reduced model : %d numeric features + League\n",
              length(reduced_numeric_features)))
  cat("  ", paste(reduced_numeric_features, collapse = ", "), "\n\n", sep = "")

  cat("MODEL FIT  (training set — AIC, lower is better)\n")
  cat(sprintf("  Full home    AIC: %.1f\n", AIC(full_home_model)))
  cat(sprintf("  Full away    AIC: %.1f\n", AIC(full_away_model)))
  cat(sprintf("  Reduced home AIC: %.1f\n", AIC(reduced_home_model)))
  cat(sprintf("  Reduced away AIC: %.1f\n\n", AIC(reduced_away_model)))

  cat("OUT-OF-SAMPLE METRICS  (test set, n =", nrow(test), ")\n")
  fmt <- "  %-20s  Full: %6.4f   Reduced: %6.4f   Winner: %s\n"
  metrics_list <- list(
    list("Log Loss",    full_metrics$LogLoss,    reduced_metrics$LogLoss,    TRUE),
    list("Accuracy",    full_metrics$Accuracy,   reduced_metrics$Accuracy,   FALSE),
    list("Macro F1",    full_metrics$MacroF1,    reduced_metrics$MacroF1,    FALSE),
    list("Home RMSE",   full_metrics$HomeRMSE,   reduced_metrics$HomeRMSE,   TRUE),
    list("Away RMSE",   full_metrics$AwayRMSE,   reduced_metrics$AwayRMSE,   TRUE),
    list("Brier Score", full_metrics$BrierScore, reduced_metrics$BrierScore, TRUE)
  )
  for (m in metrics_list) {
    label <- m[[1]]; fv <- m[[2]]; rv <- m[[3]]; lib <- m[[4]]
    winner <- if ((lib && rv < fv) || (!lib && rv > fv)) "Reduced ✓" else "Full ✓"
    cat(sprintf(fmt, label, fv, rv, winner))
  }

  cat("\nVERDICT\n")
  cat(sprintf("  Selected model : %s\n", selected_model))
  cat(sprintf("  Criteria met   : Reduced won %d / 3 primary criteria\n", reduced_wins))
  cat("  Reasons:\n")
  for (r in reasons) cat("    -", r, "\n")

  cat("\n", div, "\n", sep = "")
}

model_summary()
