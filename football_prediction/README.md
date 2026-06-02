# Football Prediction

`champions_league_prediction.py` contains the Champions League expected-goals workflow.

It builds season-to-date features without data leakage, trains Poisson goal models, evaluates historical prediction quality, compares model probabilities with no-vig market probabilities, and simulates PSG vs Arsenal scorelines.

Run:

```bash
python3 champions_league_prediction.py
```

## FIFA World Cup 2026 Simulator

`world_cup_2026_simulator.py` runs a 48-team FIFA World Cup 2026 model from group stage through the final.

It includes:

- 12 groups of 4 teams
- top 2 group qualification plus the 8 best third-place teams
- Round of 32 through final bracket logic
- regional qualifier and confederation competition context
- travel, rest days, host advantage, squad quality, injuries, and market odds
- 10,000-run terminal projection with group-stage, knockout-stage, and title probabilities
- contender-value comparison using decimal odds, with total return and net profit labeled separately

Run from the repository root:

```bash
python3 football_prediction/world_cup_2026_simulator.py
```

Create the latest simulated knockout tree:

```bash
python3 football_prediction/plot_world_cup_bracket.py
```

Create the high-speed 10,000-simulation animation:

```bash
python3 football_prediction/animate_world_cup_simulation.py
```

Train a national-team machine-learning model on historical international results, backtest Euro 2024, and run a trained-model World Cup forecast:

```bash
python3 football_prediction/trained_international_model.py
```

The trained model:

- downloads/caches historical men's international results from `martj42/international_results`
- uses World Cup, Euros, Copa America, AFCON, Asian Cup, Gold Cup, Nations League, qualifiers, and down-weighted friendlies
- builds no-leakage pre-match rolling features and Elo-style ratings
- trains Poisson goal models
- backtests on Euro 2024
- exports `outputs/trained_international_euro_2024_metrics.csv`
- exports `outputs/trained_world_cup_2026_summary.csv`

Decimal odds note:

```text
Odds 9.00 means a $1 winning stake returns $9 total, for $8 net profit.
```

## Robustness Checks

The football model was checked with three separate R validation scripts:

- `robustness_test_expected_goals.R` checks prediction accuracy, RMSE, log loss, Brier score, calibration, subgroup performance, residual plots, and bootstrap confidence intervals.
- `aligned_model_comparison.R` refits the full and reduced Poisson models in R on the same train/test split as Python to check that the reduced model improvement is not a Python-only artefact.
- `champions_league.R` runs a multinomial logistic regression significance test for match outcomes `H`, `D`, and `A`.

Run the R robustness scripts from the repository root:

```bash
Rscript football_prediction/robustness_test_expected_goals.R
Rscript football_prediction/aligned_model_comparison.R
Rscript -e 'file.choose <- function(...) "train_model_data.csv"; source("football_prediction/champions_league.R")'
```

### Important Robustness Code

The expected-goals robustness script evaluates both goal accuracy and match-result probability quality:

```r
rmse_manual <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
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

  mean((p_home - y_home)^2 + (p_draw - y_draw)^2 + (p_away - y_away)^2, na.rm = TRUE)
}
```

Bootstrap confidence intervals test whether the headline metrics are stable under resampling:

```r
set.seed(42)
B <- 1000

for (b in 1:B) {
  boot_df <- df %>% slice_sample(n = nrow(df), replace = TRUE)

  bootstrap_results$Accuracy[b] <- accuracy_manual(boot_df$FTR, boot_df$PredictedResult)
  bootstrap_results$HomeRMSE[b] <- rmse_manual(boot_df$FTHG, boot_df$LambdaHome)
  bootstrap_results$AwayRMSE[b] <- rmse_manual(boot_df$FTAG, boot_df$LambdaAway)
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
}
```

The aligned R comparison uses the same train/test split and zero-imputation rule as the Python pipeline:

```r
train <- read.csv("train_model_data.csv", stringsAsFactors = FALSE)
test  <- read.csv("test_model_data.csv", stringsAsFactors = FALSE)

impute_zero <- function(df, features) {
  all_feats <- c(features, "League")
  for (col in intersect(all_feats, names(df))) {
    if (is.numeric(df[[col]])) df[[col]][is.na(df[[col]])] <- 0
  }
  df
}

train <- impute_zero(train, all_numeric_features)
test  <- impute_zero(test, all_numeric_features)
```

The full-vs-reduced decision is based on out-of-sample log loss, macro F1, and accuracy:

```r
if (reduced_metrics$LogLoss <= full_metrics$LogLoss) reduced_wins <- reduced_wins + 1
if (reduced_metrics$MacroF1 >= full_metrics$MacroF1) reduced_wins <- reduced_wins + 1
if (reduced_metrics$Accuracy >= full_metrics$Accuracy) reduced_wins <- reduced_wins + 1

selected_model <- if (reduced_wins >= 2) "Reduced" else "Full"
```

The match-outcome significance script fits a multinomial logistic regression with Draw as the baseline:

```r
df$FTR <- as.factor(df$FTR)
df$FTR <- relevel(df$FTR, ref = "D")

numeric_features <- df %>%
  select(where(is.numeric)) %>%
  select(-any_of(exclude_cols)) %>%
  colnames()

features <- c(numeric_features, "League")

outcome_model <- multinom(
  FTR ~ .,
  data = df %>% select(FTR, all_of(features)) %>% drop_na(),
  trace = FALSE,
  maxit = 1000
)

z_values <- summary(outcome_model)$coefficients / summary(outcome_model)$standard.errors
p_values <- 2 * (1 - pnorm(abs(z_values)))
```

### Basic Robustness Results

From `robustness_metrics.csv`:

| Metric | Value |
|---|---:|
| Home goals RMSE | 1.3760 |
| Away goals RMSE | 1.1014 |
| Total goals RMSE | 1.8043 |
| Match result accuracy | 0.4767 |
| Multiclass log loss | 1.0402 |
| Multiclass Brier score | 0.6269 |
| Always home win accuracy | 0.4360 |
| Most common result accuracy | 0.4360 |
| Frequency baseline log loss | 1.0758 |

The model beats the simple baseline on both accuracy and log loss:

- Accuracy improves from `0.4360` to `0.4767`.
- Log loss improves from `1.0758` to `1.0402`, where lower is better.

### Bootstrap Confidence Intervals

From `bootstrap_confidence_intervals.csv`, using 1,000 bootstrap samples:

| Metric | Mean | Lower 95% | Upper 95% |
|---|---:|---:|---:|
| Accuracy | 0.4769 | 0.4070 | 0.5465 |
| HomeRMSE | 1.3716 | 1.2192 | 1.5424 |
| AwayRMSE | 1.0974 | 0.9759 | 1.2240 |
| TotalRMSE | 1.7932 | 1.5554 | 2.0455 |
| LogLoss | 1.0405 | 0.9984 | 1.0824 |
| BrierScore | 0.6271 | 0.5973 | 0.6579 |

This shows that the model's main evaluation metrics are not dependent on a single lucky test split result.

### Subgroup Robustness

From `subgroup_metrics.csv`:

| Group | N | Accuracy | Home RMSE | Away RMSE | Total RMSE | Log Loss | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 172 | 0.4767 | 1.3760 | 1.1014 | 1.8043 | 1.0402 | 0.6269 |
| French Ligue 1 | 73 | 0.4658 | 1.3654 | 1.1894 | 1.9151 | 1.0452 | 0.6298 |
| English Premier League | 80 | 0.4500 | 1.2141 | 1.0288 | 1.4691 | 1.0644 | 0.6436 |
| UEFA Champions League | 19 | 0.6316 | 1.9417 | 1.0434 | 2.5084 | 0.9185 | 0.5451 |
| High confidence: max prob >= 0.50 | 36 | 0.6389 | 1.2904 | 1.0156 | 1.6195 | 0.8885 | 0.5198 |
| Close games: max prob < 0.45 | 105 | 0.4381 | 1.4667 | 1.1549 | 1.8992 | 1.0810 | 0.6556 |

The high-confidence subset performs better than the overall set, while close games are harder. This is the expected pattern if the probability model is carrying useful signal.

### Aligned Full vs Reduced Model Check

From `r_aligned_model_comparison.csv`, with the same train/test split and same NA handling in R:

| Metric | Full | Reduced | Better |
|---|---:|---:|---|
| Home RMSE | 1.4160 | 1.3870 | Reduced |
| Away RMSE | 1.1019 | 1.1019 | Reduced/tie |
| Total RMSE | 1.8214 | 1.8143 | Reduced |
| Accuracy | 0.4419 | 0.4477 | Reduced |
| Macro F1 | 0.3414 | 0.3375 | Full |
| Log loss | 1.0609 | 1.0488 | Reduced |
| Brier score | 0.6404 | 0.6338 | Reduced |

Data integrity checks:

| Check | Rows |
|---|---:|
| Training rows | 514 |
| Test rows | 172 |
| Rows used by full home model | 514 |
| Rows used by full away model | 514 |
| Rows used by reduced home model | 514 |
| Rows used by reduced away model | 514 |

No rows were dropped after imputation, so the comparison is aligned. The reduced model was selected because it won 2 of the 3 primary criteria: lower log loss and higher accuracy, while the full model had slightly higher macro F1.

### Draw Threshold Sensitivity

The draw-threshold decision layer was tuned from `0.00` to `0.10`. It keeps the Poisson probabilities unchanged and only changes the final class decision when draw probability is close to the strongest win probability.

From `r_model_comparison.csv`:

| Model | Argmax Accuracy | Threshold Accuracy | Argmax Macro F1 | Threshold Macro F1 | Draw Recall Before | Draw Recall After | Best Draw Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.4118 | 0.4314 | 0.3209 | 0.4026 | 0.0000 | 0.1778 | 0.09 |
| Reduced | 0.4379 | 0.4575 | 0.3372 | 0.4221 | 0.0000 | 0.1778 | 0.08 |

The reduced model with draw-threshold tuning gives the best balance of accuracy, macro F1, and draw recall.

### Match Outcome Variable Significance

The significance check from `champions_league.R` tests whether model features are statistically associated with match outcomes. It fits a multinomial logistic model:

```text
FTR ~ numeric season-to-date features + League
```

with Draw as the baseline, so coefficients compare:

- `H vs D`
- `A vs D`

Top significant variables from `top_match_outcome_significant_variables.csv`:

| Outcome Comparison | Variable | Coefficient | P-value | Odds Ratio | Significant at 5% |
|---|---|---:|---:|---:|---|
| A vs D | home_advantage | 4.7610 | 0.0000 | 116.8615 | Yes |
| H vs D | home_advantage | 3.1834 | 0.0000 | 24.1278 | Yes |
| A vs D | away_avg_goal_difference | 1.5947 | 0.0014 | 4.9270 | Yes |
| A vs D | home_avg_goal_difference | 1.1450 | 0.0142 | 3.1426 | Yes |
| H vs D | away_season_avg_shots_for_so_far | -0.4785 | 0.0225 | 0.6197 | Yes |
| A vs D | away_season_goals_for_so_far | -0.1347 | 0.0299 | 0.8740 | Yes |
| A vs D | away_season_clean_sheets_so_far | -0.2868 | 0.0604 | 0.7507 | No |
| A vs D | away_points_per_game | -2.0738 | 0.0639 | 0.1257 | No |
| H vs D | away_points_so_far | -0.4462 | 0.0643 | 0.6401 | No |
| A vs D | away_season_avg_gf | 1.2761 | 0.0758 | 3.5828 | No |
| H vs D | home_draws_so_far | 0.6739 | 0.0812 | 1.9619 | No |
| H vs D | away_matches_played_so_far | -1.5694 | 0.0859 | 0.2082 | No |
| H vs D | away_losses_so_far | -1.0157 | 0.0863 | 0.3622 | No |

The significance test supports the use of team-strength and form variables such as average goal difference, shots, goals-for, clean sheets, and points-related measures. It also flags that some broad home/away structure can be highly influential, so the neutral-venue final prediction averages both team orientations to reduce ordering bias.

## Robustness Conclusion

The robustness checks support the reduced football model:

- It beats simple baselines on accuracy and log loss.
- Bootstrap intervals show the headline metrics are reasonably stable.
- Subgroup analysis behaves sensibly: high-confidence matches perform better and close matches perform worse.
- The aligned R refit confirms the reduced model is not only better in the Python pipeline.
- Draw-threshold tuning improves draw recall and macro F1 without changing the underlying probabilities.
- The multinomial significance test confirms that several season-to-date football features have measurable association with match outcomes.

These checks do not prove the model is perfect, but they show that the final football model is more robust than a single train/test score or one-off PSG vs Arsenal simulation.
