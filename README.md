# Sports Prediction

Sports prediction and market-implied probability analytics in Python.

This repository is an educational data analytics project. It does **not** provide betting advice. The code estimates fair probabilities, compares them with market-implied probabilities, and studies theoretical paper-trading edge under model uncertainty.

## Projects

### Football Prediction

`football_prediction/champions_league_prediction.py`

Supervised expected-goals model for a Champions League final case study: PSG vs Arsenal.

Features:

- Imports domestic league and Champions League match data.
- Builds no-leakage season-to-date team features.
- Trains Poisson expected-goals models.
- Evaluates historical predictions chronologically.
- Tunes a draw-threshold decision layer.
- Compares model probabilities with market no-vig probabilities.
- Exports model comparison and scoreline outputs.

`football_prediction/world_cup_2026_simulator.py`

FIFA World Cup 2026 tournament simulator and Monte Carlo projection.

Features:

- Models the 48-team World Cup format with 12 groups and a 32-team knockout stage.
- Applies top-two group qualification plus the 8 best third-place qualifiers.
- Adds World Cup-specific context: regional qualifiers, confederation competitions, travel, rest days, host advantage, squad quality, injuries, and outright winner odds.
- Prints a 10,000-simulation tournament breakdown in the terminal.
- Exports detailed probability tables and supports knockout-tree and high-speed simulation visualizations.
- Labels decimal-odds calculations as total return vs net profit to avoid payout confusion.

#### Football Technical Workflow

The football project builds a supervised expected-goals model for PSG vs Arsenal using 2025/26 season data:

- English Premier League 2025/26
- French Ligue 1 2025/26
- Champions League 2025/26

The model is designed to avoid data leakage. For every historical match, the feature row only uses matches that happened **before** that match. The current match is never included in its own team averages.

The workflow is:

1. Load domestic league and Champions League match data.
2. Sort matches chronologically by date.
3. Build season-to-date team features before each fixture.
4. Drop early-season rows where either team has too little history.
5. Train two Poisson regression models:
   - one model for home goals
   - one model for away goals
6. Evaluate on the final 25% of matches chronologically.
7. Use the trained model to predict neutral-site PSG vs Arsenal expected goals.
8. Simulate scorelines with Monte Carlo.
9. Compare model probabilities with market no-vig probabilities.

#### Football Season-to-Date Variables

For each team before each match, the model calculates current-season features such as:

Attacking:

- goals for so far
- average goals for so far
- shots for so far, when available
- average shots for so far, when available

Defensive:

- goals against so far
- average goals against so far
- clean sheets so far
- clean sheet rate so far

Form and table strength:

- matches played so far
- wins so far
- draws so far
- losses so far
- points so far
- points per game so far
- goal difference so far
- average goal difference so far

The model creates home-team and away-team versions of these variables, then adds difference features:

```text
avg_gf_diff = home_avg_goals_for - away_avg_goals_for
avg_ga_diff = home_avg_goals_against - away_avg_goals_against
ppg_diff = home_points_per_game - away_points_per_game
goal_difference_diff = home_avg_goal_difference - away_avg_goal_difference
clean_sheet_rate_diff = home_clean_sheet_rate - away_clean_sheet_rate
```

It also includes:

- home advantage flag
- neutral venue flag
- league one-hot encoding

#### Football Poisson Expected-Goals Model

Football goals are count data, so the project uses Poisson regression.

For home goals:

```text
FTHG_i ~ Poisson(lambda_home_i)

log(lambda_home_i) =
    beta_0
  + beta_1 * home_attack_features_i
  + beta_2 * away_defence_features_i
  + beta_3 * form_features_i
  + beta_4 * league_indicators_i
```

For away goals:

```text
FTAG_i ~ Poisson(lambda_away_i)

log(lambda_away_i) =
    alpha_0
  + alpha_1 * away_attack_features_i
  + alpha_2 * home_defence_features_i
  + alpha_3 * form_features_i
  + alpha_4 * league_indicators_i
```

The model predicts:

```text
lambda_home = expected home goals
lambda_away = expected away goals
```

For PSG vs Arsenal, the final is neutral, so:

```text
neutral_venue = 1
home_advantage = 0
```

To remove nominal home/away ordering bias, the model predicts the final twice:

1. PSG nominal home, Arsenal nominal away
2. Arsenal nominal home, PSG nominal away

Then it averages the two orientations to get:

```text
lambda_PSG
lambda_Arsenal
```

Monte Carlo scorelines are generated as:

```text
PSG_goals     ~ Poisson(lambda_PSG)
Arsenal_goals ~ Poisson(lambda_Arsenal)
```

#### Full Model Variables

The original full model used a wider feature set:

- home season average goals for
- away season average goals for
- home season average goals against
- away season average goals against
- home points per game
- away points per game
- home average goal difference
- away average goal difference
- home clean sheet rate
- away clean sheet rate
- average goals-for difference
- average goals-against difference
- points-per-game difference
- goal-difference difference
- clean-sheet-rate difference
- home matches played so far
- away matches played so far
- home advantage
- neutral venue
- league one-hot encoded

This was intentionally broad, but it created redundancy because many variables measured similar team-strength concepts.

#### Reduced Model Variables

After comparing full and reduced models, the project kept a smaller feature set that performed better on the historical test set.

Reduced variables:

- `home_avg_goal_difference`
- `away_avg_goal_difference`
- `away_season_avg_shots_for_so_far`
- `away_season_goals_for_so_far`
- `away_season_avg_gf`
- `away_season_clean_sheets_so_far`
- `away_points_per_game`
- `away_points_so_far`
- `home_draws_so_far`
- `away_matches_played_so_far`
- `away_losses_so_far`
- `League`

Some variables from the full model were dropped because they were redundant, noisy, or weaker in the reduced specification. Dropped examples include:

- duplicated attack/defence averages that overlapped with goal difference
- several difference features that repeated information already captured by team-level strength variables
- broad home/away strength variables that did not improve out-of-sample performance
- low-signal clean-sheet and points variables that added complexity without enough predictive gain

The reduced model is simpler and performed better in the historical validation table.

Full R robustness evidence, including the important code excerpts and exported results, is documented in `football_prediction/README.md`.

#### Draw Prediction Improvement

A pure argmax classifier often underpredicts draws because football draws are frequently close to the strongest win probability without being the single highest probability.

The project keeps the underlying Poisson probabilities unchanged, then adds a decision-layer correction:

```python
def predict_result_with_draw_threshold(prob_home_win, prob_draw, prob_away_win, draw_margin=0.04):
    max_win_prob = max(prob_home_win, prob_away_win)

    if prob_draw >= max_win_prob - draw_margin:
        return "D"
    elif prob_home_win >= prob_away_win:
        return "H"
    else:
        return "A"
```

The model tunes `draw_margin` from 0.00 to 0.10 and selects the value that balances:

- overall accuracy
- draw recall
- macro F1

This improves draw recognition without changing the expected-goals model itself.

#### Football Model Improvement

The reduced model outperformed the full model on the historical test set:

| Metric | Full Model | Reduced Model |
|---|---:|---:|
| Home goals RMSE | 1.3278 | 1.2994 |
| Away goals RMSE | 1.1183 | 1.1032 |
| Total goals RMSE | 1.6986 | 1.7035 |
| Argmax accuracy | 0.4118 | 0.4379 |
| Threshold accuracy | 0.4314 | 0.4575 |
| Argmax macro F1 | 0.3209 | 0.3372 |
| Threshold macro F1 | 0.4026 | 0.4221 |
| Draw recall before threshold | 0.0000 | 0.0000 |
| Draw recall after threshold | 0.1778 | 0.1778 |
| Log loss | 1.0945 | 1.0648 |
| Brier score | 0.6650 | 0.6445 |
| Best draw margin | 0.09 | 0.08 |

Main improvement:

```text
Threshold accuracy improved from 43.14% to 45.75%.
Log loss improved from 1.0945 to 1.0648.
Brier score improved from 0.6650 to 0.6445.
Macro F1 improved from 0.4026 to 0.4221.
```

The reduced model was selected because it had better probability calibration and better classification robustness, while using fewer variables.

### NBA Prediction

`nba_prediction/game_7_spurs_vs_thunder/`

NBA Game 7 simulation and market-edge analysis for Spurs vs Thunder.

Features:

- Fetches NBA team game logs with `nba_api` when available.
- Falls back to a clean manual CSV template.
- Builds full-season plus recent-form team strength.
- Simulates totals, margins, regulation ties, and winners.
- Runs 100 parameter scenarios by default.
- Performs chronological backtesting on available historical games.
- Compares model probabilities against market odds.
- Uses Gurobi, when available, for theoretical portfolio optimization.
- Falls back to a greedy optimizer when Gurobi is unavailable.

#### NBA Technical Workflow

The NBA project is structured as a small analytics pipeline:

1. `data_fetcher.py` obtains NBA team game logs through `nba_api` when possible. If online fetching fails, it creates clean CSV templates so the model remains reproducible.
2. `feature_engineering.py` converts historical team-game logs into team strength estimates.
3. `simulator.py` generates Monte Carlo score distributions across many parameter scenarios.
4. `market.py` converts simulated outcomes into market probabilities and expected values.
5. `backtest.py` performs chronological validation using only games that occurred before the game being predicted.
6. `gurobi_optimizer.py` selects a theoretical paper-trading portfolio under budget, risk, edge, Kelly, and market-conflict constraints.

#### NBA Mathematical Model

The model estimates team strength from a blend of full-season and recent-form performance:

```text
blended_points_for =
    season_weight * season_points_for
  + recent_weight * recent_points_for

blended_points_against =
    season_weight * season_points_against
  + recent_weight * recent_points_against
```

The default blend is:

```text
season_weight = 0.70
recent_weight = 0.30
```

Expected offensive output combines each team's scoring strength with the opponent's defensive allowance:

```text
Spurs expected offense =
    offense_weight * Spurs blended points_for
  + defense_weight * Thunder blended points_against

Thunder expected offense =
    offense_weight * Thunder blended points_for
  + defense_weight * Spurs blended points_against
```

The model then estimates the two basketball targets most useful for markets:

```text
expected_total = Spurs expected points + Thunder expected points
expected_margin = Spurs expected points - Thunder expected points
```

Home court, rest, injury, referee/whistle sensitivity, pace, and Game 7 variance are added as configurable adjustments. The referee variable is treated as a subjective sensitivity input, not a factual claim of referee bias.

Monte Carlo simulation samples:

```text
simulated_total  ~ Normal(expected_total, total_sd)
simulated_margin ~ Normal(expected_margin, margin_sd)
```

Scores are reconstructed as:

```text
Spurs score   = (simulated_total + simulated_margin) / 2
Thunder score = (simulated_total - simulated_margin) / 2
```

The model runs 100 parameter scenarios by default, with score simulations inside each scenario. This produces distributions for win probability, regulation draw probability, handicap covers, totals, expected margin, and expected total points.

#### Market Odds Used

The NBA project manually inputs these decimal odds.

Match Winner:

| Selection | Decimal Odds |
|---|---:|
| San Antonio Spurs | 2.25 |
| Oklahoma City Thunder | 1.63 |

Regular Time Result:

| Selection | Decimal Odds |
|---|---:|
| San Antonio Spurs | 2.42 |
| Draw | 13.00 |
| Oklahoma City Thunder | 1.70 |

Points Handicap:

| Selection | Decimal Odds |
|---|---:|
| Spurs +1 | 2.25 |
| Thunder -1 | 1.63 |
| Spurs +1.5 | 2.20 |
| Thunder -1.5 | 1.66 |
| Spurs +2 | 2.12 |
| Thunder -2 | 1.71 |
| Spurs +2.5 | 2.01 |
| Thunder -2.5 | 1.78 |
| Spurs +3 | 1.93 |
| Thunder -3 | 1.85 |
| Spurs +3.5 | 1.86 |
| Thunder -3.5 | 1.92 |
| Spurs +4 | 1.78 |
| Thunder -4 | 2.01 |
| Spurs +4.5 | 1.72 |
| Thunder -4.5 | 2.10 |
| Spurs +5 | 1.69 |
| Thunder -5 | 2.15 |
| Spurs +5.5 | 1.64 |
| Thunder -5.5 | 2.23 |
| Spurs +6 | 1.59 |
| Thunder -6 | 2.33 |

Total Points:

| Selection | Decimal Odds |
|---|---:|
| Over 209.5 | 1.71 |
| Under 209.5 | 2.12 |
| Over 210 | 1.74 |
| Under 210 | 2.07 |
| Over 210.5 | 1.77 |
| Under 210.5 | 2.02 |
| Over 211 | 1.82 |
| Under 211 | 1.96 |
| Over 211.5 | 1.86 |
| Under 211.5 | 1.92 |
| Over 212 | 1.91 |
| Under 212 | 1.87 |
| Over 212.5 | 1.94 |
| Under 212.5 | 1.84 |
| Over 213 | 1.97 |
| Under 213 | 1.81 |
| Over 213.5 | 2.04 |
| Under 213.5 | 1.76 |
| Over 214 | 2.10 |
| Under 214 | 1.72 |
| Over 214.5 | 2.13 |
| Under 214.5 | 1.70 |

#### Market-Implied Probability and Edge

For each market selection, raw implied probability is:

```text
raw_implied_probability = 1 / decimal_odds
```

Because bookmakers embed a margin, raw implied probabilities in the same market usually sum to more than 1. The model removes this margin using no-vig normalization:

```text
no_vig_probability_i =
    raw_implied_probability_i
  / sum(raw_implied_probabilities within same market)
```

The model edge is:

```text
edge_i = model_probability_i - no_vig_probability_i
```

Expected value per 1 paper unit is:

```text
EV_i =
    model_probability_i * (decimal_odds_i - 1)
  - probability_of_loss_i
```

For handicap and total markets, push probability is handled separately when the simulated result lands exactly on the line.

#### Gurobi Optimization Layer

The Gurobi layer turns the market-edge table into a constrained portfolio optimization problem.

Decision variables:

```text
x_i     in {0, 1}
stake_i >= 0
```

where `x_i = 1` means market selection `i` is included, and `stake_i` is the theoretical paper stake allocated to selection `i`.

Raw EV objective:

```text
maximize sum_i stake_i * EV_i
```

The optimizer can also use a conservative objective. First it subtracts an uncertainty buffer:

```text
conservative_probability_i =
    model_probability_i - probability_uncertainty_i
```

Then it maximizes conservative EV:

```text
maximize sum_i stake_i * conservative_EV_i
```

Risk-adjusted objective:

```text
maximize
    sum_i stake_i * EV_i
  - lambda * sum_i stake_i * probability_uncertainty_i
```

The constraints are:

```text
sum_i stake_i <= bankroll
stake_i <= max_stake_per_bet * x_i
sum_i x_i <= max_num_bets
```

Selections below the minimum edge threshold are blocked:

```text
if edge_i < minimum_edge_threshold:
    x_i = 0
    stake_i = 0
```

Negative-EV selections are also blocked:

```text
if EV_i <= 0:
    x_i = 0
    stake_i = 0
```

Conflict constraints prevent incompatible selections:

```text
sum_i x_i within each conflict group <= 1
```

Examples:

- Do not select both sides of the same handicap line.
- Do not select both Over and Under on the same total.
- With strict mode enabled, select at most one market from each broad market type.

The model also applies a fractional Kelly-inspired cap:

```text
kelly_i =
    ((decimal_odds_i - 1) * model_probability_i
    - (1 - model_probability_i))
  / (decimal_odds_i - 1)

stake_i <= bankroll * kelly_fraction * max(0, kelly_i)
```

Default optimizer settings:

```text
bankroll = 100 paper units
max_stake_per_bet = 5
minimum_edge_threshold = 0.03
max_num_bets = 5
kelly_fraction = 0.25
objective_mode = conservative_ev
```

The optimizer outputs selected markets, stake allocation, expected paper profit, expected ROI, total stake used, unused bankroll, average edge, average EV, and selection rationale. If Gurobi is unavailable, the project prints a warning and uses a deterministic greedy fallback optimizer.

## ROC/AUC Backtest

Run the AUC backtest script from the repository root:

```bash
python3 roc_auc_backtest.py
```

The script plots true positive rate against false positive rate and writes:

- `outputs/football_roc_auc.png`
- `outputs/nba_roc_auc.png`
- `outputs/roc_auc_summary.csv`

Latest AUC results:

| Sport | Target | Sample size | AUC | AUC < 0.5 |
|---|---:|---:|---:|---|
| Football | Home win | 172 | 0.6654 | No |
| Football | Draw | 172 | 0.5685 | No |
| Football | Away win | 172 | 0.6371 | No |
| Football | Micro average | 516 | 0.6449 | No |
| Football | Macro average | 172 | 0.6237 | No |
| NBA | Home win | 1,338 | 0.7252 | No |
| NBA | Away win | 1,338 | 0.7252 | No |

Football uses the model's outcome probability columns (`P_HomeWin`, `P_Draw`, `P_AwayWin`) for one-vs-rest ROC curves. NBA uses `predicted_home_margin` for home win and `-predicted_home_margin` for away win, because the NBA backtest stores margin scores rather than calibrated win probabilities.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Gurobi is optional:

```bash
python3 -m pip install gurobipy
```

You also need a valid Gurobi license for the exact optimizer. Without it, the NBA project still runs with a fallback heuristic.

## Run

Football:

```bash
python3 football_prediction/champions_league_prediction.py
```

NBA:

```bash
cd nba_prediction/game_7_spurs_vs_thunder
python3 main.py
```

Force a fresh NBA data fetch:

```bash
REFRESH_NBA_DATA=1 python3 main.py
```

## Methodology

The project focuses on:

- No-leakage chronological feature construction.
- Monte Carlo simulation.
- Market-implied probability and no-vig normalization.
- Backtesting and robustness checks.
- Conservative model uncertainty handling.
- Optimization under budget, edge, Kelly, and conflict constraints.

## Disclaimer

This repository is for educational sports analytics and portfolio optimization practice only. Model outputs are uncertain, historical performance may not generalize, and real-money betting is risky.
