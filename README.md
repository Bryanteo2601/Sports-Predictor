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
