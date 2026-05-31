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
