"""Configuration for the Spurs vs Thunder Game 7 simulation project.

This project is educational analytics / paper-trading only. It estimates
theoretical edges from model probabilities and market odds. It is not betting
advice, and real-money betting is risky.
"""

from pathlib import Path


DATA_DIR = Path(".")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SPURS = "San Antonio Spurs"
THUNDER = "Oklahoma City Thunder"

# Keep defaults short enough for quick iteration. Increase these for slower,
# smoother output:
# - SCORE_SIMULATIONS_PER_RUN = 10_000
# - MODEL_RUNS = 500
SCORE_SIMULATIONS_PER_RUN = 2_000
MODEL_RUNS = 100

RANDOM_SEED = None

# Game 7 is played at Oklahoma City's home court.
GAME7_HOME_TEAM = THUNDER

# Referee crew reported for this hypothetical/project setup.
CREW_CHIEF = "Marc Davis"
REFEREE = "John Goble"
UMPIRE = "Josh Tiven"
ALTERNATE = "Mitchell Ervin"

# This is not proof of referee bias. It is a sensitivity assumption for how a
# whistle/foul-drawing environment might affect OKC margin and total points.
OKC_FOUL_DRAWING_SENSITIVITY = 0.50
REF_MARGIN_POINTS_PER_SENSITIVITY = 1.50
REF_TOTAL_POINTS_PER_SENSITIVITY = 1.00

# Core model parameters. The optimizer/backtest can perturb these.
BASE_PARAMS = {
    "recent_form_window": 6,
    "minimum_prior_games_for_backtest": 2,
    "season_weight": 0.70,
    "recent_weight": 0.30,
    "home_court_advantage_points": 2.5,
    "injury_impact_multiplier": 1.0,
    "pace_weight": 0.20,
    "offense_weight": 0.45,
    "defense_weight": 0.35,
    "score_variance": 12.0,
    "game7_variance_multiplier": 1.10,
    "shooting_noise_sd": 2.5,
    "pace_noise_sd": 2.0,
    "turnover_noise_sd": 1.5,
}

# Gurobi / fallback portfolio optimization settings.
GUROBI_BANKROLL = 100.0
GUROBI_MAX_STAKE_PER_BET = 5.0
GUROBI_MINIMUM_EDGE_THRESHOLD = 0.03
GUROBI_MAX_NUM_BETS = 5
GUROBI_KELLY_FRACTION = 0.25
GUROBI_RISK_AVERSION = 0.50

# Choose one of: "raw_ev", "conservative_ev", "risk_adjusted_ev".
GUROBI_OBJECTIVE_MODE = "conservative_ev"

GUROBI_MARKET_TYPE_EXPOSURE = {
    "Match Winner": 40.0,
    "Regular Time Result": 30.0,
    "Handicap": 50.0,
    "Total Points": 50.0,
}

# When True, the optimizer may select at most one total, one handicap, one
# moneyline, and one regular-time result. This avoids stacking several highly
# correlated versions of the same model opinion.
STRICT_ONE_SELECTION_PER_MARKET_TYPE = True

# Previous six Spurs-Thunder games. These seed team_game_logs.csv if no richer
# data is available.
PREVIOUS_SIX_GAMES = [
    {"date": "2026-05-01", "home": THUNDER, "away": SPURS, "home_points": 115, "away_points": 122},
    {"date": "2026-05-03", "home": THUNDER, "away": SPURS, "home_points": 122, "away_points": 113},
    {"date": "2026-05-06", "home": SPURS, "away": THUNDER, "home_points": 108, "away_points": 123},
    {"date": "2026-05-08", "home": SPURS, "away": THUNDER, "home_points": 103, "away_points": 82},
    {"date": "2026-05-11", "home": THUNDER, "away": SPURS, "home_points": 127, "away_points": 114},
    {"date": "2026-05-14", "home": SPURS, "away": THUNDER, "home_points": 118, "away_points": 91},
]

ODDS_ROWS = [
    {"market_type": "Match Winner", "selection": SPURS, "line": None, "odds_decimal": 2.25},
    {"market_type": "Match Winner", "selection": THUNDER, "line": None, "odds_decimal": 1.63},
    {"market_type": "Regular Time Result", "selection": SPURS, "line": None, "odds_decimal": 2.42},
    {"market_type": "Regular Time Result", "selection": "Draw", "line": None, "odds_decimal": 13.00},
    {"market_type": "Regular Time Result", "selection": THUNDER, "line": None, "odds_decimal": 1.70},
]

for line, spurs_odds, thunder_odds in [
    (1, 2.25, 1.63),
    (1.5, 2.20, 1.66),
    (2, 2.12, 1.71),
    (2.5, 2.01, 1.78),
    (3, 1.93, 1.85),
    (3.5, 1.86, 1.92),
    (4, 1.78, 2.01),
    (4.5, 1.72, 2.10),
    (5, 1.69, 2.15),
    (5.5, 1.64, 2.23),
    (6, 1.59, 2.33),
]:
    ODDS_ROWS.append(
        {"market_type": "Handicap", "selection": f"{SPURS} +{line}", "line": line, "odds_decimal": spurs_odds}
    )
    ODDS_ROWS.append(
        {"market_type": "Handicap", "selection": f"{THUNDER} -{line}", "line": -line, "odds_decimal": thunder_odds}
    )

for line, over_odds, under_odds in [
    (209.5, 1.71, 2.12),
    (210, 1.74, 2.07),
    (210.5, 1.77, 2.02),
    (211, 1.82, 1.96),
    (211.5, 1.86, 1.92),
    (212, 1.91, 1.87),
    (212.5, 1.94, 1.84),
    (213, 1.97, 1.81),
    (213.5, 2.04, 1.76),
    (214, 2.10, 1.72),
    (214.5, 2.13, 1.70),
]:
    ODDS_ROWS.append(
        {"market_type": "Total Points", "selection": f"Over {line}", "line": line, "odds_decimal": over_odds}
    )
    ODDS_ROWS.append(
        {"market_type": "Total Points", "selection": f"Under {line}", "line": line, "odds_decimal": under_odds}
    )
