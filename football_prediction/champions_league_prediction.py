"""
Supervised expected-goals model for PSG vs Arsenal.

This script builds season-to-date features without data leakage:
for every historical fixture, team features are calculated only from matches
played before that fixture date.

Data used:
- EPL 2025/26 from Football-Data.co.uk
- Ligue 1 2025/26 from Football-Data.co.uk
- Champions League 2025/26 from OpenFootball

The final PSG vs Arsenal prediction is treated as a neutral Champions League
match and averaged across both nominal home/away orientations.
"""

# =============================================================================
# 1. Imports and settings
# =============================================================================

import io
import itertools
import math
import os
import re
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import certifi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
    recall_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

EPL_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
LIGUE_1_URL = "https://www.football-data.co.uk/mmz4281/2526/F1.csv"
CHAMPIONS_LEAGUE_URL = (
    "https://raw.githubusercontent.com/openfootball/champions-league/master/2025-26/cl.txt"
)

MINIMUM_MATCHES_PLAYED_SO_FAR = 5
N_TEST_MATCH_SIMULATIONS = 10_000
# Defaults are set for a shorter everyday run. Increase these from the shell
# when you want a slower, smoother final Monte Carlo estimate:
# N_FINAL_MONTE_CARLO_RUNS=100 N_FINAL_SIMULATIONS=100000 python3 supervised_expected_goals_model.py
N_FINAL_SIMULATIONS = int(os.getenv("N_FINAL_SIMULATIONS", "100000"))
N_FINAL_MONTE_CARLO_RUNS = int(os.getenv("N_FINAL_MONTE_CARLO_RUNS", "10"))
N_FINAL_TOTAL_SIMULATIONS = N_FINAL_SIMULATIONS * N_FINAL_MONTE_CARLO_RUNS
RANDOM_SEED = 42
RUN_DEEP_WEIGHT_BACKTEST = os.getenv("RUN_DEEP_WEIGHT_BACKTEST", "0") == "1"


# =============================================================================
# 2. Load data
# =============================================================================


def read_url_bytes(url: str) -> bytes:
    """Read a URL using certifi certificates for macOS/Python compatibility."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_context) as response:
        return response.read()


def load_football_data_csv(url: str, league_name: str) -> pd.DataFrame:
    """Load one Football-Data CSV and normalize the columns used by this model."""
    raw = pd.read_csv(io.BytesIO(read_url_bytes(url)))

    df = raw[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["League"] = league_name
    df["neutral_venue"] = 0
    df["home_advantage"] = 1

    # Shots are available in many Football-Data league files as HS and AS.
    df["HomeShots"] = raw["HS"] if "HS" in raw.columns else np.nan
    df["AwayShots"] = raw["AS"] if "AS" in raw.columns else np.nan

    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    return df


def clean_openfootball_team_name(team_name: str) -> str:
    """Remove kickoff time and country code from OpenFootball team names."""
    cleaned = team_name.strip()

    time_matches = list(re.finditer(r"\d{1,2}[:.]\d{2}\s+", cleaned))
    if time_matches:
        cleaned = cleaned[time_matches[-1].end() :]

    cleaned = re.sub(r"\s*\([A-Z]{3}\)\s*$", "", cleaned)
    return cleaned.strip()


def parse_openfootball_date(line: str) -> pd.Timestamp | None:
    """Parse date headers such as 'Tue Sep 16 2025' from OpenFootball text."""
    match = re.match(
        r"\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
        r"([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?",
        line,
    )
    if match is None:
        return None

    month = match.group(2)
    day = int(match.group(3))
    year_text = match.group(4)
    year = int(year_text) if year_text else (2025 if month in {"Aug", "Sep", "Oct", "Nov", "Dec"} else 2026)

    return pd.to_datetime(f"{day} {month} {year}", dayfirst=True)


def load_champions_league_results(url: str) -> pd.DataFrame:
    """
    Load completed Champions League 2025/26 results from OpenFootball.

    Extra-time scorelines are skipped because this model predicts 90-minute
    goals and 90-minute results.
    """
    raw_text = read_url_bytes(url).decode("utf-8")
    match_pattern = re.compile(
        r"(?P<Home>[^()\r\n]+?\s+\([A-Z]{3}\))\s+v\s+"
        r"(?P<Away>[^()\r\n]+?\s+\([A-Z]{3}\))\s+"
        r"(?P<FTHG>\d+)-(?P<FTAG>\d+)"
    )

    current_date = None
    rows = []

    for line in raw_text.splitlines():
        parsed_date = parse_openfootball_date(line)
        if parsed_date is not None:
            current_date = parsed_date
            continue

        for match in match_pattern.finditer(line):
            if "a.e.t." in line[match.end() : match.end() + 20]:
                continue

            home_goals = int(match.group("FTHG"))
            away_goals = int(match.group("FTAG"))

            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"

            rows.append(
                {
                    "Date": current_date,
                    "HomeTeam": clean_openfootball_team_name(match.group("Home")),
                    "AwayTeam": clean_openfootball_team_name(match.group("Away")),
                    "FTHG": home_goals,
                    "FTAG": away_goals,
                    "FTR": result,
                    "League": "UEFA Champions League",
                    "neutral_venue": 0,
                    "home_advantage": 1,
                    "HomeShots": np.nan,
                    "AwayShots": np.nan,
                }
            )

    return pd.DataFrame(rows).dropna(subset=["Date"])


epl = load_football_data_csv(EPL_URL, "English Premier League")
ligue_1 = load_football_data_csv(LIGUE_1_URL, "French Ligue 1")
champions_league = load_champions_league_results(CHAMPIONS_LEAGUE_URL)

matches = pd.concat([epl, ligue_1, champions_league], ignore_index=True)
matches = matches.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

print("Loaded completed matches:")
print(f"EPL: {len(epl):,}")
print(f"Ligue 1: {len(ligue_1):,}")
print(f"Champions League 90-minute results: {len(champions_league):,}")
print(f"Combined: {len(matches):,}")
print()


# =============================================================================
# 3. Build season-to-date features without leakage
# =============================================================================


def empty_team_state() -> dict[str, float]:
    """Create an empty current-season state for one team."""
    return {
        "matches": 0,
        "goals_for": 0,
        "goals_against": 0,
        "shots_for": 0,
        "shots_against": 0,
        "shot_matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "clean_sheets": 0,
    }


def team_features(prefix: str, state: dict[str, float]) -> dict[str, float]:
    """Convert a team's current state into model features."""
    matches_played = state["matches"]
    shot_matches = state["shot_matches"]

    avg_goals_for = state["goals_for"] / matches_played if matches_played else 0
    avg_goals_against = state["goals_against"] / matches_played if matches_played else 0
    avg_shots_for = state["shots_for"] / shot_matches if shot_matches else np.nan
    clean_sheet_rate = state["clean_sheets"] / matches_played if matches_played else 0
    points_per_game = state["points"] / matches_played if matches_played else 0
    goal_difference = state["goals_for"] - state["goals_against"]
    avg_goal_difference = goal_difference / matches_played if matches_played else 0

    return {
        f"{prefix}_matches_played_so_far": matches_played,
        f"{prefix}_season_goals_for_so_far": state["goals_for"],
        f"{prefix}_season_avg_gf": avg_goals_for,
        f"{prefix}_season_shots_for_so_far": state["shots_for"] if shot_matches else np.nan,
        f"{prefix}_season_avg_shots_for_so_far": avg_shots_for,
        f"{prefix}_season_goals_against_so_far": state["goals_against"],
        f"{prefix}_season_avg_ga": avg_goals_against,
        f"{prefix}_season_clean_sheets_so_far": state["clean_sheets"],
        f"{prefix}_clean_sheet_rate": clean_sheet_rate,
        f"{prefix}_wins_so_far": state["wins"],
        f"{prefix}_draws_so_far": state["draws"],
        f"{prefix}_losses_so_far": state["losses"],
        f"{prefix}_points_so_far": state["points"],
        f"{prefix}_points_per_game": points_per_game,
        f"{prefix}_goal_difference": goal_difference,
        f"{prefix}_avg_goal_difference": avg_goal_difference,
    }


def update_team_state(
    state: dict[str, float],
    goals_for: int,
    goals_against: int,
    shots_for: float,
) -> None:
    """Update one team state after a completed match."""
    state["matches"] += 1
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against

    if not pd.isna(shots_for):
        state["shots_for"] += shots_for
        state["shot_matches"] += 1

    if goals_against == 0:
        state["clean_sheets"] += 1

    if goals_for > goals_against:
        state["wins"] += 1
        state["points"] += 3
    elif goals_for == goals_against:
        state["draws"] += 1
        state["points"] += 1
    else:
        state["losses"] += 1


def build_season_to_date_dataset(match_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Build features using only matches before each fixture date.

    Matches on the same date are all featurized before any same-date match is
    used to update team state. This avoids subtle same-day leakage.
    """
    states = defaultdict(empty_team_state)
    feature_rows = []

    for match_date, date_matches in match_df.groupby("Date", sort=True):
        rows_to_update = []

        for _, row in date_matches.iterrows():
            home_state = states[row["HomeTeam"]]
            away_state = states[row["AwayTeam"]]

            feature_row = row.to_dict()
            feature_row.update(team_features("home", home_state))
            feature_row.update(team_features("away", away_state))

            feature_row["avg_gf_diff"] = (
                feature_row["home_season_avg_gf"] - feature_row["away_season_avg_gf"]
            )
            feature_row["avg_ga_diff"] = (
                feature_row["home_season_avg_ga"] - feature_row["away_season_avg_ga"]
            )
            feature_row["ppg_diff"] = (
                feature_row["home_points_per_game"] - feature_row["away_points_per_game"]
            )
            feature_row["goal_difference_diff"] = (
                feature_row["home_avg_goal_difference"]
                - feature_row["away_avg_goal_difference"]
            )
            feature_row["clean_sheet_rate_diff"] = (
                feature_row["home_clean_sheet_rate"] - feature_row["away_clean_sheet_rate"]
            )

            feature_rows.append(feature_row)
            rows_to_update.append(row)

        for row in rows_to_update:
            update_team_state(
                states[row["HomeTeam"]],
                goals_for=int(row["FTHG"]),
                goals_against=int(row["FTAG"]),
                shots_for=row["HomeShots"],
            )
            update_team_state(
                states[row["AwayTeam"]],
                goals_for=int(row["FTAG"]),
                goals_against=int(row["FTHG"]),
                shots_for=row["AwayShots"],
            )

    return pd.DataFrame(feature_rows), dict(states)


feature_data, final_team_states = build_season_to_date_dataset(matches)

model_data = feature_data[
    (feature_data["home_matches_played_so_far"] >= MINIMUM_MATCHES_PLAYED_SO_FAR)
    & (feature_data["away_matches_played_so_far"] >= MINIMUM_MATCHES_PLAYED_SO_FAR)
].copy()

model_data = model_data.sort_values("Date").reset_index(drop=True)

print("Feature dataset:")
print(f"Rows before early-season filter: {len(feature_data):,}")
print(f"Rows after early-season filter: {len(model_data):,}")
print()


# =============================================================================
# 4. Train/test split
# =============================================================================

split_index = int(len(model_data) * 0.75)
train = model_data.iloc[:split_index].copy()
test = model_data.iloc[split_index:].copy()

# ============================================================
# Export model data for R significance testing
# ============================================================

train.to_csv("train_model_data.csv", index=False)
test.to_csv("test_model_data.csv", index=False)
model_data.to_csv("full_model_data.csv", index=False)

print("Saved model data for R:")
print("train_model_data.csv")
print("test_model_data.csv")
print("full_model_data.csv")
print()

print(f"Training size: {len(train):,}")
print(f"Test size: {len(test):,}")
print()


# =============================================================================
# 5. Train Poisson regression models
# =============================================================================

home_numeric_features = [
    "home_season_avg_gf",
    "away_season_avg_ga",
    "home_points_per_game",
    "away_points_per_game",
    "home_avg_goal_difference",
    "away_avg_goal_difference",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "avg_gf_diff",
    "avg_ga_diff",
    "ppg_diff",
    "goal_difference_diff",
    "clean_sheet_rate_diff",
    "home_matches_played_so_far",
    "away_matches_played_so_far",
    "home_advantage",
    "neutral_venue",
]

away_numeric_features = [
    "away_season_avg_gf",
    "home_season_avg_ga",
    "away_points_per_game",
    "home_points_per_game",
    "away_avg_goal_difference",
    "home_avg_goal_difference",
    "away_clean_sheet_rate",
    "home_clean_sheet_rate",
    "avg_gf_diff",
    "avg_ga_diff",
    "ppg_diff",
    "goal_difference_diff",
    "clean_sheet_rate_diff",
    "home_matches_played_so_far",
    "away_matches_played_so_far",
    "home_advantage",
    "neutral_venue",
]

categorical_features = ["League"]

reduced_candidate_features = [
    "home_avg_goal_difference",
    "away_avg_goal_difference",
    "away_season_avg_shots_for_so_far",
    "away_season_goals_for_so_far",
    "away_season_avg_gf",
    "away_season_clean_sheets_so_far",
    "away_points_per_game",
    "away_points_so_far",
    "home_draws_so_far",
    "away_matches_played_so_far",
    "away_losses_so_far",
]

reduced_numeric_features = [
    feature for feature in reduced_candidate_features if feature in model_data.columns
]
missing_reduced_features = sorted(set(reduced_candidate_features) - set(reduced_numeric_features))

if missing_reduced_features:
    print("Reduced model skipped missing features:")
    print(missing_reduced_features)
    print()

print("Reduced model numeric features:")
print(reduced_numeric_features)
print()


class NumericFeatureWeighter(BaseEstimator, TransformerMixin):
    """
    Multiply standardized numeric features by chosen weights.

    Weighting before StandardScaler would mostly get normalized away. Weighting
    after scaling changes the effective importance/regularization pressure of
    feature groups during model fitting, which makes it useful for backtesting.
    """

    def __init__(self, weights: list[float] | np.ndarray):
        self.weights = np.asarray(weights, dtype=float)

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return X * self.weights


def feature_group_for_weighting(feature_name: str) -> str:
    """Assign a feature to a tunable weighting group."""
    if "clean_sheet" in feature_name:
        return "clean_sheet"
    if "goal_difference" in feature_name:
        return "goal_difference"
    if "points" in feature_name or "ppg" in feature_name:
        return "points"
    if "matches_played" in feature_name:
        return "sample_size"
    if "home_advantage" in feature_name or "neutral_venue" in feature_name:
        return "venue"
    return "attack_defense"


def numeric_weights_for_features(
    numeric_features: list[str],
    group_weights: dict[str, float] | None,
) -> list[float]:
    """Create one numeric weight per feature based on feature group."""
    if group_weights is None:
        group_weights = {}

    return [
        group_weights.get(feature_group_for_weighting(feature_name), 1.0)
        for feature_name in numeric_features
    ]


def make_poisson_pipeline(
    numeric_features: list[str],
    group_weights: dict[str, float] | None = None,
) -> Pipeline:
    """Create the preprocessing + Poisson regression pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "weighter",
                NumericFeatureWeighter(
                    numeric_weights_for_features(numeric_features, group_weights)
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            (
                "league",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", PoissonRegressor(alpha=0.01, max_iter=1000)),
        ]
    )


home_goal_model = make_poisson_pipeline(home_numeric_features)
away_goal_model = make_poisson_pipeline(away_numeric_features)
reduced_home_goal_model = make_poisson_pipeline(reduced_numeric_features)
reduced_away_goal_model = make_poisson_pipeline(reduced_numeric_features)

home_goal_model.fit(train[home_numeric_features + categorical_features], train["FTHG"])
away_goal_model.fit(train[away_numeric_features + categorical_features], train["FTAG"])
reduced_home_goal_model.fit(
    train[reduced_numeric_features + categorical_features],
    train["FTHG"],
)
reduced_away_goal_model.fit(
    train[reduced_numeric_features + categorical_features],
    train["FTAG"],
)


# =============================================================================
# 6. Evaluate on chronological test set
# =============================================================================

lambda_home = np.clip(
    home_goal_model.predict(test[home_numeric_features + categorical_features]),
    0.05,
    None,
)
lambda_away = np.clip(
    away_goal_model.predict(test[away_numeric_features + categorical_features]),
    0.05,
    None,
)
lambda_home_full = lambda_home
lambda_away_full = lambda_away

lambda_home_reduced = np.clip(
    reduced_home_goal_model.predict(test[reduced_numeric_features + categorical_features]),
    0.05,
    None,
)
lambda_away_reduced = np.clip(
    reduced_away_goal_model.predict(test[reduced_numeric_features + categorical_features]),
    0.05,
    None,
)

test_predictions = test[
    ["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
].copy()
test_predictions["lambda_home"] = lambda_home_full
test_predictions["lambda_away"] = lambda_away_full
test_predictions["lambda_home_full"] = lambda_home_full
test_predictions["lambda_away_full"] = lambda_away_full
test_predictions["lambda_home_reduced"] = lambda_home_reduced
test_predictions["lambda_away_reduced"] = lambda_away_reduced

home_goals_rmse = float(np.sqrt(mean_squared_error(test["FTHG"], lambda_home_full)))
away_goals_rmse = float(np.sqrt(mean_squared_error(test["FTAG"], lambda_away_full)))
total_goals_rmse = float(
    np.sqrt(mean_squared_error(test["FTHG"] + test["FTAG"], lambda_home_full + lambda_away_full))
)


def simulate_match_result_probabilities(
    home_lambdas: np.ndarray,
    away_lambdas: np.ndarray,
    n_simulations: int,
    random_seed: int,
) -> pd.DataFrame:
    """Monte Carlo result probabilities for many matches."""
    rng = np.random.default_rng(random_seed)
    probability_rows = []

    for home_lambda, away_lambda in zip(home_lambdas, away_lambdas):
        home_goals = rng.poisson(home_lambda, n_simulations)
        away_goals = rng.poisson(away_lambda, n_simulations)

        probability_rows.append(
            {
                "P_HomeWin": float((home_goals > away_goals).mean()),
                "P_Draw": float((home_goals == away_goals).mean()),
                "P_AwayWin": float((home_goals < away_goals).mean()),
            }
        )

    return pd.DataFrame(probability_rows)


def poisson_result_probabilities_from_lambdas(
    home_lambdas: np.ndarray,
    away_lambdas: np.ndarray,
    max_goals: int = 12,
) -> pd.DataFrame:
    """
    Fast deterministic H/D/A probabilities from independent Poisson lambdas.

    This is used for the weight-sweep backtest because running Monte Carlo for
    every candidate would add noise and waste time. Probabilities are truncated
    at max_goals and renormalized; at football lambdas, the omitted tail is tiny.
    """
    probability_rows = []
    goal_range = np.arange(max_goals + 1)

    for home_lambda, away_lambda in zip(home_lambdas, away_lambdas):
        home_pmf = np.exp(-home_lambda) * np.power(home_lambda, goal_range)
        away_pmf = np.exp(-away_lambda) * np.power(away_lambda, goal_range)

        factorials = np.array([math.factorial(goal) for goal in goal_range])
        home_pmf = home_pmf / factorials
        away_pmf = away_pmf / factorials

        score_matrix = np.outer(home_pmf, away_pmf)
        probability_mass = score_matrix.sum()

        probability_rows.append(
            {
                "P_HomeWin": float(np.tril(score_matrix, k=-1).sum() / probability_mass),
                "P_Draw": float(np.trace(score_matrix) / probability_mass),
                "P_AwayWin": float(np.triu(score_matrix, k=1).sum() / probability_mass),
            }
        )

    return pd.DataFrame(probability_rows)


def evaluate_probability_predictions(
    y_true: pd.Series,
    probability_df: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate H/D/A probabilities using argmax classification and log loss."""
    predicted_result = (
        probability_df[["P_HomeWin", "P_Draw", "P_AwayWin"]]
        .idxmax(axis=1)
        .map(label_by_probability_column)
    )

    metrics = prediction_metrics(y_true, predicted_result)
    metrics["log_loss"] = float(
        log_loss(
            y_true,
            probability_df[["P_AwayWin", "P_Draw", "P_HomeWin"]],
            labels=["A", "D", "H"],
        )
    )
    metrics["num_draw_predictions"] = int((predicted_result == "D").sum())
    return metrics


test_result_probabilities = simulate_match_result_probabilities(
    lambda_home_full,
    lambda_away_full,
    N_TEST_MATCH_SIMULATIONS,
    RANDOM_SEED,
)
reduced_test_result_probabilities = simulate_match_result_probabilities(
    lambda_home_reduced,
    lambda_away_reduced,
    N_TEST_MATCH_SIMULATIONS,
    RANDOM_SEED + 1,
)

test_predictions = pd.concat(
    [test_predictions.reset_index(drop=True), test_result_probabilities],
    axis=1,
)

test_predictions["P_HomeWin_Full"] = test_predictions["P_HomeWin"]
test_predictions["P_Draw_Full"] = test_predictions["P_Draw"]
test_predictions["P_AwayWin_Full"] = test_predictions["P_AwayWin"]

test_predictions["P_HomeWin_Reduced"] = reduced_test_result_probabilities["P_HomeWin"]
test_predictions["P_Draw_Reduced"] = reduced_test_result_probabilities["P_Draw"]
test_predictions["P_AwayWin_Reduced"] = reduced_test_result_probabilities["P_AwayWin"]

probability_columns = ["P_HomeWin", "P_Draw", "P_AwayWin"]
label_by_probability_column = {
    "P_HomeWin": "H",
    "P_Draw": "D",
    "P_AwayWin": "A",
}

test_predictions["PredictedResult_Original"] = (
    test_predictions[probability_columns]
    .idxmax(axis=1)
    .map(label_by_probability_column)
)

test_predictions["PredictedResult"] = test_predictions["PredictedResult_Original"]


def predict_result_with_draw_threshold(
    prob_home_win: float,
    prob_draw: float,
    prob_away_win: float,
    draw_margin: float = 0.04,
) -> str:
    """
    Predict H/D/A with a draw threshold.

    Football draws are often not the highest-probability outcome, but they can
    be close to the strongest win probability. A pure argmax rule can therefore
    underpredict draws. This threshold rule is a decision-layer correction, not
    a change to the underlying Poisson probability model.
    """
    max_win_prob = max(prob_home_win, prob_away_win)

    if prob_draw >= max_win_prob - draw_margin:
        return "D"
    elif prob_home_win >= prob_away_win:
        return "H"
    else:
        return "A"


def prediction_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate classification metrics for H/D/A result predictions."""
    recalls = recall_score(
        y_true,
        y_pred,
        labels=["H", "D", "A"],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=["H", "D", "A"], average="macro")),
        "home_recall": float(recalls[0]),
        "draw_recall": float(recalls[1]),
        "away_recall": float(recalls[2]),
    }


def multiclass_brier_score(y_true: pd.Series, probability_df: pd.DataFrame) -> float:
    """Calculate multiclass Brier score for H/D/A probabilities."""
    actual = pd.get_dummies(y_true).reindex(columns=["H", "D", "A"], fill_value=0).to_numpy()
    predicted = probability_df[["P_HomeWin", "P_Draw", "P_AwayWin"]].to_numpy()
    return float(np.mean(np.sum((predicted - actual) ** 2, axis=1)))


def tune_draw_margin_for_probabilities(
    y_true: pd.Series,
    probability_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, pd.Series, dict[str, float], np.ndarray]:
    """
    Tune draw margin from 0.00 to 0.10.

    Best margin is selected by macro F1 first, then accuracy second, because
    the goal is to improve class balance without ignoring overall correctness.
    """
    rows = []
    margins = np.round(np.arange(0.00, 0.101, 0.01), 2)

    for margin in margins:
        predictions = probability_df.apply(
            lambda row: predict_result_with_draw_threshold(
                row["P_HomeWin"],
                row["P_Draw"],
                row["P_AwayWin"],
                draw_margin=margin,
            ),
            axis=1,
        )
        metrics = prediction_metrics(y_true, predictions)
        matrix = confusion_matrix(y_true, predictions, labels=["H", "D", "A"])
        rows.append(
            {
                "draw_margin": margin,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "draw_recall": metrics["draw_recall"],
                "home_recall": metrics["home_recall"],
                "away_recall": metrics["away_recall"],
                "num_draw_predictions": int((predictions == "D").sum()),
                "confusion_matrix": matrix.tolist(),
            }
        )

    tuning = pd.DataFrame(rows)
    best_row = tuning.sort_values(
        ["macro_f1", "accuracy", "draw_recall"],
        ascending=[False, False, False],
    ).iloc[0]
    best_margin = float(best_row["draw_margin"])

    best_predictions = probability_df.apply(
        lambda row: predict_result_with_draw_threshold(
            row["P_HomeWin"],
            row["P_Draw"],
            row["P_AwayWin"],
            draw_margin=best_margin,
        ),
        axis=1,
    )
    best_metrics = prediction_metrics(y_true, best_predictions)
    best_matrix = confusion_matrix(y_true, best_predictions, labels=["H", "D", "A"])

    return tuning, best_margin, best_predictions, best_metrics, best_matrix


def evaluate_model_variant(
    model_type: str,
    lambda_home_values: np.ndarray,
    lambda_away_values: np.ndarray,
    probability_df: pd.DataFrame,
    num_features_home: int,
    num_features_away: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Evaluate one model variant on goals, probabilities, and H/D/A classes."""
    argmax_predictions = (
        probability_df[["P_HomeWin", "P_Draw", "P_AwayWin"]]
        .idxmax(axis=1)
        .map(label_by_probability_column)
    )
    argmax_metrics = prediction_metrics(test_predictions["FTR"], argmax_predictions)
    argmax_matrix = confusion_matrix(
        test_predictions["FTR"],
        argmax_predictions,
        labels=["H", "D", "A"],
    )

    tuning, best_margin, threshold_predictions, threshold_metrics, threshold_matrix = (
        tune_draw_margin_for_probabilities(test_predictions["FTR"], probability_df)
    )

    result = {
        "ModelType": model_type,
        "NumFeaturesHome": num_features_home,
        "NumFeaturesAway": num_features_away,
        "HomeRMSE": float(np.sqrt(mean_squared_error(test["FTHG"], lambda_home_values))),
        "AwayRMSE": float(np.sqrt(mean_squared_error(test["FTAG"], lambda_away_values))),
        "TotalRMSE": float(
            np.sqrt(
                mean_squared_error(
                    test["FTHG"] + test["FTAG"],
                    lambda_home_values + lambda_away_values,
                )
            )
        ),
        "Accuracy_Argmax": argmax_metrics["accuracy"],
        "Accuracy_Threshold": threshold_metrics["accuracy"],
        "MacroF1_Argmax": argmax_metrics["macro_f1"],
        "MacroF1_Threshold": threshold_metrics["macro_f1"],
        "DrawRecall_Argmax": argmax_metrics["draw_recall"],
        "DrawRecall_Threshold": threshold_metrics["draw_recall"],
        "LogLoss": float(
            log_loss(
                test_predictions["FTR"],
                probability_df[["P_AwayWin", "P_Draw", "P_HomeWin"]],
                labels=["A", "D", "H"],
            )
        ),
        "BrierScore": multiclass_brier_score(test_predictions["FTR"], probability_df),
        "BestDrawMargin": best_margin,
        "ConfusionMatrix_Argmax": argmax_matrix.tolist(),
        "ConfusionMatrix_Threshold": threshold_matrix.tolist(),
    }

    return result, tuning, threshold_predictions, argmax_matrix, threshold_matrix


original_metrics = prediction_metrics(
    test_predictions["FTR"],
    test_predictions["PredictedResult_Original"],
)

result_accuracy = original_metrics["accuracy"]
result_confusion_matrix = confusion_matrix(
    test_predictions["FTR"],
    test_predictions["PredictedResult_Original"],
    labels=["H", "D", "A"],
)
result_log_loss = float(
    log_loss(
        test_predictions["FTR"],
        test_predictions[["P_AwayWin", "P_Draw", "P_HomeWin"]],
        labels=["A", "D", "H"],
    )
)

always_home_accuracy = float((test_predictions["FTR"] == "H").mean())
most_common_training_result = train["FTR"].value_counts().idxmax()
most_common_training_accuracy = float((test_predictions["FTR"] == most_common_training_result).mean())


# Tune the decision-layer draw threshold.
# The probability model stays exactly the same; only the final class decision
# changes when draw probability is close to the strongest win probability.
draw_margin_rows = []
draw_margins = np.round(np.arange(0.00, 0.101, 0.01), 2)

for draw_margin in draw_margins:
    threshold_predictions = test_predictions.apply(
        lambda row: predict_result_with_draw_threshold(
            row["P_HomeWin"],
            row["P_Draw"],
            row["P_AwayWin"],
            draw_margin=draw_margin,
        ),
        axis=1,
    )

    threshold_metrics_for_margin = prediction_metrics(
        test_predictions["FTR"],
        threshold_predictions,
    )
    threshold_confusion_matrix_for_margin = confusion_matrix(
        test_predictions["FTR"],
        threshold_predictions,
        labels=["H", "D", "A"],
    )

    draw_margin_rows.append(
        {
            "draw_margin": draw_margin,
            "accuracy": threshold_metrics_for_margin["accuracy"],
            "macro_f1": threshold_metrics_for_margin["macro_f1"],
            "draw_recall": threshold_metrics_for_margin["draw_recall"],
            "home_recall": threshold_metrics_for_margin["home_recall"],
            "away_recall": threshold_metrics_for_margin["away_recall"],
            "num_draw_predictions": int((threshold_predictions == "D").sum()),
            "confusion_matrix": threshold_confusion_matrix_for_margin.tolist(),
        }
    )

draw_margin_tuning_results = pd.DataFrame(draw_margin_rows)

# Balance rule:
# - Prefer margins that improve draw recall and macro F1.
# - Keep accuracy within 3 percentage points of the original argmax accuracy,
#   so the model does not predict too many draws and destroy overall quality.
maximum_allowed_accuracy_drop = 0.03
candidate_margins = draw_margin_tuning_results[
    draw_margin_tuning_results["accuracy"] >= result_accuracy - maximum_allowed_accuracy_drop
].copy()

if candidate_margins.empty:
    candidate_margins = draw_margin_tuning_results.copy()

candidate_margins["balance_score"] = (
    candidate_margins["accuracy"]
    + candidate_margins["macro_f1"]
    + candidate_margins["draw_recall"]
)

best_margin_row = candidate_margins.sort_values(
    ["balance_score", "accuracy", "macro_f1", "num_draw_predictions"],
    ascending=[False, False, False, True],
).iloc[0]
best_draw_margin = float(best_margin_row["draw_margin"])

test_predictions["PredictedResult_Threshold"] = test_predictions.apply(
    lambda row: predict_result_with_draw_threshold(
        row["P_HomeWin"],
        row["P_Draw"],
        row["P_AwayWin"],
        draw_margin=best_draw_margin,
    ),
    axis=1,
)
test_predictions["draw_margin_used"] = best_draw_margin

threshold_metrics = prediction_metrics(
    test_predictions["FTR"],
    test_predictions["PredictedResult_Threshold"],
)
threshold_confusion_matrix = confusion_matrix(
    test_predictions["FTR"],
    test_predictions["PredictedResult_Threshold"],
    labels=["H", "D", "A"],
)

full_probability_df = test_predictions[
    ["P_HomeWin_Full", "P_Draw_Full", "P_AwayWin_Full"]
].rename(
    columns={
        "P_HomeWin_Full": "P_HomeWin",
        "P_Draw_Full": "P_Draw",
        "P_AwayWin_Full": "P_AwayWin",
    }
)
reduced_probability_df = test_predictions[
    ["P_HomeWin_Reduced", "P_Draw_Reduced", "P_AwayWin_Reduced"]
].rename(
    columns={
        "P_HomeWin_Reduced": "P_HomeWin",
        "P_Draw_Reduced": "P_Draw",
        "P_AwayWin_Reduced": "P_AwayWin",
    }
)

full_comparison_row, draw_margin_tuning_full, full_threshold_predictions, full_argmax_matrix, full_threshold_matrix = (
    evaluate_model_variant(
        "Full",
        lambda_home_full,
        lambda_away_full,
        full_probability_df,
        num_features_home=len(home_numeric_features) + len(categorical_features),
        num_features_away=len(away_numeric_features) + len(categorical_features),
    )
)
reduced_comparison_row, draw_margin_tuning_reduced, reduced_threshold_predictions, reduced_argmax_matrix, reduced_threshold_matrix = (
    evaluate_model_variant(
        "Reduced",
        lambda_home_reduced,
        lambda_away_reduced,
        reduced_probability_df,
        num_features_home=len(reduced_numeric_features) + len(categorical_features),
        num_features_away=len(reduced_numeric_features) + len(categorical_features),
    )
)

model_comparison = pd.DataFrame([full_comparison_row, reduced_comparison_row])

test_predictions["PredictedResult_Full"] = (
    full_probability_df[["P_HomeWin", "P_Draw", "P_AwayWin"]]
    .idxmax(axis=1)
    .map(label_by_probability_column)
)
test_predictions["PredictedResult_Reduced"] = (
    reduced_probability_df[["P_HomeWin", "P_Draw", "P_AwayWin"]]
    .idxmax(axis=1)
    .map(label_by_probability_column)
)
test_predictions["PredictedResult_Full_Threshold"] = full_threshold_predictions
test_predictions["PredictedResult_Reduced_Threshold"] = reduced_threshold_predictions
test_predictions["draw_margin_used_full"] = full_comparison_row["BestDrawMargin"]
test_predictions["draw_margin_used_reduced"] = reduced_comparison_row["BestDrawMargin"]

# The reduced model is not automatically better just because variables were
# significant in R. Statistical significance can be affected by
# multicollinearity and sample size. Therefore, the reduced model is judged by
# out-of-sample performance against the full model.
reduced_wins = 0
selection_reasons = []

if reduced_comparison_row["LogLoss"] <= full_comparison_row["LogLoss"]:
    reduced_wins += 1
    selection_reasons.append("reduced has lower or equal log loss")
else:
    selection_reasons.append("full has lower log loss")

if reduced_comparison_row["MacroF1_Threshold"] >= full_comparison_row["MacroF1_Threshold"]:
    reduced_wins += 1
    selection_reasons.append("reduced has higher or equal threshold macro F1")
else:
    selection_reasons.append("full has higher threshold macro F1")

if reduced_comparison_row["DrawRecall_Threshold"] >= full_comparison_row["DrawRecall_Threshold"]:
    reduced_wins += 1
    selection_reasons.append("reduced has higher or equal threshold draw recall")
else:
    selection_reasons.append("full has higher threshold draw recall")

selected_model_type = "Reduced" if reduced_wins >= 2 else "Full"

if selected_model_type == "Reduced":
    selected_home_goal_model = reduced_home_goal_model
    selected_away_goal_model = reduced_away_goal_model
    selected_home_features = reduced_numeric_features
    selected_away_features = reduced_numeric_features
else:
    selected_home_goal_model = home_goal_model
    selected_away_goal_model = away_goal_model
    selected_home_features = home_numeric_features
    selected_away_features = away_numeric_features


# =============================================================================
# 6b. Feature-weight sweep backtest
# =============================================================================

# This deeper sweep does not replace the main model. It answers:
# "If I emphasized different feature groups more or less, which settings would
# have backtested best across several chronological folds?"
#
# The sweep uses deterministic Poisson scoreline probabilities rather than
# Monte Carlo so candidates are compared without simulation noise. It also
# includes the draw-margin decision layer because draw prediction quality is a
# decision problem as much as a lambda-estimation problem.
if RUN_DEEP_WEIGHT_BACKTEST:
    weight_values = [0.50, 0.75, 1.00, 1.25, 1.50]
    draw_margin_weight_sweep_values = np.round(np.arange(0.00, 0.101, 0.01), 2)
    time_series_backtest_folds = 4
    initial_train_fraction = 0.50
else:
    # Keep normal script runs fast. To run the full slower sweep:
    # RUN_DEEP_WEIGHT_BACKTEST=1 python3 supervised_expected_goals_model.py
    weight_values = [1.00]
    draw_margin_weight_sweep_values = np.array([best_draw_margin])
    time_series_backtest_folds = 1
    initial_train_fraction = 0.75

weight_group_names = [
    "attack_defense",
    "points",
    "goal_difference",
    "clean_sheet",
    "sample_size",
]


def chronological_backtest_splits(
    dataset: pd.DataFrame,
    n_folds: int,
    initial_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window chronological train/test splits."""
    n_rows = len(dataset)
    initial_train_size = int(n_rows * initial_fraction)
    test_size = (n_rows - initial_train_size) // n_folds

    splits = []
    for fold_number in range(n_folds):
        train_end = initial_train_size + fold_number * test_size
        test_end = (
            initial_train_size + (fold_number + 1) * test_size
            if fold_number < n_folds - 1
            else n_rows
        )
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))

    return splits


def evaluate_threshold_predictions(
    y_true: pd.Series,
    probability_df: pd.DataFrame,
    draw_margin: float,
) -> dict[str, float]:
    """Evaluate H/D/A predictions after applying a draw threshold."""
    threshold_predictions = probability_df.apply(
        lambda row: predict_result_with_draw_threshold(
            row["P_HomeWin"],
            row["P_Draw"],
            row["P_AwayWin"],
            draw_margin=draw_margin,
        ),
        axis=1,
    )

    metrics = prediction_metrics(y_true, threshold_predictions)
    metrics["num_draw_predictions"] = int((threshold_predictions == "D").sum())
    return metrics


backtest_splits = chronological_backtest_splits(
    model_data,
    n_folds=time_series_backtest_folds,
    initial_fraction=initial_train_fraction,
)

weight_sweep_rows = []
total_weight_candidates = len(weight_values) ** len(weight_group_names)

print(
    "Running deeper variable-weight backtest: "
    f"{total_weight_candidates:,} weight settings x "
    f"{time_series_backtest_folds} chronological folds x "
    f"{len(draw_margin_weight_sweep_values)} draw margins"
)
print()

for weight_tuple in itertools.product(weight_values, repeat=len(weight_group_names)):
    group_weights = dict(zip(weight_group_names, weight_tuple))

    fold_probability_frames = []
    fold_actual_results = []
    fold_home_goals_actual = []
    fold_away_goals_actual = []
    fold_lambda_home = []
    fold_lambda_away = []

    for fold_train_idx, fold_test_idx in backtest_splits:
        fold_train = model_data.iloc[fold_train_idx]
        fold_test = model_data.iloc[fold_test_idx]

        sweep_home_model = make_poisson_pipeline(home_numeric_features, group_weights)
        sweep_away_model = make_poisson_pipeline(away_numeric_features, group_weights)

        sweep_home_model.fit(
            fold_train[home_numeric_features + categorical_features],
            fold_train["FTHG"],
        )
        sweep_away_model.fit(
            fold_train[away_numeric_features + categorical_features],
            fold_train["FTAG"],
        )

        sweep_lambda_home = np.clip(
            sweep_home_model.predict(fold_test[home_numeric_features + categorical_features]),
            0.05,
            None,
        )
        sweep_lambda_away = np.clip(
            sweep_away_model.predict(fold_test[away_numeric_features + categorical_features]),
            0.05,
            None,
        )

        fold_probability_frames.append(
            poisson_result_probabilities_from_lambdas(
                sweep_lambda_home,
                sweep_lambda_away,
            )
        )
        fold_actual_results.append(fold_test["FTR"].reset_index(drop=True))
        fold_home_goals_actual.append(fold_test["FTHG"].to_numpy())
        fold_away_goals_actual.append(fold_test["FTAG"].to_numpy())
        fold_lambda_home.append(sweep_lambda_home)
        fold_lambda_away.append(sweep_lambda_away)

    all_probabilities = pd.concat(fold_probability_frames, ignore_index=True)
    all_actual_results = pd.concat(fold_actual_results, ignore_index=True)
    all_home_goals_actual = np.concatenate(fold_home_goals_actual)
    all_away_goals_actual = np.concatenate(fold_away_goals_actual)
    all_lambda_home = np.concatenate(fold_lambda_home)
    all_lambda_away = np.concatenate(fold_lambda_away)

    probability_metrics = evaluate_probability_predictions(
        all_actual_results,
        all_probabilities,
    )

    for draw_margin in draw_margin_weight_sweep_values:
        threshold_backtest_metrics = evaluate_threshold_predictions(
            all_actual_results,
            all_probabilities,
            draw_margin=draw_margin,
        )

        weight_sweep_rows.append(
            {
                **group_weights,
                "draw_margin": draw_margin,
                "home_goals_rmse": float(
                    np.sqrt(mean_squared_error(all_home_goals_actual, all_lambda_home))
                ),
                "away_goals_rmse": float(
                    np.sqrt(mean_squared_error(all_away_goals_actual, all_lambda_away))
                ),
                "total_goals_rmse": float(
                    np.sqrt(
                        mean_squared_error(
                            all_home_goals_actual + all_away_goals_actual,
                            all_lambda_home + all_lambda_away,
                        )
                    )
                ),
                "log_loss": probability_metrics["log_loss"],
                "argmax_accuracy": probability_metrics["accuracy"],
                "argmax_macro_f1": probability_metrics["macro_f1"],
                "argmax_draw_recall": probability_metrics["draw_recall"],
                "accuracy": threshold_backtest_metrics["accuracy"],
                "macro_f1": threshold_backtest_metrics["macro_f1"],
                "draw_recall": threshold_backtest_metrics["draw_recall"],
                "home_recall": threshold_backtest_metrics["home_recall"],
                "away_recall": threshold_backtest_metrics["away_recall"],
                "num_draw_predictions": threshold_backtest_metrics["num_draw_predictions"],
            }
        )

variable_weight_backtest_results = pd.DataFrame(weight_sweep_rows)

# Rank by a balanced objective:
# - log loss rewards calibrated probabilities,
# - macro F1 prevents one class from dominating,
# - draw recall rewards actually finding draws.
#
# The log-loss term is inverted so larger is better.
variable_weight_backtest_results["selection_score"] = (
    -variable_weight_backtest_results["log_loss"]
    + variable_weight_backtest_results["macro_f1"]
    + 0.50 * variable_weight_backtest_results["draw_recall"]
)
variable_weight_backtest_results = variable_weight_backtest_results.sort_values(
    ["selection_score", "log_loss", "accuracy"],
    ascending=[False, True, False],
).reset_index(drop=True)

best_weight_backtest_row = variable_weight_backtest_results.iloc[0]

evaluation_metrics = pd.DataFrame(
    [
        {"Metric": "training_size", "Value": len(train)},
        {"Metric": "test_size", "Value": len(test)},
        {"Metric": "home_goals_rmse", "Value": home_goals_rmse},
        {"Metric": "away_goals_rmse", "Value": away_goals_rmse},
        {"Metric": "total_goals_rmse", "Value": total_goals_rmse},
        {"Metric": "match_result_accuracy", "Value": result_accuracy},
        {"Metric": "original_macro_f1", "Value": original_metrics["macro_f1"]},
        {"Metric": "original_draw_recall", "Value": original_metrics["draw_recall"]},
        {"Metric": "best_draw_margin", "Value": best_draw_margin},
        {"Metric": "threshold_accuracy", "Value": threshold_metrics["accuracy"]},
        {"Metric": "threshold_macro_f1", "Value": threshold_metrics["macro_f1"]},
        {"Metric": "threshold_draw_recall", "Value": threshold_metrics["draw_recall"]},
        {"Metric": "best_weight_sweep_log_loss", "Value": best_weight_backtest_row["log_loss"]},
        {"Metric": "best_weight_sweep_accuracy", "Value": best_weight_backtest_row["accuracy"]},
        {"Metric": "best_weight_sweep_macro_f1", "Value": best_weight_backtest_row["macro_f1"]},
        {"Metric": "best_weight_attack_defense", "Value": best_weight_backtest_row["attack_defense"]},
        {"Metric": "best_weight_points", "Value": best_weight_backtest_row["points"]},
        {"Metric": "best_weight_goal_difference", "Value": best_weight_backtest_row["goal_difference"]},
        {"Metric": "best_weight_clean_sheet", "Value": best_weight_backtest_row["clean_sheet"]},
        {"Metric": "best_weight_sample_size", "Value": best_weight_backtest_row["sample_size"]},
        {"Metric": "multiclass_log_loss", "Value": result_log_loss},
        {"Metric": "always_home_win_baseline_accuracy", "Value": always_home_accuracy},
        {
            "Metric": "most_common_training_result_baseline_accuracy",
            "Value": most_common_training_accuracy,
        },
    ]
)

print("Evaluation:")
print(f"Training size: {len(train):,}")
print(f"Test size: {len(test):,}")
print(f"Home goals RMSE: {home_goals_rmse:.4f}")
print(f"Away goals RMSE: {away_goals_rmse:.4f}")
print(f"Total goals RMSE: {total_goals_rmse:.4f}")
print(f"Original match result accuracy: {result_accuracy:.4f}")
print("Original confusion matrix, rows=true H/D/A, columns=predicted H/D/A:")
print(result_confusion_matrix)
print(f"Multiclass log loss: {result_log_loss:.4f}")
print(f"Baseline accuracy if always predicting Home win: {always_home_accuracy:.4f}")
print(
    "Baseline accuracy if predicting most common training result "
    f"({most_common_training_result}): {most_common_training_accuracy:.4f}"
)
print()

print("Draw margin tuning results:")
print(
    draw_margin_tuning_results[
        [
            "draw_margin",
            "accuracy",
            "macro_f1",
            "draw_recall",
            "home_recall",
            "away_recall",
            "num_draw_predictions",
        ]
    ].round(4)
)
print()

print(f"Best draw margin selected: {best_draw_margin:.2f}")
print("Original vs threshold decision rule:")
print(f"Original accuracy: {original_metrics['accuracy']:.4f}")
print(f"Threshold accuracy: {threshold_metrics['accuracy']:.4f}")
print("Original confusion matrix, rows=true H/D/A, columns=predicted H/D/A:")
print(result_confusion_matrix)
print("Threshold confusion matrix, rows=true H/D/A, columns=predicted H/D/A:")
print(threshold_confusion_matrix)
print(f"Original macro F1: {original_metrics['macro_f1']:.4f}")
print(f"Threshold macro F1: {threshold_metrics['macro_f1']:.4f}")
print(f"Original draw recall: {original_metrics['draw_recall']:.4f}")
print(f"Threshold draw recall: {threshold_metrics['draw_recall']:.4f}")
print()

print("Variable weight backtest results, top 10 by balanced selection score:")
print(variable_weight_backtest_results.head(10).round(4))
print()

print("Best variable weights from backtest:")
print(
    best_weight_backtest_row[
        weight_group_names
        + [
            "draw_margin",
            "selection_score",
            "log_loss",
            "accuracy",
            "macro_f1",
            "draw_recall",
        ]
    ].round(4)
)
print()

print("Full vs reduced model comparison:")
comparison_display_columns = [
    "ModelType",
    "NumFeaturesHome",
    "NumFeaturesAway",
    "HomeRMSE",
    "AwayRMSE",
    "TotalRMSE",
    "Accuracy_Argmax",
    "Accuracy_Threshold",
    "MacroF1_Argmax",
    "MacroF1_Threshold",
    "DrawRecall_Argmax",
    "DrawRecall_Threshold",
    "LogLoss",
    "BrierScore",
    "BestDrawMargin",
]
print(model_comparison[comparison_display_columns].round(4))
print()

print(f"Selected final model: {selected_model_type}")
print("Selection rationale:")
for reason in selection_reasons:
    print(f"- {reason}")
print()


# =============================================================================
# 7. Apply trained model to PSG vs Arsenal neutral final
# =============================================================================


def find_team_state(states: dict[str, dict[str, float]], search_terms: list[str]) -> tuple[str, dict[str, float]]:
    """Find a team state by matching one or more search terms."""
    for team_name in sorted(states):
        if any(term.lower() in team_name.lower() for term in search_terms):
            return team_name, states[team_name]
    raise ValueError(f"Could not find team state for search terms: {search_terms}")


psg_state_name, psg_state = find_team_state(final_team_states, ["Paris Saint-Germain", "Paris SG", "PSG"])
arsenal_state_name, arsenal_state = find_team_state(final_team_states, ["Arsenal"])

print("Final model team states:")
print(f"PSG matched as: {psg_state_name}")
print(f"Arsenal matched as: {arsenal_state_name}")
print()


def make_fixture_feature_row(
    home_team: str,
    away_team: str,
    home_state: dict[str, float],
    away_state: dict[str, float],
    league: str,
    neutral_venue: int,
    home_advantage: int,
) -> pd.DataFrame:
    """Create one model feature row for a future fixture."""
    row = {
        "Date": pd.NaT,
        "League": league,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "neutral_venue": neutral_venue,
        "home_advantage": home_advantage,
    }
    row.update(team_features("home", home_state))
    row.update(team_features("away", away_state))

    row["avg_gf_diff"] = row["home_season_avg_gf"] - row["away_season_avg_gf"]
    row["avg_ga_diff"] = row["home_season_avg_ga"] - row["away_season_avg_ga"]
    row["ppg_diff"] = row["home_points_per_game"] - row["away_points_per_game"]
    row["goal_difference_diff"] = (
        row["home_avg_goal_difference"] - row["away_avg_goal_difference"]
    )
    row["clean_sheet_rate_diff"] = (
        row["home_clean_sheet_rate"] - row["away_clean_sheet_rate"]
    )

    return pd.DataFrame([row])


orientation_1 = make_fixture_feature_row(
    home_team="Paris Saint-Germain FC",
    away_team="Arsenal FC",
    home_state=psg_state,
    away_state=arsenal_state,
    league="UEFA Champions League",
    neutral_venue=1,
    home_advantage=0,
)

orientation_2 = make_fixture_feature_row(
    home_team="Arsenal FC",
    away_team="Paris Saint-Germain FC",
    home_state=arsenal_state,
    away_state=psg_state,
    league="UEFA Champions League",
    neutral_venue=1,
    home_advantage=0,
)

psg_home_lambda = float(
    np.clip(
        selected_home_goal_model.predict(orientation_1[selected_home_features + categorical_features])[0],
        0.05,
        None,
    )
)
arsenal_away_lambda = float(
    np.clip(
        selected_away_goal_model.predict(orientation_1[selected_away_features + categorical_features])[0],
        0.05,
        None,
    )
)

arsenal_home_lambda = float(
    np.clip(
        selected_home_goal_model.predict(orientation_2[selected_home_features + categorical_features])[0],
        0.05,
        None,
    )
)
psg_away_lambda = float(
    np.clip(
        selected_away_goal_model.predict(orientation_2[selected_away_features + categorical_features])[0],
        0.05,
        None,
    )
)

lambda_psg = (psg_home_lambda + psg_away_lambda) / 2
lambda_arsenal = (arsenal_away_lambda + arsenal_home_lambda) / 2

print("PSG vs Arsenal neutral final lambdas:")
print(f"Using selected model: {selected_model_type}")
print(f"PSG nominal-home lambda: {psg_home_lambda:.4f}")
print(f"PSG nominal-away lambda: {psg_away_lambda:.4f}")
print(f"Arsenal nominal-away lambda: {arsenal_away_lambda:.4f}")
print(f"Arsenal nominal-home lambda: {arsenal_home_lambda:.4f}")
print(f"lambda_PSG: {lambda_psg:.4f}")
print(f"lambda_Arsenal: {lambda_arsenal:.4f}")
print()


# =============================================================================
# 8. Final Monte Carlo simulation and market comparison
# =============================================================================

final_rng = np.random.default_rng()
final_outcome_counts = {
    "PSG win": 0,
    "Draw": 0,
    "Arsenal win": 0,
}
final_scoreline_counts: dict[int, int] = {}

for _ in range(N_FINAL_MONTE_CARLO_RUNS):
    psg_goals = final_rng.poisson(lambda_psg, N_FINAL_SIMULATIONS)
    arsenal_goals = final_rng.poisson(lambda_arsenal, N_FINAL_SIMULATIONS)

    final_outcome_counts["PSG win"] += int((psg_goals > arsenal_goals).sum())
    final_outcome_counts["Draw"] += int((psg_goals == arsenal_goals).sum())
    final_outcome_counts["Arsenal win"] += int((psg_goals < arsenal_goals).sum())

    encoded_scorelines = psg_goals * 100 + arsenal_goals
    unique_scorelines, counts = np.unique(encoded_scorelines, return_counts=True)
    for encoded_scoreline, count in zip(unique_scorelines, counts):
        final_scoreline_counts[int(encoded_scoreline)] = (
            final_scoreline_counts.get(int(encoded_scoreline), 0) + int(count)
        )

psg_win_probability = final_outcome_counts["PSG win"] / N_FINAL_TOTAL_SIMULATIONS
draw_probability = final_outcome_counts["Draw"] / N_FINAL_TOTAL_SIMULATIONS
arsenal_win_probability = final_outcome_counts["Arsenal win"] / N_FINAL_TOTAL_SIMULATIONS

print("PSG vs Arsenal 90-minute probabilities:")
print(
    f"Final simulation runs: {N_FINAL_MONTE_CARLO_RUNS} x "
    f"{N_FINAL_SIMULATIONS:,} = {N_FINAL_TOTAL_SIMULATIONS:,}"
)
print(f"PSG win: {psg_win_probability:.4f}")
print(f"Draw: {draw_probability:.4f}")
print(f"Arsenal win: {arsenal_win_probability:.4f}")
print()

scoreline_rows = []
for encoded_scoreline, count in final_scoreline_counts.items():
    psg_score = encoded_scoreline // 100
    arsenal_score = encoded_scoreline % 100
    scoreline_rows.append(
        {
            "PSGGoals": psg_score,
            "ArsenalGoals": arsenal_score,
            "Count": count,
            "Probability": count / N_FINAL_TOTAL_SIMULATIONS,
            "Scoreline": f"{psg_score}-{arsenal_score}",
        }
    )

scoreline_counts = pd.DataFrame(scoreline_rows)
psg_arsenal_scorelines = scoreline_counts.sort_values("Count", ascending=False)

print("Top 10 PSG-Arsenal scorelines:")
print(psg_arsenal_scorelines[["Scoreline", "Count", "Probability"]].head(10).round(4))
print()

market_comparison = pd.DataFrame(
    {
        "Outcome": ["PSG win", "Draw", "Arsenal win"],
        "DecimalOdds": [2.39, 3.28, 3.02],
        "ModelProbability": [
            psg_win_probability,
            draw_probability,
            arsenal_win_probability,
        ],
    }
)
market_comparison["RawImpliedProbability"] = 1 / market_comparison["DecimalOdds"]
market_comparison["NoVigMarketProbability"] = (
    market_comparison["RawImpliedProbability"]
    / market_comparison["RawImpliedProbability"].sum()
)
market_comparison["Edge"] = (
    market_comparison["ModelProbability"] - market_comparison["NoVigMarketProbability"]
)
market_comparison["ExpectedValuePerDollar"] = (
    market_comparison["ModelProbability"] * market_comparison["DecimalOdds"] - 1
)

market_comparison = market_comparison[
    [
        "Outcome",
        "DecimalOdds",
        "RawImpliedProbability",
        "NoVigMarketProbability",
        "ModelProbability",
        "Edge",
        "ExpectedValuePerDollar",
    ]
]

print("PSG-Arsenal market comparison:")
print(market_comparison.round(4))
print()


# =============================================================================
# 9. Save outputs
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    draw_margin_tuning_results["draw_margin"],
    draw_margin_tuning_results["accuracy"],
    marker="o",
    label="Accuracy",
)
ax.plot(
    draw_margin_tuning_results["draw_margin"],
    draw_margin_tuning_results["draw_recall"],
    marker="o",
    label="Draw recall",
)
ax.axvline(best_draw_margin, color="black", linestyle="--", linewidth=1, label="Selected margin")
ax.set_title("Draw Threshold Tuning")
ax.set_xlabel("Draw margin")
ax.set_ylabel("Metric value")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "draw_margin_tuning.png", dpi=150)
plt.close()

comparison_plot_data = model_comparison.set_index("ModelType")

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(comparison_plot_data.index, comparison_plot_data["Accuracy_Threshold"])
ax.set_title("Full vs Reduced: Threshold Accuracy")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "full_vs_reduced_accuracy.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(comparison_plot_data.index, comparison_plot_data["LogLoss"])
ax.set_title("Full vs Reduced: Log Loss")
ax.set_ylabel("Log loss")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "full_vs_reduced_logloss.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(comparison_plot_data.index, comparison_plot_data["DrawRecall_Threshold"])
ax.set_title("Full vs Reduced: Threshold Draw Recall")
ax.set_ylabel("Draw recall")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "full_vs_reduced_draw_recall.png", dpi=150)
plt.close()

market_plot_data = market_comparison.set_index("Outcome")[
    ["NoVigMarketProbability", "ModelProbability"]
]

fig, ax = plt.subplots(figsize=(9, 5))
market_plot_data.plot(kind="bar", ax=ax)
ax.set_title(f"PSG vs Arsenal Probabilities ({selected_model_type} Model)")
ax.set_ylabel("Probability")
ax.set_xlabel("Outcome")
ax.set_ylim(0, max(market_plot_data.max()) + 0.08)
ax.legend(["No-vig market", "Model"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "psg_arsenal_market_probabilities.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
edge_colors = ["#2ca02c" if edge > 0 else "#d62728" for edge in market_comparison["Edge"]]
ax.bar(market_comparison["Outcome"], market_comparison["Edge"], color=edge_colors)
ax.axhline(0, color="black", linewidth=1)
ax.set_title(f"PSG vs Arsenal Model Edge ({selected_model_type} Model)")
ax.set_ylabel("Model probability - no-vig market probability")
ax.set_xlabel("Outcome")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "psg_arsenal_market_edge.png", dpi=150)
plt.close()

top_10_final_scorelines = psg_arsenal_scorelines.head(10).copy()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(top_10_final_scorelines["Scoreline"], top_10_final_scorelines["Probability"])
ax.set_title(f"Top 10 PSG-Arsenal Scorelines ({selected_model_type} Model)")
ax.set_xlabel("Scoreline: PSG-Arsenal")
ax.set_ylabel("Probability")
ax.set_ylim(0, top_10_final_scorelines["Probability"].max() + 0.02)

for _, row in top_10_final_scorelines.iterrows():
    ax.text(
        row["Scoreline"],
        row["Probability"] + 0.002,
        f"{row['Probability']:.1%}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "psg_arsenal_top_10_scorelines.png", dpi=150)
plt.close()

for tuning_data, model_name, filename in [
    (draw_margin_tuning_full, "Full", "draw_margin_tuning_full.png"),
    (draw_margin_tuning_reduced, "Reduced", "draw_margin_tuning_reduced.png"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tuning_data["draw_margin"], tuning_data["accuracy"], marker="o", label="Accuracy")
    ax.plot(tuning_data["draw_margin"], tuning_data["macro_f1"], marker="o", label="Macro F1")
    ax.plot(tuning_data["draw_margin"], tuning_data["draw_recall"], marker="o", label="Draw recall")
    ax.set_title(f"{model_name} Model Draw-Margin Tuning")
    ax.set_xlabel("Draw margin")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
test_predictions.to_csv(OUTPUT_DIR / "test_predictions_full_vs_reduced.csv", index=False)
evaluation_metrics.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)
draw_margin_tuning_results.to_csv(OUTPUT_DIR / "draw_margin_tuning_results.csv", index=False)
model_comparison[comparison_display_columns].to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)
draw_margin_tuning_full.to_csv(OUTPUT_DIR / "draw_margin_tuning_full.csv", index=False)
draw_margin_tuning_reduced.to_csv(OUTPUT_DIR / "draw_margin_tuning_reduced.csv", index=False)
variable_weight_backtest_results.to_csv(
    OUTPUT_DIR / "variable_weight_backtest_results.csv",
    index=False,
)
market_comparison.to_csv(OUTPUT_DIR / "psg_arsenal_market_comparison.csv", index=False)
psg_arsenal_scorelines.to_csv(OUTPUT_DIR / "psg_arsenal_scorelines.csv", index=False)

print("Saved files:")
print(OUTPUT_DIR / "test_predictions.csv")
print(OUTPUT_DIR / "test_predictions_full_vs_reduced.csv")
print(OUTPUT_DIR / "evaluation_metrics.csv")
print(OUTPUT_DIR / "model_comparison.csv")
print(OUTPUT_DIR / "draw_margin_tuning_results.csv")
print(OUTPUT_DIR / "draw_margin_tuning_full.csv")
print(OUTPUT_DIR / "draw_margin_tuning_reduced.csv")
print(OUTPUT_DIR / "draw_margin_tuning.png")
print(OUTPUT_DIR / "full_vs_reduced_accuracy.png")
print(OUTPUT_DIR / "full_vs_reduced_logloss.png")
print(OUTPUT_DIR / "full_vs_reduced_draw_recall.png")
print(OUTPUT_DIR / "psg_arsenal_market_probabilities.png")
print(OUTPUT_DIR / "psg_arsenal_market_edge.png")
print(OUTPUT_DIR / "psg_arsenal_top_10_scorelines.png")
print(OUTPUT_DIR / "draw_margin_tuning_full.png")
print(OUTPUT_DIR / "draw_margin_tuning_reduced.png")
print(OUTPUT_DIR / "variable_weight_backtest_results.csv")
print(OUTPUT_DIR / "psg_arsenal_market_comparison.csv")
print(OUTPUT_DIR / "psg_arsenal_scorelines.csv")
