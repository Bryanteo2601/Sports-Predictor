"""
Trained international football model for World Cup 2026 forecasting.

This is the ML version of the World Cup simulator:
- downloads/caches historical men's international results
- builds no-leakage pre-match national-team features
- trains Poisson goal models chronologically
- backtests on Euro 2024
- plugs the trained model into the 2026 World Cup tournament engine

Data source:
https://github.com/martj42/international_results
"""

from __future__ import annotations

import io
import math
import os
import ssl
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from world_cup_2026_simulator import (
    GROUPS,
    NEXT_ROUNDS,
    ROUND_OF_32_TEMPLATE,
    STADIUMS,
    Team,
    assign_third_place_teams,
    create_teams,
    knockout_context,
)


os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RESULTS_PATH = DATA_DIR / "international_results.csv"

RANDOM_SEED = 42
MAX_GOALS_FOR_PROBABILITY_GRID = 10
EURO_2024_START = pd.Timestamp("2024-06-14")
WORLD_CUP_2026_START = pd.Timestamp("2026-06-11")

HOST_COUNTRIES = {"United States", "Mexico", "Canada"}
HOST_TEAM_COUNTRY = {
    "United States": "United States",
    "Mexico": "Mexico",
    "Canada": "Canada",
}

CURRENT_AVAILABILITY_ADJUSTMENTS = {
    # Transparent live-news overlay. These are deliberately small multipliers
    # applied after the trained model predicts xG, because the trained model
    # itself does not know current squad availability.
    "Spain": {
        "attack_multiplier": 0.965,
        "opponent_attack_multiplier": 1.005,
        "note": "Fermin Lopez ruled out with a right-foot fifth-metatarsal fracture; Lamine Yamal reported expected fit but returning from a hamstring issue.",
    },
}

COMPETITION_WEIGHTS = {
    "FIFA World Cup": 1.50,
    "UEFA Euro": 1.40,
    "Copa América": 1.35,
    "African Cup of Nations": 1.30,
    "AFC Asian Cup": 1.25,
    "Gold Cup": 1.20,
    "Oceania Nations Cup": 1.10,
    "UEFA Nations League": 1.05,
    "CONCACAF Nations League": 1.00,
    "FIFA World Cup qualification": 1.10,
    "UEFA Euro qualification": 1.05,
    "AFC Asian Cup qualification": 1.00,
    "African Cup of Nations qualification": 1.00,
    "Gold Cup qualification": 0.95,
    "Friendly": 0.45,
}

MODEL_TOURNAMENTS = set(COMPETITION_WEIGHTS)

COMPETITION_BACKTESTS = [
    {
        "name": "World Cup 2022",
        "region": "Global",
        "tournament": "FIFA World Cup",
        "start": "2022-11-20",
        "end": "2022-12-18",
    },
    {
        "name": "Euro 2024",
        "region": "UEFA",
        "tournament": "UEFA Euro",
        "start": "2024-06-14",
        "end": "2024-07-14",
    },
    {
        "name": "Copa America 2024",
        "region": "CONMEBOL",
        "tournament": "Copa América",
        "start": "2024-06-20",
        "end": "2024-07-14",
    },
    {
        "name": "AFCON 2025",
        "region": "CAF",
        "tournament": "African Cup of Nations",
        "start": "2025-12-21",
        "end": "2026-01-18",
    },
    {
        "name": "Asian Cup 2023",
        "region": "AFC",
        "tournament": "AFC Asian Cup",
        "start": "2024-01-12",
        "end": "2024-02-10",
    },
    {
        "name": "Gold Cup 2025",
        "region": "CONCACAF",
        "tournament": "Gold Cup",
        "start": "2025-06-14",
        "end": "2025-07-06",
    },
    {
        "name": "UEFA Nations League 2024/25",
        "region": "UEFA",
        "tournament": "UEFA Nations League",
        "start": "2024-09-05",
        "end": "2025-06-08",
    },
    {
        "name": "CONCACAF Nations League 2024/25",
        "region": "CONCACAF",
        "tournament": "CONCACAF Nations League",
        "start": "2024-09-04",
        "end": "2025-03-23",
    },
    {
        "name": "World Cup 2026 Qualifiers",
        "region": "Global qualifiers",
        "tournament": "FIFA World Cup qualification",
        "start": "2023-09-07",
        "end": "2026-03-31",
    },
]

TEAM_NAME_ALIASES = {
    "Turkey": "Turkey",
    "Türkiye": "Turkey",
    "United States": "United States",
    "USA": "United States",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
}


@dataclass
class TeamState:
    elo: float = 1500.0
    matches: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0
    last_match_date: pd.Timestamp | None = None
    recent_goals_for: deque[int] | None = None
    recent_goals_against: deque[int] | None = None
    recent_points: deque[int] | None = None

    def __post_init__(self) -> None:
        if self.recent_goals_for is None:
            self.recent_goals_for = deque(maxlen=10)
        if self.recent_goals_against is None:
            self.recent_goals_against = deque(maxlen=10)
        if self.recent_points is None:
            self.recent_points = deque(maxlen=10)


def normalize_team_name(name: str) -> str:
    return TEAM_NAME_ALIASES.get(name, name)


def read_url_bytes(url: str) -> bytes:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_context) as response:
        return response.read()


def load_international_results(refresh: bool = False) -> pd.DataFrame:
    if refresh or not RESULTS_PATH.exists():
        raw_bytes = read_url_bytes(RESULTS_URL)
        RESULTS_PATH.write_bytes(raw_bytes)

    df = pd.read_csv(RESULTS_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    df["neutral"] = df["neutral"].astype(bool)
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return df


def tournament_category(tournament: str) -> str:
    if tournament in COMPETITION_WEIGHTS:
        return tournament
    if "qualification" in tournament.lower():
        return "Other qualification"
    if tournament == "Friendly":
        return "Friendly"
    return "Other tournament"


def competition_weight(tournament: str) -> float:
    if tournament in COMPETITION_WEIGHTS:
        return COMPETITION_WEIGHTS[tournament]
    if "qualification" in tournament.lower():
        return 0.90
    return 0.80


def include_for_model(tournament: str) -> bool:
    if tournament in MODEL_TOURNAMENTS:
        return True
    if "qualification" in tournament.lower():
        return True
    return tournament == "Friendly"


def result_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def expected_elo_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def actual_elo_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def goal_margin_multiplier(goals_for: int, goals_against: int) -> float:
    margin = abs(goals_for - goals_against)
    if margin <= 1:
        return 1.0
    return math.log(margin + 1)


def state_features(prefix: str, state: TeamState, match_date: pd.Timestamp) -> dict[str, float]:
    matches = state.matches
    recent_matches = len(state.recent_points or [])
    rest_days = 21.0
    if state.last_match_date is not None:
        rest_days = min(90.0, max(0.0, float((match_date - state.last_match_date).days)))

    return {
        f"{prefix}_elo": state.elo,
        f"{prefix}_matches": matches,
        f"{prefix}_avg_goals_for": state.goals_for / matches if matches else 0.0,
        f"{prefix}_avg_goals_against": state.goals_against / matches if matches else 0.0,
        f"{prefix}_avg_goal_difference": (state.goals_for - state.goals_against) / matches if matches else 0.0,
        f"{prefix}_points_per_match": state.points / matches if matches else 0.0,
        f"{prefix}_win_rate": state.wins / matches if matches else 0.0,
        f"{prefix}_draw_rate": state.draws / matches if matches else 0.0,
        f"{prefix}_recent_avg_goals_for": np.mean(state.recent_goals_for) if recent_matches else 0.0,
        f"{prefix}_recent_avg_goals_against": np.mean(state.recent_goals_against) if recent_matches else 0.0,
        f"{prefix}_recent_points_per_match": np.mean(state.recent_points) if recent_matches else 0.0,
        f"{prefix}_recent_matches": recent_matches,
        f"{prefix}_rest_days": rest_days,
    }


def update_state(
    state: TeamState,
    match_date: pd.Timestamp,
    goals_for: int,
    goals_against: int,
    elo_delta: float,
) -> None:
    points = result_points(goals_for, goals_against)
    state.matches += 1
    state.goals_for += goals_for
    state.goals_against += goals_against
    state.points += points
    state.wins += int(points == 3)
    state.draws += int(points == 1)
    state.losses += int(points == 0)
    state.last_match_date = match_date
    state.elo += elo_delta
    state.recent_goals_for.append(goals_for)
    state.recent_goals_against.append(goals_against)
    state.recent_points.append(points)


def build_feature_dataset(results: pd.DataFrame, min_date: str = "2000-01-01") -> tuple[pd.DataFrame, dict[str, TeamState]]:
    states = defaultdict(TeamState)
    rows = []
    filtered = results[results["date"].ge(pd.Timestamp(min_date))].copy()
    filtered = filtered[filtered["tournament"].map(include_for_model)].copy()

    for match_date, date_matches in filtered.groupby("date", sort=True):
        updates = []

        for _, match in date_matches.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            home_state = states[home_team]
            away_state = states[away_team]

            row = {
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(match["home_score"]),
                "away_score": int(match["away_score"]),
                "result": "H" if match["home_score"] > match["away_score"] else ("A" if match["home_score"] < match["away_score"] else "D"),
                "tournament": match["tournament"],
                "tournament_category": tournament_category(match["tournament"]),
                "competition_weight": competition_weight(match["tournament"]),
                "country": match["country"],
                "neutral": int(bool(match["neutral"])),
                "home_host": int((not bool(match["neutral"])) and match["country"] == home_team),
                "away_host": int((not bool(match["neutral"])) and match["country"] == away_team),
            }
            row.update(state_features("home", home_state, match_date))
            row.update(state_features("away", away_state, match_date))
            row["elo_diff"] = row["home_elo"] - row["away_elo"]
            row["avg_gf_diff"] = row["home_avg_goals_for"] - row["away_avg_goals_for"]
            row["avg_ga_diff"] = row["home_avg_goals_against"] - row["away_avg_goals_against"]
            row["avg_gd_diff"] = row["home_avg_goal_difference"] - row["away_avg_goal_difference"]
            row["ppg_diff"] = row["home_points_per_match"] - row["away_points_per_match"]
            row["recent_gf_diff"] = row["home_recent_avg_goals_for"] - row["away_recent_avg_goals_for"]
            row["recent_ga_diff"] = row["home_recent_avg_goals_against"] - row["away_recent_avg_goals_against"]
            row["recent_ppg_diff"] = row["home_recent_points_per_match"] - row["away_recent_points_per_match"]
            row["rest_days_diff"] = row["home_rest_days"] - row["away_rest_days"]
            rows.append(row)

            home_expected = expected_elo_score(home_state.elo, away_state.elo)
            home_actual = actual_elo_score(int(match["home_score"]), int(match["away_score"]))
            multiplier = goal_margin_multiplier(int(match["home_score"]), int(match["away_score"]))
            k_factor = 22 * competition_weight(match["tournament"]) * multiplier
            home_delta = k_factor * (home_actual - home_expected)
            updates.append((home_state, match_date, int(match["home_score"]), int(match["away_score"]), home_delta))
            updates.append((away_state, match_date, int(match["away_score"]), int(match["home_score"]), -home_delta))

        for update in updates:
            update_state(*update)

    features = pd.DataFrame(rows)
    return features, dict(states)


NUMERIC_FEATURES = [
    "neutral",
    "home_host",
    "away_host",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_matches",
    "away_matches",
    "home_avg_goals_for",
    "away_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_against",
    "home_avg_goal_difference",
    "away_avg_goal_difference",
    "home_points_per_match",
    "away_points_per_match",
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "home_recent_avg_goals_for",
    "away_recent_avg_goals_for",
    "home_recent_avg_goals_against",
    "away_recent_avg_goals_against",
    "home_recent_points_per_match",
    "away_recent_points_per_match",
    "home_recent_matches",
    "away_recent_matches",
    "home_rest_days",
    "away_rest_days",
    "avg_gf_diff",
    "avg_ga_diff",
    "avg_gd_diff",
    "ppg_diff",
    "recent_gf_diff",
    "recent_ga_diff",
    "recent_ppg_diff",
    "rest_days_diff",
]

CATEGORICAL_FEATURES = ["tournament_category"]


def make_goal_model(alpha: float = 0.02) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", PoissonRegressor(alpha=alpha, max_iter=1000)),
        ]
    )


def poisson_result_probabilities(home_lambda: float, away_lambda: float, max_goals: int = MAX_GOALS_FOR_PROBABILITY_GRID) -> tuple[float, float, float]:
    goals = np.arange(max_goals + 1)
    home_probs = np.exp(-home_lambda) * np.power(home_lambda, goals) / np.array([math.factorial(g) for g in goals])
    away_probs = np.exp(-away_lambda) * np.power(away_lambda, goals) / np.array([math.factorial(g) for g in goals])
    matrix = np.outer(home_probs, away_probs)
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def train_models(feature_data: pd.DataFrame, train_mask: pd.Series) -> tuple[Pipeline, Pipeline]:
    train = feature_data[train_mask].copy()
    sample_weight = train["competition_weight"].to_numpy()
    home_model = make_goal_model()
    away_model = make_goal_model()
    home_model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["home_score"], model__sample_weight=sample_weight)
    away_model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["away_score"], model__sample_weight=sample_weight)
    return home_model, away_model


def evaluate_predictions(df: pd.DataFrame, home_lambda: np.ndarray, away_lambda: np.ndarray) -> dict[str, float]:
    result_probs = np.array(
        [
            poisson_result_probabilities(float(h), float(a))
            for h, a in zip(home_lambda, away_lambda)
        ]
    )
    labels = ["H", "D", "A"]
    predicted = np.array([labels[index] for index in result_probs.argmax(axis=1)])
    actual = df["result"].to_numpy()
    y_true = np.array([[1 if label == result else 0 for label in labels] for result in actual])

    return {
        "matches": float(len(df)),
        "home_goals_rmse": float(mean_squared_error(df["home_score"], home_lambda) ** 0.5),
        "away_goals_rmse": float(mean_squared_error(df["away_score"], away_lambda) ** 0.5),
        "result_accuracy": float(accuracy_score(actual, predicted)),
        # sklearn orders string labels lexicographically. Our probability
        # columns are H, D, A, so reorder to A, D, H for log_loss.
        "multiclass_log_loss": float(log_loss(actual, result_probs[:, [2, 1, 0]], labels=["A", "D", "H"])),
        "brier_score": float(np.mean(np.sum((result_probs - y_true) ** 2, axis=1))),
    }


def backtest_euro_2024(feature_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, Pipeline]:
    train_mask = feature_data["date"].lt(EURO_2024_START)
    euro_mask = feature_data["tournament"].eq("UEFA Euro") & feature_data["date"].ge(EURO_2024_START)
    home_model, away_model = train_models(feature_data, train_mask)
    euro = feature_data[euro_mask].copy()
    X_euro = euro[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    euro["pred_home_goals"] = home_model.predict(X_euro)
    euro["pred_away_goals"] = away_model.predict(X_euro)
    probabilities = [
        poisson_result_probabilities(float(h), float(a))
        for h, a in zip(euro["pred_home_goals"], euro["pred_away_goals"])
    ]
    euro[["p_home", "p_draw", "p_away"]] = pd.DataFrame(probabilities, index=euro.index)
    euro["predicted_result"] = euro[["p_home", "p_draw", "p_away"]].idxmax(axis=1).str.replace("p_", "").map({"home": "H", "draw": "D", "away": "A"})
    metrics = pd.DataFrame([evaluate_predictions(euro, euro["pred_home_goals"].to_numpy(), euro["pred_away_goals"].to_numpy())])
    return euro, metrics, home_model, away_model


def predict_match_rows(df: pd.DataFrame, home_model: Pipeline, away_model: Pipeline) -> pd.DataFrame:
    predicted = df.copy()
    X = predicted[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predicted["pred_home_goals"] = home_model.predict(X)
    predicted["pred_away_goals"] = away_model.predict(X)
    probabilities = [
        poisson_result_probabilities(float(h), float(a))
        for h, a in zip(predicted["pred_home_goals"], predicted["pred_away_goals"])
    ]
    predicted[["p_home", "p_draw", "p_away"]] = pd.DataFrame(probabilities, index=predicted.index)
    predicted["predicted_result"] = (
        predicted[["p_home", "p_draw", "p_away"]]
        .idxmax(axis=1)
        .str.replace("p_", "")
        .map({"home": "H", "draw": "D", "away": "A"})
    )
    return predicted


def backtest_competition_window(feature_data: pd.DataFrame, spec: dict[str, str]) -> tuple[pd.DataFrame, dict[str, float | str]]:
    start = pd.Timestamp(spec["start"])
    end = pd.Timestamp(spec["end"])
    train_mask = feature_data["date"].lt(start)
    test_mask = (
        feature_data["tournament"].eq(spec["tournament"])
        & feature_data["date"].between(start, end, inclusive="both")
    )

    test = feature_data[test_mask].copy()
    if test.empty:
        metrics: dict[str, float | str] = {
            "competition": spec["name"],
            "region": spec["region"],
            "tournament": spec["tournament"],
            "start": spec["start"],
            "end": spec["end"],
            "matches": 0.0,
            "home_goals_rmse": np.nan,
            "away_goals_rmse": np.nan,
            "result_accuracy": np.nan,
            "multiclass_log_loss": np.nan,
            "brier_score": np.nan,
        }
        return test, metrics

    home_model, away_model = train_models(feature_data, train_mask)
    predictions = predict_match_rows(test, home_model, away_model)
    metrics = evaluate_predictions(
        predictions,
        predictions["pred_home_goals"].to_numpy(),
        predictions["pred_away_goals"].to_numpy(),
    )
    metrics.update(
        {
            "competition": spec["name"],
            "region": spec["region"],
            "tournament": spec["tournament"],
            "start": spec["start"],
            "end": spec["end"],
        }
    )
    return predictions, metrics


def backtest_major_competitions(feature_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    metric_rows = []

    for spec in COMPETITION_BACKTESTS:
        print(f"Backtesting {spec['name']}...")
        predictions, metrics = backtest_competition_window(feature_data, spec)
        if not predictions.empty:
            predictions = predictions.copy()
            predictions["backtest_competition"] = spec["name"]
            predictions["backtest_region"] = spec["region"]
            prediction_frames.append(predictions)
        metric_rows.append(metrics)

    all_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    ordered_columns = [
        "competition",
        "region",
        "tournament",
        "start",
        "end",
        "matches",
        "home_goals_rmse",
        "away_goals_rmse",
        "result_accuracy",
        "multiclass_log_loss",
        "brier_score",
    ]
    return all_predictions, metrics[ordered_columns]


def latest_state_snapshot(results: pd.DataFrame, cutoff_date: pd.Timestamp = WORLD_CUP_2026_START) -> tuple[pd.DataFrame, dict[str, TeamState]]:
    cutoff_results = results[results["date"].lt(cutoff_date)].copy()
    return build_feature_dataset(cutoff_results)


def blank_match_row(team_1: Team, team_2: Team, states: dict[str, TeamState], match_date: pd.Timestamp, tournament: str, neutral: bool = True, country: str = "United States") -> pd.DataFrame:
    home_state = states.get(normalize_team_name(team_1.name), TeamState())
    away_state = states.get(normalize_team_name(team_2.name), TeamState())
    row = {
        "date": match_date,
        "home_team": normalize_team_name(team_1.name),
        "away_team": normalize_team_name(team_2.name),
        "tournament": tournament,
        "tournament_category": tournament_category(tournament),
        "competition_weight": competition_weight(tournament),
        "country": country,
        "neutral": int(neutral),
        "home_host": int(team_1.name in HOST_TEAM_COUNTRY and HOST_TEAM_COUNTRY[team_1.name] == country),
        "away_host": int(team_2.name in HOST_TEAM_COUNTRY and HOST_TEAM_COUNTRY[team_2.name] == country),
    }
    row.update(state_features("home", home_state, match_date))
    row.update(state_features("away", away_state, match_date))
    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    row["avg_gf_diff"] = row["home_avg_goals_for"] - row["away_avg_goals_for"]
    row["avg_ga_diff"] = row["home_avg_goals_against"] - row["away_avg_goals_against"]
    row["avg_gd_diff"] = row["home_avg_goal_difference"] - row["away_avg_goal_difference"]
    row["ppg_diff"] = row["home_points_per_match"] - row["away_points_per_match"]
    row["recent_gf_diff"] = row["home_recent_avg_goals_for"] - row["away_recent_avg_goals_for"]
    row["recent_ga_diff"] = row["home_recent_avg_goals_against"] - row["away_recent_avg_goals_against"]
    row["recent_ppg_diff"] = row["home_recent_points_per_match"] - row["away_recent_points_per_match"]
    row["rest_days_diff"] = row["home_rest_days"] - row["away_rest_days"]
    return pd.DataFrame([row])


def trained_expected_goals(home_model: Pipeline, away_model: Pipeline, states: dict[str, TeamState], team_1: Team, team_2: Team, match_date: pd.Timestamp, country: str) -> tuple[float, float]:
    row = blank_match_row(team_1, team_2, states, match_date, "FIFA World Cup", neutral=True, country=country)
    X = row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    lambda_1 = float(home_model.predict(X)[0])
    lambda_2 = float(away_model.predict(X)[0])
    return apply_current_availability_adjustments(team_1.name, team_2.name, lambda_1, lambda_2)


def apply_current_availability_adjustments(team_1_name: str, team_2_name: str, lambda_1: float, lambda_2: float) -> tuple[float, float]:
    team_1_adjustment = CURRENT_AVAILABILITY_ADJUSTMENTS.get(team_1_name, {})
    team_2_adjustment = CURRENT_AVAILABILITY_ADJUSTMENTS.get(team_2_name, {})

    lambda_1 *= float(team_1_adjustment.get("attack_multiplier", 1.0))
    lambda_2 *= float(team_2_adjustment.get("attack_multiplier", 1.0))
    lambda_2 *= float(team_1_adjustment.get("opponent_attack_multiplier", 1.0))
    lambda_1 *= float(team_2_adjustment.get("opponent_attack_multiplier", 1.0))
    return max(0.05, lambda_1), max(0.05, lambda_2)


def print_current_availability_notes() -> None:
    if not CURRENT_AVAILABILITY_ADJUSTMENTS:
        print("Current availability overlay: none.")
        return

    print("Current availability overlay:")
    for team_name, adjustment in sorted(CURRENT_AVAILABILITY_ADJUSTMENTS.items()):
        attack_multiplier = float(adjustment.get("attack_multiplier", 1.0))
        opponent_attack_multiplier = float(adjustment.get("opponent_attack_multiplier", 1.0))
        note = str(adjustment.get("note", "No note provided."))
        print(
            f"  - {team_name}: attack x {attack_multiplier:.3f}, "
            f"opponent attack x {opponent_attack_multiplier:.3f}. {note}"
        )


def simulate_trained_match(
    home_model: Pipeline,
    away_model: Pipeline,
    states: dict[str, TeamState],
    team_1: Team,
    team_2: Team,
    match_date: pd.Timestamp,
    country: str,
    rng: np.random.Generator,
    allow_draw: bool,
    xg_cache: dict[tuple[str, str, str, str], tuple[float, float]] | None = None,
) -> tuple[int, int, Team | None, Team | None, str]:
    cache_key = (team_1.name, team_2.name, str(match_date.date()), country)
    if xg_cache is not None and cache_key in xg_cache:
        lambda_1, lambda_2 = xg_cache[cache_key]
    else:
        lambda_1, lambda_2 = trained_expected_goals(home_model, away_model, states, team_1, team_2, match_date, country)
        if xg_cache is not None:
            xg_cache[cache_key] = (lambda_1, lambda_2)
    goals_1 = int(rng.poisson(lambda_1))
    goals_2 = int(rng.poisson(lambda_2))
    decided_by = "normal_time"

    if goals_1 == goals_2 and allow_draw:
        return goals_1, goals_2, None, None, "draw"

    if goals_1 == goals_2:
        goals_1 += int(rng.poisson(lambda_1 * 0.32))
        goals_2 += int(rng.poisson(lambda_2 * 0.32))
        decided_by = "extra_time"

    if goals_1 == goals_2:
        p_team_1 = team_1.penalty_rating / (team_1.penalty_rating + team_2.penalty_rating)
        winner = team_1 if rng.random() < p_team_1 else team_2
        loser = team_2 if winner == team_1 else team_1
        return goals_1, goals_2, winner, loser, "penalties"

    winner = team_1 if goals_1 > goals_2 else team_2
    loser = team_2 if winner == team_1 else team_1
    return goals_1, goals_2, winner, loser, decided_by


def rank_group_table(table: dict[str, dict[str, int]]) -> list[str]:
    return sorted(
        table,
        key=lambda team: (
            -table[team]["points"],
            -(table[team]["goals_for"] - table[team]["goals_against"]),
            -table[team]["goals_for"],
            team,
        ),
    )


def run_trained_world_cup_monte_carlo(home_model: Pipeline, away_model: Pipeline, states: dict[str, TeamState], n_simulations: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    team_templates = {team.name: team for team in create_teams()}
    counters = defaultdict(lambda: defaultdict(int))
    xg_cache: dict[tuple[str, str, str, str], tuple[float, float]] = {}

    for _ in range(n_simulations):
        teams = {name: Team(**team.__dict__) for name, team in team_templates.items()}
        qualified: dict[str, Team | list] = {}
        third_rows = []

        for group, names in GROUPS.items():
            table = {
                name: {"points": 0, "goals_for": 0, "goals_against": 0}
                for name in names
            }
            for idx, (team_1_name, team_2_name) in enumerate([(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]):
                team_1 = teams[team_1_name]
                team_2 = teams[team_2_name]
                stadium = list(STADIUMS.values())[(ord(group) - ord("A") + idx) % len(STADIUMS)]
                match_date = pd.Timestamp(date(2026, 6, 11) + timedelta(days=idx * 3))
                goals_1, goals_2, _, _, _ = simulate_trained_match(
                    home_model,
                    away_model,
                    states,
                    team_1,
                    team_2,
                    match_date,
                    stadium.country,
                    rng,
                    allow_draw=True,
                    xg_cache=xg_cache,
                )
                table[team_1.name]["goals_for"] += goals_1
                table[team_1.name]["goals_against"] += goals_2
                table[team_2.name]["goals_for"] += goals_2
                table[team_2.name]["goals_against"] += goals_1
                if goals_1 > goals_2:
                    table[team_1.name]["points"] += 3
                elif goals_2 > goals_1:
                    table[team_2.name]["points"] += 3
                else:
                    table[team_1.name]["points"] += 1
                    table[team_2.name]["points"] += 1

            ranked = rank_group_table(table)
            counters[ranked[0]]["Win group"] += 1
            counters[ranked[0]]["Reach Round of 32"] += 1
            counters[ranked[1]]["Reach Round of 32"] += 1
            qualified[f"1{group}"] = teams[ranked[0]]
            qualified[f"2{group}"] = teams[ranked[1]]
            third_rows.append(
                (
                    table[ranked[2]]["points"],
                    table[ranked[2]]["goals_for"] - table[ranked[2]]["goals_against"],
                    table[ranked[2]]["goals_for"],
                    teams[ranked[2]],
                )
            )

        third_rows = sorted(third_rows, key=lambda item: (-item[0], -item[1], -item[2], item[3].name))[:8]
        qualified["third_place_rows"] = [type("ThirdRow", (), {"team": item[3]}) for item in third_rows]
        for _, _, _, team in third_rows:
            counters[team.name]["Reach Round of 32"] += 1

        third_slots = [slot for _, _, slot in ROUND_OF_32_TEMPLATE if slot.startswith("3:")]
        third_assignments = assign_third_place_teams(qualified["third_place_rows"], third_slots)
        winners = {}
        losers = {}
        context_index = 0

        for match_id, slot_1, slot_2 in ROUND_OF_32_TEMPLATE:
            team_1 = qualified[slot_1] if not slot_1.startswith("3:") else third_assignments[slot_1]
            team_2 = qualified[slot_2] if not slot_2.startswith("3:") else third_assignments[slot_2]
            context = knockout_context(context_index)
            _, _, winner, loser, _ = simulate_trained_match(
                home_model,
                away_model,
                states,
                team_1,
                team_2,
                pd.Timestamp(context.match_date),
                context.stadium.country,
                rng,
                allow_draw=False,
                xg_cache=xg_cache,
            )
            winners[match_id] = winner
            losers[match_id] = loser
            counters[winner.name]["Reach Round of 16"] += 1
            context_index += 1

        round_map = {
            "Round of 16": ("Reach Quarterfinal", NEXT_ROUNDS["Round of 16"]),
            "Quarterfinal": ("Reach Semifinal", NEXT_ROUNDS["Quarterfinal"]),
            "Semifinal": ("Reach Final", NEXT_ROUNDS["Semifinal"]),
        }
        for _, (counter_name, fixtures) in round_map.items():
            for match_id, source_1, source_2 in fixtures:
                context = knockout_context(context_index)
                _, _, winner, loser, _ = simulate_trained_match(
                    home_model,
                    away_model,
                    states,
                    winners[source_1],
                    winners[source_2],
                    pd.Timestamp(context.match_date),
                    context.stadium.country,
                    rng,
                allow_draw=False,
                xg_cache=xg_cache,
            )
                winners[match_id] = winner
                losers[match_id] = loser
                counters[winner.name][counter_name] += 1
                context_index += 1

        _, _, third, _, _ = simulate_trained_match(
            home_model,
            away_model,
            states,
            losers[101],
            losers[102],
            pd.Timestamp("2026-07-18"),
            "United States",
            rng,
            allow_draw=False,
            xg_cache=xg_cache,
        )
        counters[third.name]["Finish third"] += 1

        _, _, champion, runner_up, _ = simulate_trained_match(
            home_model,
            away_model,
            states,
            winners[101],
            winners[102],
            pd.Timestamp("2026-07-19"),
            "United States",
            rng,
            allow_draw=False,
            xg_cache=xg_cache,
        )
        counters[champion.name]["Win World Cup"] += 1
        counters[runner_up.name]["Finish runner-up"] += 1

    rows = []
    for team in create_teams():
        rows.append(
            {
                "Team": team.name,
                "Group": team.group,
                "Win group": counters[team.name]["Win group"] / n_simulations,
                "Reach Round of 32": counters[team.name]["Reach Round of 32"] / n_simulations,
                "Reach Round of 16": counters[team.name]["Reach Round of 16"] / n_simulations,
                "Reach Quarterfinal": counters[team.name]["Reach Quarterfinal"] / n_simulations,
                "Reach Semifinal": counters[team.name]["Reach Semifinal"] / n_simulations,
                "Reach Final": counters[team.name]["Reach Final"] / n_simulations,
                "Finish runner-up": counters[team.name]["Finish runner-up"] / n_simulations,
                "Finish third": counters[team.name]["Finish third"] / n_simulations,
                "Win World Cup": counters[team.name]["Win World Cup"] / n_simulations,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["Win World Cup", "Reach Final"], ascending=False)
    summary.to_csv(OUTPUT_DIR / "trained_world_cup_2026_summary.csv", index=False)
    return summary


def main() -> None:
    refresh = os.getenv("REFRESH_INTERNATIONAL_RESULTS", "0") == "1"
    print("Loading international results...")
    results = load_international_results(refresh=refresh)
    print(f"Loaded {len(results):,} international matches from {results['date'].min().date()} to {results['date'].max().date()}.")

    print("Building no-leakage pre-match features...")
    feature_data, _ = build_feature_dataset(results)
    feature_data.to_csv(OUTPUT_DIR / "trained_international_feature_data.csv", index=False)
    print(f"Feature rows: {len(feature_data):,}")

    print("Backtesting major international competitions by region...")
    competition_predictions, competition_metrics = backtest_major_competitions(feature_data)
    competition_predictions.to_csv(OUTPUT_DIR / "trained_international_competition_backtest_predictions.csv", index=False)
    competition_metrics.to_csv(OUTPUT_DIR / "trained_international_competition_backtest_metrics.csv", index=False)
    print(competition_metrics.to_string(index=False))

    print("Training on pre-Euro 2024 matches and backtesting Euro 2024...")
    euro_predictions, euro_metrics, _, _ = backtest_euro_2024(feature_data)
    euro_predictions.to_csv(OUTPUT_DIR / "trained_international_euro_2024_predictions.csv", index=False)
    euro_metrics.to_csv(OUTPUT_DIR / "trained_international_euro_2024_metrics.csv", index=False)
    print(euro_metrics.to_string(index=False))

    print("Training final model on all pre-World Cup 2026 data...")
    final_train_mask = feature_data["date"].lt(WORLD_CUP_2026_START)
    home_model, away_model = train_models(feature_data, final_train_mask)
    _, final_states = latest_state_snapshot(results, WORLD_CUP_2026_START)
    print_current_availability_notes()

    n_simulations = int(os.getenv("TRAINED_WC_SIMULATIONS", "5000"))
    print(f"Running trained-model World Cup forecast ({n_simulations:,} tournaments)...")
    wc_summary = run_trained_world_cup_monte_carlo(home_model, away_model, final_states, n_simulations=n_simulations)
    probability_cols = [col for col in wc_summary.columns if col not in {"Team", "Group"}]
    print(
        wc_summary[["Team", "Group", "Win World Cup", "Reach Final", "Reach Semifinal", "Reach Quarterfinal"]]
        .head(16)
        .to_string(index=False, formatters={col: "{:.2%}".format for col in probability_cols})
    )
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
