"""Lightweight validation hooks for the Game 7 project.

With only six head-to-head games available, this is not a real robust backtest.
These functions are intentionally conservative and report that limitation.
"""

import numpy as np
import pandas as pd

from config import BASE_PARAMS, OUTPUT_DIR, SPURS, THUNDER
from feature_engineering import estimate_game_means, team_summary


def validation_warning(game_logs: pd.DataFrame) -> str:
    games = len(game_logs) // 2
    if games < 20:
        return (
            f"Only {games} completed games are available. No reliable historical "
            "parameter validation was possible; results should be treated as "
            "scenario analysis, not a validated betting model."
        )
    return "Enough rows exist for a basic validation pass, but this template keeps validation lightweight."


def _games_from_team_logs(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Collapse two team-game rows into one game row."""

    game_logs = game_logs.copy()
    if "game_id" not in game_logs.columns:
        game_logs["game_id"] = (
            game_logs["date"].astype(str)
            + "_"
            + game_logs["team"].astype(str).where(game_logs["home_away"].eq("home"), game_logs["opponent"].astype(str))
        )

    games = []
    for game_id, group in game_logs.groupby("game_id", sort=True):
        if len(group) < 2:
            continue
        home_rows = group[group["home_away"].eq("home")]
        away_rows = group[group["home_away"].eq("away")]
        if home_rows.empty or away_rows.empty:
            continue
        home = home_rows.iloc[0]
        away = away_rows.iloc[0]
        games.append(
            {
                "game_id": game_id,
                "date": home["date"],
                "home_team": home["team"],
                "away_team": away["team"],
                "home_points": float(home["points_for"]),
                "away_points": float(away["points_for"]),
            }
        )
    return pd.DataFrame(games)


def _generic_team_strength(train_logs: pd.DataFrame, team: str, params: dict) -> dict:
    group = train_logs[train_logs["team"].eq(team)].sort_values("date")
    league_avg_points = pd.to_numeric(train_logs["points_for"], errors="coerce").mean()
    league_avg_pace = pd.to_numeric(train_logs["pace"], errors="coerce").mean()
    if pd.isna(league_avg_points):
        league_avg_points = 110.0
    if pd.isna(league_avg_pace):
        league_avg_pace = 100.0

    recent = group.tail(int(params["recent_form_window"]))
    season_for = pd.to_numeric(group["points_for"], errors="coerce").mean()
    season_against = pd.to_numeric(group["points_against"], errors="coerce").mean()
    recent_for = pd.to_numeric(recent["points_for"], errors="coerce").mean()
    recent_against = pd.to_numeric(recent["points_against"], errors="coerce").mean()
    pace = pd.to_numeric(group["pace"], errors="coerce").mean()

    season_for = league_avg_points if pd.isna(season_for) else season_for
    season_against = league_avg_points if pd.isna(season_against) else season_against
    recent_for = season_for if pd.isna(recent_for) else recent_for
    recent_against = season_against if pd.isna(recent_against) else recent_against
    pace = league_avg_pace if pd.isna(pace) else pace

    season_weight = params["season_weight"]
    recent_weight = params["recent_weight"]
    blend_total = max(0.01, season_weight + recent_weight)
    return {
        "points_for": (season_weight * season_for + recent_weight * recent_for) / blend_total,
        "points_against": (season_weight * season_against + recent_weight * recent_against) / blend_total,
        "pace": pace,
    }


def _predict_game_from_training_logs(
    train_logs: pd.DataFrame,
    injuries: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict:
    """Predict one held-out game using the same team-strength ingredients.

    This respects the actual home team for the held-out game. It excludes the
    Game 7 referee adjustment because those refs are a future-game assumption,
    not something we can fairly backtest against earlier games.
    """

    params = BASE_PARAMS.copy()
    home = _generic_team_strength(train_logs, home_team, params)
    away = _generic_team_strength(train_logs, away_team, params)

    offense_weight = params["offense_weight"]
    defense_weight = params["defense_weight"]
    weight_total = max(0.01, offense_weight + defense_weight)
    pace = np.mean([home["pace"], away["pace"]])
    pace_adjustment = (pace - 100.0) * params["pace_weight"]

    home_points = (
        offense_weight * home["points_for"] + defense_weight * away["points_against"]
    ) / weight_total
    away_points = (
        offense_weight * away["points_for"] + defense_weight * home["points_against"]
    ) / weight_total
    home_points += pace_adjustment + params["home_court_advantage_points"]
    away_points += pace_adjustment

    predicted_winner = home_team if home_points > away_points else away_team
    return {
        "predicted_home_points": home_points,
        "predicted_away_points": away_points,
        "predicted_total": home_points + away_points,
        "predicted_home_margin": home_points - away_points,
        "predicted_winner": predicted_winner,
    }


def leave_one_game_out_backtest(game_logs: pd.DataFrame, injuries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically predict each game using only earlier games."""

    games = _games_from_team_logs(game_logs)
    rows = []

    for _, game in games.iterrows():
        train_logs = game_logs[pd.to_datetime(game_logs["date"]) < pd.to_datetime(game["date"])].copy()
        prior_counts = train_logs.groupby("team").size()
        min_games = BASE_PARAMS["minimum_prior_games_for_backtest"]
        if prior_counts.get(game["home_team"], 0) < min_games or prior_counts.get(game["away_team"], 0) < min_games:
            continue

        pred = _predict_game_from_training_logs(train_logs, injuries, game["home_team"], game["away_team"])

        actual_winner = game["home_team"] if game["home_points"] > game["away_points"] else game["away_team"]
        actual_total = game["home_points"] + game["away_points"]
        actual_home_margin = game["home_points"] - game["away_points"]

        rows.append(
            {
                "game_id": game["game_id"],
                "date": game["date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "actual_home_points": game["home_points"],
                "actual_away_points": game["away_points"],
                "actual_winner": actual_winner,
                **pred,
                "winner_correct": pred["predicted_winner"] == actual_winner,
                "actual_total": actual_total,
                "total_error": pred["predicted_total"] - actual_total,
                "absolute_total_error": abs(pred["predicted_total"] - actual_total),
                "actual_home_margin": actual_home_margin,
                "margin_error": pred["predicted_home_margin"] - actual_home_margin,
                "absolute_margin_error": abs(pred["predicted_home_margin"] - actual_home_margin),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        summary = pd.DataFrame(
            [
                {
                    "games_tested": 0,
                    "winner_accuracy": np.nan,
                    "total_mae": np.nan,
                    "total_rmse": np.nan,
                    "margin_mae": np.nan,
                    "margin_rmse": np.nan,
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "games_tested": len(results),
                    "winner_accuracy": results["winner_correct"].mean(),
                    "total_mae": results["absolute_total_error"].mean(),
                    "total_rmse": np.sqrt((results["total_error"] ** 2).mean()),
                    "margin_mae": results["absolute_margin_error"].mean(),
                    "margin_rmse": np.sqrt((results["margin_error"] ** 2).mean()),
                }
            ]
        )

    results.to_csv(OUTPUT_DIR / "spurs_thunder_leave_one_game_backtest.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "spurs_thunder_leave_one_game_backtest_summary.csv", index=False)
    return results, summary
