"""Feature engineering for team-strength based NBA simulation."""

import numpy as np
import pandas as pd

from config import BASE_PARAMS, SPURS, THUNDER


def _safe_mean(series: pd.Series, fallback: float) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()
    return fallback if pd.isna(value) else float(value)


def team_summary(game_logs: pd.DataFrame, injuries: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Summarize team strength from available game logs."""

    params = params or BASE_PARAMS
    recent_window = int(params["recent_form_window"])
    season_weight = float(params["season_weight"])
    recent_weight = float(params["recent_weight"])
    weight_total = max(0.01, season_weight + recent_weight)

    rows = []
    league_avg_points = _safe_mean(game_logs["points_for"], 110.0)
    league_avg_pace = _safe_mean(game_logs["pace"], 100.0)

    for team in [SPURS, THUNDER]:
        group = game_logs[game_logs["team"].eq(team)].copy()
        if group.empty:
            group = game_logs.copy()

        injury_impact = (
            pd.to_numeric(injuries[injuries["team"].eq(team)]["estimated_impact_points"], errors="coerce")
            .fillna(0)
            .sum()
        )

        group = group.sort_values("date")
        recent = group.tail(recent_window)

        season_points_for = _safe_mean(group["points_for"], league_avg_points)
        season_points_against = _safe_mean(group["points_against"], league_avg_points)
        recent_points_for = _safe_mean(recent["points_for"], season_points_for)
        recent_points_against = _safe_mean(recent["points_against"], season_points_against)

        points_for = (season_weight * season_points_for + recent_weight * recent_points_for) / weight_total
        points_against = (
            season_weight * season_points_against + recent_weight * recent_points_against
        ) / weight_total
        net_rating = points_for - points_against
        pace = _safe_mean(group["pace"], league_avg_pace)
        rest_days = _safe_mean(group["rest_days"], 2.0)

        rows.append(
            {
                "team": team,
                "games": len(group),
                "season_points_for": season_points_for,
                "season_points_against": season_points_against,
                "recent_points_for": recent_points_for,
                "recent_points_against": recent_points_against,
                "points_for": points_for,
                "points_against": points_against,
                "net_rating": net_rating,
                "pace": pace,
                "rest_days": rest_days,
                "injury_impact_points": injury_impact,
            }
        )

    return pd.DataFrame(rows)


def estimate_game_means(team_features: pd.DataFrame, params: dict, ref_adjustments: dict) -> dict:
    """Estimate expected Spurs/Thunder points, total, and Spurs margin."""

    spurs = team_features[team_features["team"].eq(SPURS)].iloc[0]
    thunder = team_features[team_features["team"].eq(THUNDER)].iloc[0]

    offense_weight = params["offense_weight"]
    defense_weight = params["defense_weight"]
    weight_total = max(0.01, offense_weight + defense_weight)
    pace_weight = params["pace_weight"]

    # Normalize weights so expected points stay on an NBA scoring scale even
    # when the optimizer perturbs offense/defense weights.
    spurs_expected_offense = (
        offense_weight * spurs["points_for"] + defense_weight * thunder["points_against"]
    ) / weight_total
    thunder_expected_offense = (
        offense_weight * thunder["points_for"] + defense_weight * spurs["points_against"]
    ) / weight_total

    avg_pace = np.mean([spurs["pace"], thunder["pace"]])
    pace_adjustment = (avg_pace - 100.0) * pace_weight

    rest_adjustment_spurs = (spurs["rest_days"] - thunder["rest_days"]) * 0.25
    rest_adjustment_thunder = -rest_adjustment_spurs

    injury_multiplier = params["injury_impact_multiplier"]
    spurs_injury_adj = -spurs["injury_impact_points"] * injury_multiplier
    thunder_injury_adj = -thunder["injury_impact_points"] * injury_multiplier

    thunder_home_adj = params["home_court_advantage_points"]
    okc_ref_margin_adj = ref_adjustments["okc_ref_margin_adj"]
    ref_total_points_adj = ref_adjustments["ref_total_points_adj"]

    base_total = spurs_expected_offense + thunder_expected_offense + 2 * pace_adjustment
    base_spurs_margin = spurs_expected_offense - thunder_expected_offense

    adjusted_total = (
        base_total
        + rest_adjustment_spurs
        + rest_adjustment_thunder
        + spurs_injury_adj
        + thunder_injury_adj
        + ref_total_points_adj
    )
    adjusted_spurs_margin = (
        base_spurs_margin
        + rest_adjustment_spurs
        - rest_adjustment_thunder
        + spurs_injury_adj
        - thunder_injury_adj
        - thunder_home_adj
        - okc_ref_margin_adj
    )

    spurs_points = (adjusted_total + adjusted_spurs_margin) / 2
    thunder_points = (adjusted_total - adjusted_spurs_margin) / 2

    spurs_points = max(80.0, float(spurs_points))
    thunder_points = max(80.0, float(thunder_points))

    return {
        "spurs_expected_points": spurs_points,
        "thunder_expected_points": thunder_points,
        "expected_total": spurs_points + thunder_points,
        "expected_margin_spurs_minus_thunder": spurs_points - thunder_points,
    }
