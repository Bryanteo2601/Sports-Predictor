"""Monte Carlo simulation engine."""

import numpy as np
import pandas as pd

from config import (
    ALTERNATE,
    BASE_PARAMS,
    CREW_CHIEF,
    MODEL_RUNS,
    OKC_FOUL_DRAWING_SENSITIVITY,
    REFEREE,
    REF_MARGIN_POINTS_PER_SENSITIVITY,
    REF_TOTAL_POINTS_PER_SENSITIVITY,
    SCORE_SIMULATIONS_PER_RUN,
    UMPIRE,
)
from feature_engineering import estimate_game_means


def referee_adjustments() -> dict:
    return {
        "crew_chief": CREW_CHIEF,
        "referee": REFEREE,
        "umpire": UMPIRE,
        "alternate": ALTERNATE,
        "okc_foul_drawing_sensitivity": OKC_FOUL_DRAWING_SENSITIVITY,
        "okc_ref_margin_adj": OKC_FOUL_DRAWING_SENSITIVITY * REF_MARGIN_POINTS_PER_SENSITIVITY,
        "ref_total_points_adj": OKC_FOUL_DRAWING_SENSITIVITY * REF_TOTAL_POINTS_PER_SENSITIVITY,
    }


def simulate_game(team_features: pd.DataFrame, params: dict, n_sims: int, rng: np.random.Generator) -> pd.DataFrame:
    refs = referee_adjustments()
    means = estimate_game_means(team_features, params, refs)

    total_sd = params.get("total_sd", params["score_variance"] * 1.6) * params["game7_variance_multiplier"]
    margin_sd = params.get("margin_sd", params["score_variance"] * 1.4) * params["game7_variance_multiplier"]
    shooting_noise = rng.normal(0, params["shooting_noise_sd"], n_sims)
    pace_noise = rng.normal(0, params["pace_noise_sd"], n_sims)
    turnover_noise = rng.normal(0, params["turnover_noise_sd"], n_sims)

    sim_total = means["expected_total"] + rng.normal(0, total_sd, n_sims) + pace_noise
    sim_margin = (
        means["expected_margin_spurs_minus_thunder"]
        + rng.normal(0, margin_sd, n_sims)
        + shooting_noise
        - turnover_noise
    )

    spurs_score = (sim_total + sim_margin) / 2
    thunder_score = (sim_total - sim_margin) / 2

    spurs_reg = np.maximum(70, np.rint(spurs_score)).astype(int)
    thunder_reg = np.maximum(70, np.rint(thunder_score)).astype(int)

    regulation_tie = spurs_reg == thunder_reg
    overtime_edge = 0.52
    thunder_ot_wins = rng.random(regulation_tie.sum()) < overtime_edge

    spurs_final = spurs_reg.copy()
    thunder_final = thunder_reg.copy()
    spurs_final[regulation_tie] += (~thunder_ot_wins).astype(int)
    thunder_final[regulation_tie] += thunder_ot_wins.astype(int)

    simulation = pd.DataFrame(
        {
            "spurs_score_regulation": spurs_reg,
            "thunder_score_regulation": thunder_reg,
            "spurs_score_final": spurs_final,
            "thunder_score_final": thunder_final,
            "regular_time_draw": regulation_tie,
        }
    )
    simulation["total_points"] = simulation["spurs_score_regulation"] + simulation["thunder_score_regulation"]
    simulation["margin_spurs_minus_thunder"] = (
        simulation["spurs_score_regulation"] - simulation["thunder_score_regulation"]
    )
    simulation["winner"] = np.where(simulation["spurs_score_final"] > simulation["thunder_score_final"], "Spurs", "Thunder")
    return simulation


def perturb_params(base_params: dict, rng: np.random.Generator) -> dict:
    params = base_params.copy()
    params["home_court_advantage_points"] += rng.normal(0, 0.75)
    params["injury_impact_multiplier"] = max(0, params["injury_impact_multiplier"] + rng.normal(0, 0.20))
    params["pace_weight"] = max(0, params["pace_weight"] + rng.normal(0, 0.05))
    params["offense_weight"] = max(0.05, params["offense_weight"] + rng.normal(0, 0.07))
    params["defense_weight"] = max(0.05, params["defense_weight"] + rng.normal(0, 0.07))
    params["score_variance"] = max(6.0, params["score_variance"] + rng.normal(0, 1.5))
    params["game7_variance_multiplier"] = max(0.8, params["game7_variance_multiplier"] + rng.normal(0, 0.08))
    params["season_weight"] = float(np.clip(params["season_weight"] + rng.normal(0, 0.06), 0.45, 0.90))
    params["recent_weight"] = max(0.05, 1 - params["season_weight"])
    if "total_sd" in params:
        params["total_sd"] = max(6.0, params["total_sd"] + rng.normal(0, 1.0))
    if "margin_sd" in params:
        params["margin_sd"] = max(6.0, params["margin_sd"] + rng.normal(0, 1.0))
    return params


def run_parameter_scenarios(
    team_features: pd.DataFrame,
    model_runs: int = MODEL_RUNS,
    sims_per_run: int = SCORE_SIMULATIONS_PER_RUN,
    seed: int | None = None,
    base_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base_params = base_params or BASE_PARAMS
    all_sims = []
    run_summaries = []

    for run_id in range(1, model_runs + 1):
        params = perturb_params(base_params, rng)
        run_features = team_features.copy()
        sim = simulate_game(run_features, params, sims_per_run, rng)
        sim["model_run"] = run_id
        all_sims.append(sim)
        run_summaries.append(
            {
                "model_run": run_id,
                "spurs_win_probability": (sim["winner"] == "Spurs").mean(),
                "thunder_win_probability": (sim["winner"] == "Thunder").mean(),
                "draw_probability_regulation": sim["regular_time_draw"].mean(),
                "mean_total": sim["total_points"].mean(),
                "mean_margin_spurs_minus_thunder": sim["margin_spurs_minus_thunder"].mean(),
            }
        )

    return pd.concat(all_sims, ignore_index=True), pd.DataFrame(run_summaries)
