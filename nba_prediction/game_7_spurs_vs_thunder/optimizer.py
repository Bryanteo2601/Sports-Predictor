"""Simple parameter robustness and stability ranking."""

import pandas as pd

from market import evaluate_markets


def summarize_market_stability(all_sims: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Evaluate markets by model run and summarize stability."""

    per_run_results = []
    for run_id, sim in all_sims.groupby("model_run"):
        edge = evaluate_markets(sim, odds)
        edge["model_run"] = run_id
        edge["rank_in_run"] = edge["expected_value_per_1"].rank(ascending=False, method="first")
        per_run_results.append(edge)

    per_run = pd.concat(per_run_results, ignore_index=True)
    key_cols = ["market_type", "selection", "line"]
    summary = (
        per_run.groupby(key_cols, dropna=False)
        .agg(
            avg_expected_value=("expected_value_per_1", "mean"),
            avg_edge=("edge", "mean"),
            positive_ev_rate=("expected_value_per_1", lambda x: (x > 0).mean()),
            best_ev_frequency=("rank_in_run", lambda x: (x == 1).mean()),
            model_probability_mean=("model_probability", "mean"),
            model_probability_std=("model_probability", "std"),
        )
        .reset_index()
    )
    summary["stability_score"] = (
        summary["positive_ev_rate"] * 0.45
        + summary["best_ev_frequency"] * 0.35
        + (1 - summary["model_probability_std"].fillna(0).clip(0, 1)) * 0.20
    )
    return summary.sort_values(
        ["avg_expected_value", "avg_edge", "stability_score"], ascending=False
    ).reset_index(drop=True)
