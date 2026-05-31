"""Market probability and expected-value calculations."""

import numpy as np
import pandas as pd


def add_no_vig_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    odds = odds.copy()
    odds["raw_implied_probability"] = 1 / odds["odds_decimal"]
    odds["market_group"] = odds.apply(
        lambda row: abs(row["line"]) if row["market_type"] == "Handicap" else row["line"],
        axis=1,
    )
    grouped = []
    for key, group in odds.groupby(["market_type", "market_group"], dropna=False):
        group = group.copy()
        group["no_vig_probability"] = group["raw_implied_probability"] / group["raw_implied_probability"].sum()
        grouped.append(group)
    return pd.concat(grouped, ignore_index=True).drop(columns=["market_group"])


def _market_probability(sim: pd.DataFrame, market_type: str, selection: str, line: float | None) -> tuple[float, float]:
    spurs_reg = sim["spurs_score_regulation"]
    thunder_reg = sim["thunder_score_regulation"]
    total = sim["total_points"]
    margin_spurs = sim["margin_spurs_minus_thunder"]

    if market_type == "Match Winner":
        wins = sim["winner"].eq("Spurs") if "Spurs" in selection else sim["winner"].eq("Thunder")
        return float(wins.mean()), 0.0

    if market_type == "Regular Time Result":
        if selection == "Draw":
            wins = spurs_reg.eq(thunder_reg)
        elif "Spurs" in selection:
            wins = spurs_reg.gt(thunder_reg)
        else:
            wins = thunder_reg.gt(spurs_reg)
        return float(wins.mean()), 0.0

    if market_type == "Handicap":
        line = float(line)
        adjusted_margin = margin_spurs + line
        if "Spurs" in selection:
            wins = adjusted_margin.gt(0)
            pushes = adjusted_margin.eq(0)
        else:
            # Thunder -x is equivalent to Spurs margin + (-x) < 0.
            wins = adjusted_margin.lt(0)
            pushes = adjusted_margin.eq(0)
        return float(wins.mean()), float(pushes.mean())

    if market_type == "Total Points":
        line = float(line)
        if selection.startswith("Over"):
            wins = total.gt(line)
            pushes = total.eq(line)
        else:
            wins = total.lt(line)
            pushes = total.eq(line)
        return float(wins.mean()), float(pushes.mean())

    raise ValueError(f"Unknown market type: {market_type}")


def evaluate_markets(sim: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    odds = add_no_vig_probabilities(odds)
    rows = []

    for _, row in odds.iterrows():
        model_probability, push_probability = _market_probability(
            sim,
            row["market_type"],
            row["selection"],
            None if pd.isna(row["line"]) else row["line"],
        )
        loss_probability = 1 - model_probability - push_probability
        expected_value = model_probability * (row["odds_decimal"] - 1) - loss_probability
        edge = model_probability - row["no_vig_probability"]

        rows.append(
            {
                "market_type": row["market_type"],
                "selection": row["selection"],
                "line": row["line"],
                "odds_decimal": row["odds_decimal"],
                "raw_implied_probability": row["raw_implied_probability"],
                "no_vig_probability": row["no_vig_probability"],
                "model_probability": model_probability,
                "push_probability": push_probability,
                "edge": edge,
                "expected_value_per_1": expected_value,
            }
        )

    result = pd.DataFrame(rows)
    result["edge_label"] = pd.cut(
        result["edge"],
        bins=[-np.inf, 0.01, 0.04, 0.08, np.inf],
        labels=["no-play/negative", "weak edge", "moderate edge", "strong edge"],
    )
    return result.sort_values(["expected_value_per_1", "edge"], ascending=False).reset_index(drop=True)
