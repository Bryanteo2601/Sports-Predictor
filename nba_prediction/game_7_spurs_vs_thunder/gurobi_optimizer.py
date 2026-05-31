"""Gurobi/fallback portfolio optimizer for theoretical market selection.

This is educational paper-trading only. The optimizer allocates paper stake to
model-selected markets; it does not make real-money recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    GUROBI_BANKROLL,
    GUROBI_KELLY_FRACTION,
    GUROBI_MARKET_TYPE_EXPOSURE,
    GUROBI_MAX_NUM_BETS,
    GUROBI_MAX_STAKE_PER_BET,
    GUROBI_MINIMUM_EDGE_THRESHOLD,
    GUROBI_OBJECTIVE_MODE,
    GUROBI_RISK_AVERSION,
    STRICT_ONE_SELECTION_PER_MARKET_TYPE,
)


@dataclass
class OptimizerSettings:
    bankroll: float = GUROBI_BANKROLL
    max_stake_per_bet: float = GUROBI_MAX_STAKE_PER_BET
    minimum_edge_threshold: float = GUROBI_MINIMUM_EDGE_THRESHOLD
    max_num_bets: int = GUROBI_MAX_NUM_BETS
    kelly_fraction: float = GUROBI_KELLY_FRACTION
    risk_aversion: float = GUROBI_RISK_AVERSION
    objective_mode: str = GUROBI_OBJECTIVE_MODE
    market_type_exposure: dict | None = None

    def __post_init__(self) -> None:
        if self.market_type_exposure is None:
            self.market_type_exposure = GUROBI_MARKET_TYPE_EXPOSURE.copy()


def _market_group(row: pd.Series) -> str:
    if STRICT_ONE_SELECTION_PER_MARKET_TYPE:
        return str(row["market_type"])
    if row["market_type"] == "Handicap":
        return f"Handicap_{abs(float(row['line']))}"
    if row["market_type"] == "Total Points":
        return f"Total_{float(row['line'])}"
    return str(row["market_type"])


def prepare_optimizer_data(market_edges: pd.DataFrame, ranked_markets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add EV, Kelly, uncertainty, and conflict-group columns."""

    data = market_edges.copy().reset_index(drop=True)
    data["selection_id"] = np.arange(len(data))
    data["game_id"] = "Spurs_vs_Thunder_Game7"
    data["expected_value"] = data["expected_value_per_1"]

    if ranked_markets is not None and not ranked_markets.empty:
        uncertainty_cols = [
            "market_type",
            "selection",
            "line",
            "model_probability_std",
            "stability_score",
            "positive_ev_rate",
        ]
        available = [col for col in uncertainty_cols if col in ranked_markets.columns]
        data = data.merge(ranked_markets[available], on=["market_type", "selection", "line"], how="left")

    data["probability_uncertainty"] = data.get("model_probability_std", pd.Series(index=data.index, dtype=float))
    data["probability_uncertainty"] = data["probability_uncertainty"].fillna(0.03).clip(lower=0.0)
    data["uncertainty_buffer"] = data["probability_uncertainty"].clip(lower=0.01, upper=0.15)
    data["conservative_probability"] = (data["model_probability"] - data["uncertainty_buffer"]).clip(0.0, 1.0)
    data["conservative_ev"] = (
        data["conservative_probability"] * (data["odds_decimal"] - 1)
        - (1 - data["conservative_probability"] - data.get("push_probability", 0).fillna(0))
    )
    data["risk_adjusted_ev"] = data["expected_value"] - data["probability_uncertainty"] * GUROBI_RISK_AVERSION
    data["kelly"] = (
        ((data["odds_decimal"] - 1) * data["model_probability"] - (1 - data["model_probability"]))
        / (data["odds_decimal"] - 1)
    ).clip(lower=0.0)
    data["market_group"] = data.apply(_market_group, axis=1)
    return data


def _objective_column(settings: OptimizerSettings) -> str:
    if settings.objective_mode == "raw_ev":
        return "expected_value"
    if settings.objective_mode == "conservative_ev":
        return "conservative_ev"
    if settings.objective_mode == "risk_adjusted_ev":
        return "risk_adjusted_ev"
    raise ValueError("objective_mode must be raw_ev, conservative_ev, or risk_adjusted_ev")


def build_optimization_model(data: pd.DataFrame, settings: OptimizerSettings):
    """Build a Gurobi model and decision variables.

    Returns (model, x, stake). Importing gurobipy happens inside this function
    so the rest of the project can run without Gurobi installed.
    """

    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("game7_market_portfolio")
    model.Params.OutputFlag = 0

    x = {}
    stake = {}
    for i in data.index:
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        stake[i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"stake_{i}")

    add_budget_constraints(model, data, settings, x, stake)
    add_conflict_constraints(model, data, x)
    add_edge_ev_constraints(model, data, settings, x, stake)
    add_kelly_constraints(model, data, settings, stake)

    obj_col = _objective_column(settings)
    model.setObjective(gp.quicksum(stake[i] * float(data.loc[i, obj_col]) for i in data.index), GRB.MAXIMIZE)
    return model, x, stake


def add_budget_constraints(model, data: pd.DataFrame, settings: OptimizerSettings, x: dict, stake: dict) -> None:
    import gurobipy as gp

    model.addConstr(gp.quicksum(stake[i] for i in data.index) <= settings.bankroll, name="total_budget")
    model.addConstr(gp.quicksum(x[i] for i in data.index) <= settings.max_num_bets, name="max_num_bets")

    for i in data.index:
        model.addConstr(stake[i] <= settings.max_stake_per_bet * x[i], name=f"stake_only_if_selected_{i}")

    for market_type, max_exposure in settings.market_type_exposure.items():
        idx = data.index[data["market_type"].eq(market_type)].tolist()
        if idx:
            model.addConstr(gp.quicksum(stake[i] for i in idx) <= max_exposure, name=f"exposure_{market_type}")


def add_conflict_constraints(model, data: pd.DataFrame, x: dict) -> None:
    import gurobipy as gp

    for group_name, group in data.groupby("market_group", dropna=False):
        model.addConstr(gp.quicksum(x[i] for i in group.index) <= 1, name=f"conflict_{group_name}")


def add_edge_ev_constraints(model, data: pd.DataFrame, settings: OptimizerSettings, x: dict, stake: dict) -> None:
    for i, row in data.iterrows():
        allowed = row["edge"] >= settings.minimum_edge_threshold and row["expected_value"] > 0
        if not allowed:
            model.addConstr(x[i] == 0, name=f"edge_ev_block_x_{i}")
            model.addConstr(stake[i] == 0, name=f"edge_ev_block_stake_{i}")


def add_kelly_constraints(model, data: pd.DataFrame, settings: OptimizerSettings, stake: dict) -> None:
    for i, row in data.iterrows():
        kelly_cap = settings.bankroll * settings.kelly_fraction * max(0.0, float(row["kelly"]))
        model.addConstr(stake[i] <= kelly_cap, name=f"kelly_cap_{i}")


def solve_model(model):
    model.optimize()
    return model


def extract_solution(data: pd.DataFrame, model, x: dict, stake: dict, settings: OptimizerSettings) -> tuple[pd.DataFrame, dict]:
    selected_rows = []
    obj_col = _objective_column(settings)

    for i in data.index:
        stake_value = float(stake[i].X)
        if stake_value <= 1e-6:
            continue
        row = data.loc[i].to_dict()
        row["selected"] = int(round(x[i].X))
        row["stake"] = stake_value
        row["objective_ev_used"] = row[obj_col]
        row["expected_profit"] = stake_value * row["expected_value"]
        row["reason"] = (
            f"Passed edge/EV filters; {settings.objective_mode}={row[obj_col]:.4f}; "
            f"edge={row['edge']:.4f}; Kelly cap respected."
        )
        selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    return selected, _summary_from_selection(selected, data, settings, solver="Gurobi")


def _summary_from_selection(
    selected: pd.DataFrame,
    data: pd.DataFrame,
    settings: OptimizerSettings,
    solver: str,
    warning: str | None = None,
) -> dict:
    total_stake = 0.0 if selected.empty else float(selected["stake"].sum())
    expected_profit = 0.0 if selected.empty else float(selected["expected_profit"].sum())
    return {
        "solver": solver,
        "warning": warning or "",
        "objective_mode": settings.objective_mode,
        "bankroll": settings.bankroll,
        "total_stake_used": total_stake,
        "unused_bankroll": settings.bankroll - total_stake,
        "expected_profit": expected_profit,
        "expected_roi": 0.0 if total_stake == 0 else expected_profit / total_stake,
        "num_selected": 0 if selected.empty else int(len(selected)),
        "average_edge": 0.0 if selected.empty else float(selected["edge"].mean()),
        "average_ev": 0.0 if selected.empty else float(selected["expected_value"].mean()),
        "eligible_markets": int(((data["edge"] >= settings.minimum_edge_threshold) & (data["expected_value"] > 0)).sum()),
    }


def _fallback_greedy(data: pd.DataFrame, settings: OptimizerSettings, warning: str) -> tuple[pd.DataFrame, dict]:
    obj_col = _objective_column(settings)
    eligible = data[(data["edge"] >= settings.minimum_edge_threshold) & (data["expected_value"] > 0)].copy()
    eligible = eligible[eligible[obj_col] > 0].sort_values([obj_col, "edge"], ascending=False)

    selected = []
    used_groups = set()
    market_exposure = {market_type: 0.0 for market_type in settings.market_type_exposure}
    remaining_budget = settings.bankroll

    for _, row in eligible.iterrows():
        if len(selected) >= settings.max_num_bets or remaining_budget <= 0:
            break
        if row["market_group"] in used_groups:
            continue

        max_market_exposure = settings.market_type_exposure.get(row["market_type"], settings.bankroll)
        exposure_left = max_market_exposure - market_exposure.get(row["market_type"], 0.0)
        kelly_cap = settings.bankroll * settings.kelly_fraction * max(0.0, float(row["kelly"]))
        stake_value = min(settings.max_stake_per_bet, remaining_budget, exposure_left, kelly_cap)
        if stake_value <= 1e-9:
            continue

        out = row.to_dict()
        out["selected"] = 1
        out["stake"] = stake_value
        out["objective_ev_used"] = row[obj_col]
        out["expected_profit"] = stake_value * row["expected_value"]
        out["reason"] = (
            f"Greedy fallback selected highest available {settings.objective_mode}; "
            f"edge={row['edge']:.4f}; EV={row['expected_value']:.4f}; no group conflict."
        )
        selected.append(out)

        used_groups.add(row["market_group"])
        market_exposure[row["market_type"]] = market_exposure.get(row["market_type"], 0.0) + stake_value
        remaining_budget -= stake_value

    selected_df = pd.DataFrame(selected)
    return selected_df, _summary_from_selection(selected_df, data, settings, solver="Greedy fallback", warning=warning)


def optimize_portfolio(
    market_edges: pd.DataFrame,
    ranked_markets: pd.DataFrame | None = None,
    settings: OptimizerSettings | None = None,
) -> tuple[pd.DataFrame, dict]:
    settings = settings or OptimizerSettings()
    data = prepare_optimizer_data(market_edges, ranked_markets)

    try:
        model, x, stake = build_optimization_model(data, settings)
        solve_model(model)
        selected, summary = extract_solution(data, model, x, stake, settings)
    except Exception as exc:
        warning = f"Gurobi unavailable or failed ({exc}). Used greedy fallback optimizer instead."
        print(f"WARNING: {warning}")
        selected, summary = _fallback_greedy(data, settings, warning)

    if selected.empty:
        selected = pd.DataFrame(
            columns=[
                "game_id",
                "market_type",
                "selection",
                "line",
                "odds_decimal",
                "model_probability",
                "edge",
                "expected_value",
                "stake",
                "expected_profit",
                "reason",
            ]
        )

    return selected, summary


def save_optimization_outputs(selected: pd.DataFrame, summary: dict) -> None:
    selected.to_csv("gurobi_selected_portfolio.csv", index=False)
    with open("gurobi_optimization_summary.txt", "w", encoding="utf-8") as file:
        file.write("Gurobi / fallback portfolio optimization summary\n")
        file.write("Educational paper-trading only. Not betting advice.\n\n")
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    print("\nPortfolio optimization:")
    if summary.get("warning"):
        print(f"WARNING: {summary['warning']}")
    print(f"Solver: {summary['solver']}")
    print(f"Objective mode: {summary['objective_mode']}")
    print(f"Selected markets: {summary['num_selected']}")
    print(f"Total paper stake used: {summary['total_stake_used']:.2f}")
    print(f"Expected paper profit: {summary['expected_profit']:.2f}")
    print(f"Expected ROI on staked amount: {summary['expected_roi']:.2%}")
    print("Saved gurobi_selected_portfolio.csv")
    print("Saved gurobi_optimization_summary.txt")
