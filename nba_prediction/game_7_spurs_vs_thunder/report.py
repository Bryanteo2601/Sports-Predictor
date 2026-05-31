"""Reporting and chart helpers."""

import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_DIR


def save_charts(market_edges: pd.DataFrame, run_summaries: pd.DataFrame) -> None:
    top_10 = market_edges.head(10).copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = top_10["market_type"] + ": " + top_10["selection"]
    ax.barh(labels[::-1], top_10["expected_value_per_1"][::-1])
    ax.set_title("Top 10 Theoretical Market Edges")
    ax.set_xlabel("Expected value per $1")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spurs_thunder_game7_market_edges.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(run_summaries["mean_margin_spurs_minus_thunder"], bins=20)
    ax.set_title("Parameter-Scenario Margin Distribution")
    ax.set_xlabel("Spurs margin minus Thunder")
    ax.set_ylabel("Model runs")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spurs_thunder_game7_parameter_margins.png", dpi=150)
    plt.close()


def print_summary(market_edges: pd.DataFrame, run_summaries: pd.DataFrame, warning: str) -> None:
    spurs_win = run_summaries["spurs_win_probability"].mean()
    thunder_win = run_summaries["thunder_win_probability"].mean()
    total_mean = run_summaries["mean_total"].mean()
    margin_mean = run_summaries["mean_margin_spurs_minus_thunder"].mean()

    print("\nEducational warning:")
    print("This is paper-trading analytics only, not betting advice. Real-money betting is risky and not recommended.")
    print(warning)

    print("\nProjected probabilities:")
    print(f"Spurs win probability: {spurs_win:.2%}")
    print(f"Thunder win probability: {thunder_win:.2%}")
    print(f"Predicted total points mean: {total_mean:.2f}")
    print(
        "Predicted total points range across parameter runs: "
        f"{run_summaries['mean_total'].min():.2f} to {run_summaries['mean_total'].max():.2f}"
    )
    print(f"Predicted Spurs margin mean: {margin_mean:.2f}")
    print(
        "Predicted Spurs margin range across parameter runs: "
        f"{run_summaries['mean_margin_spurs_minus_thunder'].min():.2f} "
        f"to {run_summaries['mean_margin_spurs_minus_thunder'].max():.2f}"
    )

    print("\nTop 10 theoretical edges:")
    cols = [
        "market_type",
        "selection",
        "line",
        "odds_decimal",
        "no_vig_probability",
        "model_probability",
        "edge",
        "expected_value_per_1",
        "edge_label",
    ]
    print(market_edges[cols].head(10).to_string(index=False))

    best = market_edges.iloc[0]
    print("\nStrongest paper-trade candidate by model EV:")
    print(
        f"{best['market_type']} | {best['selection']} | line={best['line']} | "
        f"EV={best['expected_value_per_1']:.3f} | edge={best['edge']:.3f}"
    )
