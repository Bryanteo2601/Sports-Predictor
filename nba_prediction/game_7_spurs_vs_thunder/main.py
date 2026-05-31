"""Run the Spurs vs Thunder Game 7 simulation and market-edge analysis."""

import pandas as pd

from backtest import leave_one_game_out_backtest, validation_warning
from config import BASE_PARAMS, OUTPUT_DIR, RANDOM_SEED
from data_loader import load_inputs
from feature_engineering import team_summary
from gurobi_optimizer import optimize_portfolio, save_optimization_outputs
from market import evaluate_markets
from optimizer import summarize_market_stability
from report import print_summary, save_charts
from simulator import referee_adjustments, run_parameter_scenarios


def main() -> None:
    game_logs, recent_form, injuries, odds = load_inputs()
    backtest_results, backtest_summary = leave_one_game_out_backtest(game_logs, injuries)

    calibrated_params = BASE_PARAMS.copy()
    if not backtest_summary.empty and backtest_summary.loc[0, "games_tested"] > 0:
        calibrated_params["total_sd"] = max(8.0, float(backtest_summary.loc[0, "total_rmse"]))
        calibrated_params["margin_sd"] = max(8.0, float(backtest_summary.loc[0, "margin_rmse"]))

    features = team_summary(game_logs, injuries, calibrated_params)

    all_sims, run_summaries = run_parameter_scenarios(features, seed=RANDOM_SEED, base_params=calibrated_params)
    market_edges = evaluate_markets(all_sims, odds)
    ranked_markets = summarize_market_stability(all_sims, odds)
    selected_portfolio, optimization_summary = optimize_portfolio(market_edges, ranked_markets)

    all_sims.to_csv("simulation_results.csv", index=False)
    market_edges.to_csv("market_edge_results.csv", index=False)
    ranked_markets.to_csv("ranked_markets.csv", index=False)
    save_optimization_outputs(selected_portfolio, optimization_summary)
    run_summaries.to_csv(OUTPUT_DIR / "spurs_thunder_game7_model_runs.csv", index=False)
    features.to_csv(OUTPUT_DIR / "spurs_thunder_game7_team_features.csv", index=False)
    pd.DataFrame([referee_adjustments()]).to_csv(
        OUTPUT_DIR / "spurs_thunder_game7_referee_assumptions.csv", index=False
    )

    save_charts(market_edges, run_summaries)
    print_summary(market_edges, run_summaries, validation_warning(game_logs))

    print("\nChronological backtest on available historical games:")
    print(backtest_summary.to_string(index=False))

    print("\nSaved CSV files:")
    print("team_game_logs.csv")
    print("recent_form_features.csv")
    print("injuries_manual.csv")
    print("odds_game7.csv")
    print("simulation_results.csv")
    print("market_edge_results.csv")
    print("ranked_markets.csv")
    print("gurobi_selected_portfolio.csv")
    print("gurobi_optimization_summary.txt")


if __name__ == "__main__":
    main()
