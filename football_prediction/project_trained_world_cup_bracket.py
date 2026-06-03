"""
Print a probability-projected 2026 World Cup path from the trained model.

This is not one random tournament. It runs many tournaments, estimates the
most likely group order and knockout progression, then prints the bracket path.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from trained_international_model import (
    RANDOM_SEED,
    WORLD_CUP_2026_START,
    build_feature_dataset,
    latest_state_snapshot,
    load_international_results,
    print_current_availability_notes,
    rank_group_table,
    simulate_trained_match,
    train_models,
)
from world_cup_2026_simulator import (
    GROUPS,
    NEXT_ROUNDS,
    ROUND_OF_32_TEMPLATE,
    STADIUMS,
    GroupStanding,
    Team,
    assign_third_place_teams,
    create_teams,
    knockout_context,
)


STAGE_COLUMNS = [
    "Reach Round of 32",
    "Reach Round of 16",
    "Reach Quarterfinal",
    "Reach Semifinal",
    "Reach Final",
    "Finish runner-up",
    "Finish third",
    "Win World Cup",
]


def simulate_projection(home_model, away_model, states, n_simulations: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    team_templates = {team.name: team for team in create_teams()}
    counters = defaultdict(lambda: defaultdict(float))
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
            for idx, (team_1_name, team_2_name) in enumerate(
                [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
            ):
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
            for rank, team_name in enumerate(ranked, start=1):
                counters[team_name][f"Rank {rank}"] += 1
                counters[team_name]["Average group points"] += table[team_name]["points"]
                counters[team_name]["Average group goal difference"] += (
                    table[team_name]["goals_for"] - table[team_name]["goals_against"]
                )
                counters[team_name]["Average group rank"] += rank
            counters[ranked[0]]["Win group"] += 1
            counters[ranked[0]]["Top two group"] += 1
            counters[ranked[1]]["Top two group"] += 1
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
            counters[team.name]["Best third-place qualifier"] += 1
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

        for counter_name, fixtures in [
            ("Reach Quarterfinal", NEXT_ROUNDS["Round of 16"]),
            ("Reach Semifinal", NEXT_ROUNDS["Quarterfinal"]),
            ("Reach Final", NEXT_ROUNDS["Semifinal"]),
        ]:
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
        row = {"Team": team.name, "Group": team.group}
        for column in ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Win group", "Top two group", "Best third-place qualifier", *STAGE_COLUMNS]:
            row[column] = counters[team.name][column] / n_simulations
        row["Average group points"] = counters[team.name]["Average group points"] / n_simulations
        row["Average group goal difference"] = counters[team.name]["Average group goal difference"] / n_simulations
        row["Average group rank"] = counters[team.name]["Average group rank"] / n_simulations
        rows.append(row)
    return pd.DataFrame(rows)


def choose_group_order(group_rows: pd.DataFrame) -> list[str]:
    return (
        group_rows.sort_values(
            ["Average group rank", "Win group", "Reach Round of 32", "Average group points"],
            ascending=[True, False, False, False],
        )["Team"]
        .tolist()
    )


def conditional(summary_by_team: pd.DataFrame, team_name: str, numerator: str, denominator: str) -> float:
    denominator_value = float(summary_by_team.loc[team_name, denominator])
    if denominator_value <= 0:
        return 0.0
    return float(summary_by_team.loc[team_name, numerator]) / denominator_value


def choose_winner(summary_by_team: pd.DataFrame, team_1: Team, team_2: Team, numerator: str, denominator: str):
    team_1_chance = conditional(summary_by_team, team_1.name, numerator, denominator)
    team_2_chance = conditional(summary_by_team, team_2.name, numerator, denominator)
    if team_1_chance >= team_2_chance:
        return team_1, team_2, team_1_chance, team_2_chance
    return team_2, team_1, team_2_chance, team_1_chance


def print_projection(summary: pd.DataFrame, n_simulations: int) -> None:
    teams_by_name = {team.name: team for team in create_teams()}
    summary_by_team = summary.set_index("Team")
    qualified: dict[str, Team | list[GroupStanding]] = {}
    projected_groups: dict[str, list[str]] = {}
    projected_third_rows = []

    print("\n" + "=" * 104)
    print(f"TRAINED MODEL WORLD CUP PATH PROJECTION ({n_simulations:,} simulations)")
    print("=" * 104)
    print("Group rank is chosen by average simulated rank. Knockout winners use conditional next-round probability.")

    print("\nGROUP STAGE")
    for group in GROUPS:
        group_rows = summary[summary["Group"].eq(group)].copy()
        order = choose_group_order(group_rows)
        projected_groups[group] = order
        qualified[f"1{group}"] = teams_by_name[order[0]]
        qualified[f"2{group}"] = teams_by_name[order[1]]
        projected_third_rows.append(
            (
                float(summary_by_team.loc[order[2], "Best third-place qualifier"]),
                GroupStanding(team=teams_by_name[order[2]]),
            )
        )
        print(f"\nGroup {group}")
        print("Rank  Team                      P1      P2      P3      Qualify  AvgPts  AvgGD")
        for rank, team_name in enumerate(order, start=1):
            row = summary_by_team.loc[team_name]
            print(
                f"{rank:<5} {team_name:<25} "
                f"{row['Rank 1']:>6.1%} {row['Rank 2']:>6.1%} {row['Rank 3']:>6.1%} "
                f"{row['Reach Round of 32']:>8.1%} {row['Average group points']:>7.2f} {row['Average group goal difference']:>6.2f}"
            )

    projected_third_rows = sorted(projected_third_rows, key=lambda item: item[0], reverse=True)[:8]
    qualified["third_place_rows"] = [row for _, row in projected_third_rows]
    third_qualifiers = [row.team.name for _, row in projected_third_rows]

    print("\nPROJECTED THIRD-PLACE QUALIFIERS")
    for name in third_qualifiers:
        row = summary_by_team.loc[name]
        print(f"- {name:<25} Group {row['Group']} | best-third {row['Best third-place qualifier']:.1%}, qualify {row['Reach Round of 32']:.1%}")

    third_slots = [slot for _, _, slot in ROUND_OF_32_TEMPLATE if slot.startswith("3:")]
    third_assignments = assign_third_place_teams(qualified["third_place_rows"], third_slots)
    round_of_32 = []
    for match_id, slot_1, slot_2 in ROUND_OF_32_TEMPLATE:
        team_1 = qualified[slot_1] if not slot_1.startswith("3:") else third_assignments[slot_1]
        team_2 = qualified[slot_2] if not slot_2.startswith("3:") else third_assignments[slot_2]
        round_of_32.append((match_id, team_1, team_2))

    winners: dict[int, Team] = {}
    losers: dict[int, Team] = {}

    print("\nROUND OF 32")
    for match_id, team_1, team_2 in round_of_32:
        winner, loser, winner_chance, loser_chance = choose_winner(summary_by_team, team_1, team_2, "Reach Round of 16", "Reach Round of 32")
        winners[match_id] = winner
        losers[match_id] = loser
        print(f"M{match_id}: {team_1.name} vs {team_2.name} -> {winner.name} ({winner_chance:.1%} vs {loser_chance:.1%})")

    for title, numerator, denominator, fixtures in [
        ("ROUND OF 16", "Reach Quarterfinal", "Reach Round of 16", NEXT_ROUNDS["Round of 16"]),
        ("QUARTERFINALS", "Reach Semifinal", "Reach Quarterfinal", NEXT_ROUNDS["Quarterfinal"]),
        ("SEMIFINALS", "Reach Final", "Reach Semifinal", NEXT_ROUNDS["Semifinal"]),
    ]:
        print(f"\n{title}")
        for match_id, source_1, source_2 in fixtures:
            team_1 = winners[source_1]
            team_2 = winners[source_2]
            winner, loser, winner_chance, loser_chance = choose_winner(summary_by_team, team_1, team_2, numerator, denominator)
            winners[match_id] = winner
            losers[match_id] = loser
            print(f"M{match_id}: {team_1.name} vs {team_2.name} -> {winner.name} ({winner_chance:.1%} vs {loser_chance:.1%})")

    champion, runner_up, champion_chance, runner_up_chance = choose_winner(summary_by_team, winners[101], winners[102], "Win World Cup", "Reach Final")
    print("\nFINAL")
    print(f"M104: {winners[101].name} vs {winners[102].name} -> {champion.name} champion ({champion_chance:.1%} vs {runner_up_chance:.1%})")

    print("\nMOST LIKELY FINAL")
    print(f"{winners[101].name} vs {winners[102].name}")
    print(f"Projected winner: {champion.name}")


def main() -> None:
    n_simulations = int(os.getenv("TRAINED_WC_SIMULATIONS", "10000"))
    print("Loading international results and training final model...")
    results = load_international_results(refresh=os.getenv("REFRESH_INTERNATIONAL_RESULTS", "0") == "1")
    feature_data, _ = build_feature_dataset(results)
    train_mask = feature_data["date"].lt(WORLD_CUP_2026_START)
    home_model, away_model = train_models(feature_data, train_mask)
    _, states = latest_state_snapshot(results, WORLD_CUP_2026_START)
    print_current_availability_notes()
    print(f"Simulating {n_simulations:,} tournaments...")
    summary = simulate_projection(home_model, away_model, states, n_simulations)
    summary.to_csv("outputs/trained_world_cup_2026_path_projection.csv", index=False)
    print_projection(summary, n_simulations)
    print("\nSaved outputs/trained_world_cup_2026_path_projection.csv")


if __name__ == "__main__":
    main()
