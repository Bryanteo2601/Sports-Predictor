"""
Plot the latest simulated World Cup 2026 knockout tree.

Reads:
- outputs/world_cup_2026_knockout_results.csv

Writes:
- outputs/world_cup_2026_knockout_tree.png
- outputs/world_cup_2026_knockout_tree.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")
KNOCKOUT_RESULTS_PATH = OUTPUT_DIR / "world_cup_2026_knockout_results.csv"
PNG_PATH = OUTPUT_DIR / "world_cup_2026_knockout_tree.png"
PDF_PATH = OUTPUT_DIR / "world_cup_2026_knockout_tree.pdf"

ROUND_OF_32 = [73, 75, 74, 77, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87]
BRACKET_SOURCES = {
    89: (73, 75),
    90: (74, 77),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
    101: (97, 98),
    102: (99, 100),
    104: (101, 102),
}

ROUND_X = {
    "Round of 32": 0,
    "Round of 16": 1,
    "Quarterfinal": 2,
    "Semifinal": 3,
    "Final": 4,
}

ROUND_LABELS = {
    0: "Round of 32",
    1: "Round of 16",
    2: "Quarterfinals",
    3: "Semifinals",
    4: "Final",
}


def score_label(row: pd.Series) -> str:
    suffix = ""
    if row["decided_by"] == "penalties":
        suffix = " (pens)"
    elif row["decided_by"] == "extra_time":
        suffix = " (aet)"

    return (
        f"M{int(row['match_id'])}\n"
        f"{row['team_1']} {int(row['goals_1'])}-{int(row['goals_2'])} {row['team_2']}{suffix}\n"
        f"Winner: {row['winner']}"
    )


def build_positions(results: pd.DataFrame) -> dict[int, tuple[int, float]]:
    positions = {}
    ordered_r32 = [match_id for match_id in ROUND_OF_32 if match_id in set(results["match_id"])]

    for index, match_id in enumerate(ordered_r32):
        positions[match_id] = (0, len(ordered_r32) - 1 - index)

    for match_id, sources in BRACKET_SOURCES.items():
        if match_id not in set(results["match_id"]):
            continue
        source_ys = [positions[source][1] for source in sources]
        stage = results.loc[results["match_id"].eq(match_id), "stage"].iloc[0]
        positions[match_id] = (ROUND_X[stage], sum(source_ys) / len(source_ys))

    return positions


def draw_match_box(ax, x: float, y: float, label: str, stage: str, is_final: bool) -> None:
    if is_final:
        facecolor = "#12355b"
        edgecolor = "#0b1f35"
        text_color = "white"
        width = 0.82
    elif stage == "Semifinal":
        facecolor = "#e9f2ff"
        edgecolor = "#245a92"
        text_color = "#102033"
        width = 0.80
    elif stage == "Quarterfinal":
        facecolor = "#f3f7fb"
        edgecolor = "#6f8faf"
        text_color = "#102033"
        width = 0.78
    else:
        facecolor = "white"
        edgecolor = "#a9b7c6"
        text_color = "#102033"
        width = 0.76

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=7.4 if not is_final else 8.4,
        color=text_color,
        linespacing=1.2,
        bbox={
            "boxstyle": "round,pad=0.42,rounding_size=0.08",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": 1.2 if not is_final else 1.8,
        },
        zorder=3,
    )


def plot_bracket() -> None:
    if not KNOCKOUT_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {KNOCKOUT_RESULTS_PATH}. Run world_cup_2026_simulator.py first.")

    results = pd.read_csv(KNOCKOUT_RESULTS_PATH)
    match_rows = results.set_index("match_id", drop=False)
    positions = build_positions(results)

    fig, ax = plt.subplots(figsize=(18, 12), dpi=160)
    fig.patch.set_facecolor("#f7f9fb")
    ax.set_facecolor("#f7f9fb")

    for match_id, sources in BRACKET_SOURCES.items():
        if match_id not in positions:
            continue
        target_x, target_y = positions[match_id]
        for source in sources:
            if source not in positions:
                continue
            source_x, source_y = positions[source]
            ax.plot(
                [source_x + 0.32, target_x - 0.32],
                [source_y, target_y],
                color="#9aa8b6",
                linewidth=1.3,
                zorder=1,
            )

    for match_id, (x, y) in positions.items():
        row = match_rows.loc[match_id]
        draw_match_box(
            ax=ax,
            x=x,
            y=y,
            label=score_label(row),
            stage=row["stage"],
            is_final=match_id == 104,
        )

    final_row = match_rows.loc[104]
    champion = final_row["winner"]
    ax.text(
        4,
        positions[104][1] + 1.35,
        f"Champion: {champion}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#12355b",
    )

    if 103 in match_rows.index:
        third = match_rows.loc[103]
        third_label = (
            f"Third-place match\n"
            f"{third['team_1']} {int(third['goals_1'])}-{int(third['goals_2'])} {third['team_2']}\n"
            f"Third: {third['winner']}"
        )
        ax.text(
            4,
            -1.25,
            third_label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#102033",
            bbox={
                "boxstyle": "round,pad=0.48,rounding_size=0.08",
                "facecolor": "#fff8e8",
                "edgecolor": "#d19a2a",
                "linewidth": 1.3,
            },
        )

    for x, label in ROUND_LABELS.items():
        ax.text(x, 16.3, label, ha="center", va="bottom", fontsize=11, fontweight="bold", color="#30465c")

    ax.text(
        0,
        17.25,
        "FIFA World Cup 2026 Simulated Knockout Tree",
        ha="left",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color="#102033",
    )
    ax.text(
        0,
        16.85,
        "Latest single simulated tournament. Monte Carlo winner probabilities are in outputs/world_cup_2026_monte_carlo_summary.csv.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#52677c",
    )

    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-2.0, 17.75)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_bracket()
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")
