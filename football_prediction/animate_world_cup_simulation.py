"""
Animate 10,000 World Cup 2026 Monte Carlo simulations at high speed.

The animation shows:
- a dark World Cup-style bracket frame
- the running simulation count
- the current leader/champion probability
- live champion probability bars
- live stage survival probabilities

Writes:
- outputs/world_cup_2026_10000_simulation_animation.gif
- outputs/world_cup_2026_10000_simulation_animation.png
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

from world_cup_2026_simulator import RANDOM_SEED, create_teams, simulate_tournament


OUTPUT_DIR = Path("outputs")
GIF_PATH = OUTPUT_DIR / "world_cup_2026_10000_simulation_animation.gif"
FINAL_FRAME_PATH = OUTPUT_DIR / "world_cup_2026_10000_simulation_animation.png"
SNAPSHOT_PATH = OUTPUT_DIR / "world_cup_2026_10000_animation_snapshots.csv"

N_SIMULATIONS = 10_000
SNAPSHOT_EVERY = 100
TOP_TEAMS_TO_SHOW = 10
FPS = 18

BACKGROUND = "#111213"
PANEL = "#1b1d1f"
GOLD = "#c5a75c"
GOLD_DARK = "#806b35"
SILVER = "#d7d9dc"
LINE = "#55585d"
TEXT = "#f1f4f7"
MUTED = "#9ba3ad"
BLUE = "#6ea8fe"


def collect_snapshots(n_simulations: int = N_SIMULATIONS, snapshot_every: int = SNAPSHOT_EVERY) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    teams = [team.name for team in create_teams()]
    counters = defaultdict(lambda: defaultdict(int))
    snapshots = []

    for simulation_index in range(1, n_simulations + 1):
        tournament = simulate_tournament(seed=int(rng.integers(0, 1_000_000_000)))

        qualified_names = set()
        for key, team in tournament["qualified"].items():  # type: ignore[union-attr]
            if key.startswith(("1", "2")):
                qualified_names.add(team.name)  # type: ignore[union-attr]
        for row in tournament["qualified"]["third_place_rows"]:  # type: ignore[index]
            qualified_names.add(row.team.name)

        for team_name in qualified_names:
            counters[team_name]["Round of 32"] += 1

        for result in tournament["knockout_results"]:  # type: ignore[union-attr]
            if result.stage == "Round of 32":
                counters[result.winner]["Round of 16"] += 1
            elif result.stage == "Round of 16":
                counters[result.winner]["Quarterfinal"] += 1
            elif result.stage == "Quarterfinal":
                counters[result.winner]["Semifinal"] += 1
            elif result.stage == "Semifinal":
                counters[result.winner]["Final"] += 1
            elif result.stage == "Final":
                counters[result.winner]["Champion"] += 1

        if simulation_index % snapshot_every == 0 or simulation_index == 1:
            champion_probs = {
                team: counters[team]["Champion"] / simulation_index
                for team in teams
            }
            leader = max(champion_probs, key=champion_probs.get)
            ordered = sorted(champion_probs.items(), key=lambda item: item[1], reverse=True)
            row = {
                "simulation": simulation_index,
                "leader": leader,
                "leader_probability": champion_probs[leader],
            }
            for rank, (team, probability) in enumerate(ordered[:TOP_TEAMS_TO_SHOW], start=1):
                row[f"rank_{rank}_team"] = team
                row[f"rank_{rank}_champion_probability"] = probability

            for stage in ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]:
                row[f"{stage}_leader_probability"] = counters[leader][stage] / simulation_index

            snapshots.append(row)

    snapshot_df = pd.DataFrame(snapshots)
    snapshot_df.to_csv(SNAPSHOT_PATH, index=False)
    return snapshot_df


def draw_static_bracket(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(BACKGROUND)

    y_positions = np.linspace(86, 14, 16)
    left_round_x = [5, 17, 29, 41]
    right_round_x = [95, 83, 71, 59]

    for side, xs in [("left", left_round_x), ("right", right_round_x)]:
        for idx, y in enumerate(y_positions):
            x = xs[0]
            ax.add_patch(Rectangle((x - 4.0, y - 1.8), 8, 3.6, facecolor=GOLD, edgecolor=GOLD_DARK, linewidth=1.0))

        current_ys = y_positions
        for round_index in range(1, 4):
            next_ys = []
            for pair_index in range(0, len(current_ys), 2):
                y_1 = current_ys[pair_index]
                y_2 = current_ys[pair_index + 1]
                y_mid = (y_1 + y_2) / 2
                x_prev = xs[round_index - 1]
                x_next = xs[round_index]
                connector_x = (x_prev + x_next) / 2
                ax.plot([x_prev + (4 if side == "left" else -4), connector_x], [y_1, y_1], color=LINE, linewidth=1.2)
                ax.plot([x_prev + (4 if side == "left" else -4), connector_x], [y_2, y_2], color=LINE, linewidth=1.2)
                ax.plot([connector_x, connector_x], [y_1, y_2], color=LINE, linewidth=1.2)
                ax.plot([connector_x, x_next - (4 if side == "left" else -4)], [y_mid, y_mid], color=LINE, linewidth=1.2)
                ax.add_patch(Rectangle((x_next - 4.0, y_mid - 1.7), 8, 3.4, facecolor=SILVER, edgecolor="#909297", linewidth=1.0))
                next_ys.append(y_mid)
            current_ys = np.array(next_ys)

    ax.plot([45, 50], [50, 50], color=LINE, linewidth=1.3)
    ax.plot([55, 50], [50, 50], color=LINE, linewidth=1.3)
    ax.add_patch(Circle((50, 50), 6.8, facecolor=GOLD, edgecolor="#e6d08a", linewidth=1.4, alpha=0.95))
    ax.text(50, 50.9, "FIFA", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    ax.text(50, 47.8, "2026", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    ax.text(50, 92, "10,000 WORLD CUP SIMULATIONS", ha="center", va="center", color=TEXT, fontsize=18, fontweight="bold")
    ax.text(50, 8, "Round of 32       Round of 16       Quarterfinals       Semifinals       Final       Semifinals       Quarterfinals       Round of 16       Round of 32", ha="center", va="center", color=MUTED, fontsize=7)


def update_frame(frame_index: int, snapshots: pd.DataFrame, ax) -> None:
    ax.clear()
    draw_static_bracket(ax)

    row = snapshots.iloc[frame_index]
    simulation_count = int(row["simulation"])
    leader = row["leader"]
    leader_probability = row["leader_probability"]

    ax.add_patch(Rectangle((30, 61.5), 40, 22.5, facecolor=PANEL, edgecolor="#30343a", linewidth=1.3, alpha=0.96))
    ax.text(50, 80, f"{simulation_count:,} / {N_SIMULATIONS:,}", ha="center", va="center", color=TEXT, fontsize=22, fontweight="bold")
    ax.text(50, 75.8, "tournaments simulated", ha="center", va="center", color=MUTED, fontsize=9)
    ax.text(50, 70.7, f"Current leader: {leader}", ha="center", va="center", color=GOLD, fontsize=16, fontweight="bold")
    ax.text(50, 66.5, f"{leader_probability:.2%} champion probability", ha="center", va="center", color=TEXT, fontsize=12)

    stages = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]
    stage_labels = ["R32", "R16", "QF", "SF", "F", "WIN"]
    stage_x = np.linspace(34, 66, len(stages))
    for x, stage, label in zip(stage_x, stages, stage_labels):
        p = row[f"{stage}_leader_probability"]
        ax.add_patch(Circle((x, 29), 2.2, facecolor=BLUE if stage != "Champion" else GOLD, edgecolor="white", linewidth=0.6, alpha=0.9))
        ax.text(x, 29.1, label, ha="center", va="center", color="#081018", fontsize=7, fontweight="bold")
        ax.text(x, 24.8, f"{p:.0%}", ha="center", va="center", color=TEXT, fontsize=8, fontweight="bold")
        if x != stage_x[-1]:
            ax.plot([x + 2.5, x + 4.8], [29, 29], color=LINE, linewidth=1.2)

    bar_left = 31
    bar_bottom = 36
    bar_width = 38
    bar_height = 1.75
    gap = 0.8
    ax.text(50, 58.5, "Live champion probability leaderboard", ha="center", va="center", color=TEXT, fontsize=10, fontweight="bold")

    max_probability = max(row[f"rank_{rank}_champion_probability"] for rank in range(1, TOP_TEAMS_TO_SHOW + 1))
    max_probability = max(max_probability, 0.001)

    for rank in range(1, TOP_TEAMS_TO_SHOW + 1):
        team = row[f"rank_{rank}_team"]
        probability = row[f"rank_{rank}_champion_probability"]
        y = bar_bottom + (TOP_TEAMS_TO_SHOW - rank) * (bar_height + gap)
        fill_width = bar_width * probability / max_probability
        color = GOLD if rank == 1 else "#c9d4df"
        ax.text(bar_left - 1.2, y + bar_height / 2, f"{rank}", ha="right", va="center", color=MUTED, fontsize=7)
        ax.text(bar_left, y + bar_height / 2, team, ha="left", va="center", color=TEXT, fontsize=7.6)
        ax.add_patch(Rectangle((bar_left + 13, y), bar_width, bar_height, facecolor="#282c31", edgecolor="#33383f", linewidth=0.4))
        ax.add_patch(Rectangle((bar_left + 13, y), fill_width, bar_height, facecolor=color, edgecolor=color, linewidth=0.3))
        ax.text(bar_left + 13 + bar_width + 1.0, y + bar_height / 2, f"{probability:.2%}", ha="left", va="center", color=TEXT, fontsize=7.6)

    progress_width = 64 * simulation_count / N_SIMULATIONS
    ax.add_patch(Rectangle((18, 12), 64, 1.4, facecolor="#2a2d31", edgecolor="#42464d", linewidth=0.5))
    ax.add_patch(Rectangle((18, 12), progress_width, 1.4, facecolor=GOLD, edgecolor=GOLD, linewidth=0.5))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    snapshots = collect_snapshots()

    fig, ax = plt.subplots(figsize=(14, 8), dpi=140)
    fig.patch.set_facecolor(BACKGROUND)

    def animate(frame_index: int):
        update_frame(frame_index, snapshots, ax)
        return []

    anim = animation.FuncAnimation(fig, animate, frames=len(snapshots), interval=1000 / FPS, blit=False)
    anim.save(GIF_PATH, writer=animation.PillowWriter(fps=FPS))
    update_frame(len(snapshots) - 1, snapshots, ax)
    fig.savefig(FINAL_FRAME_PATH, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)

    print(f"Saved {GIF_PATH}")
    print(f"Saved {FINAL_FRAME_PATH}")
    print(f"Saved {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
