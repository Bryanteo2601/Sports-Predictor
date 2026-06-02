"""ROC/AUC backtest plots for the football and NBA model outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs")
FOOTBALL_PREDICTION_CANDIDATES = [
    Path("test_predictions.csv"),
    OUTPUT_DIR / "test_predictions.csv",
    Path("football_prediction") / "test_predictions.csv",
    Path("football_prediction") / "outputs" / "test_predictions.csv",
]
NBA_BACKTEST_CANDIDATES = [
    OUTPUT_DIR / "spurs_thunder_leave_one_game_backtest.csv",
    Path("nba_prediction") / "game_7_spurs_vs_thunder" / "outputs" / "spurs_thunder_leave_one_game_backtest.csv",
]


def first_existing_path(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    choices = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Expected one of:\n{choices}")


def roc_curve_auc(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return FPR, TPR, and AUC for a binary ranking score.

    The score is assumed to rank positives higher than negatives. Tied scores
    are grouped at the same threshold, which keeps the ROC curve stable.
    """

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    valid = ~np.isnan(y) & ~np.isnan(s)
    y = y[valid]
    s = s[valid]

    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.nan

    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and s[end] == s[start]:
            end += 1
        block = y[start:end]
        tp += int(block.sum())
        fp += int(len(block) - block.sum())
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        start = end

    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr.append(1.0)
        tpr.append(1.0)

    fpr_arr = np.asarray(fpr)
    tpr_arr = np.asarray(tpr)
    auc = float(np.trapz(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc


def football_roc() -> list[dict]:
    football = pd.read_csv(first_existing_path(FOOTBALL_PREDICTION_CANDIDATES, "football prediction CSV"))
    classes = [
        ("Home win", "H", "P_HomeWin"),
        ("Draw", "D", "P_Draw"),
        ("Away win", "A", "P_AwayWin"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    rows = []
    all_labels = []
    all_scores = []

    for label, outcome, score_col in classes:
        y_true = football["FTR"].eq(outcome).astype(int)
        scores = pd.to_numeric(football[score_col], errors="coerce")
        fpr, tpr, auc = roc_curve_auc(y_true, scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} AUC={auc:.3f}")
        rows.append(
            {
                "sport": "football",
                "target": label,
                "n": int(y_true.notna().sum()),
                "positives": int(y_true.sum()),
                "negatives": int(len(y_true) - y_true.sum()),
                "auc": auc,
                "auc_less_than_0_5": bool(auc < 0.5),
            }
        )
        all_labels.append(y_true.to_numpy())
        all_scores.append(scores.to_numpy())

    micro_labels = np.concatenate(all_labels)
    micro_scores = np.concatenate(all_scores)
    fpr, tpr, micro_auc = roc_curve_auc(micro_labels, micro_scores)
    ax.plot(fpr, tpr, linewidth=2.5, color="black", label=f"Micro avg AUC={micro_auc:.3f}")
    rows.append(
        {
            "sport": "football",
            "target": "Micro average",
            "n": int(len(micro_labels)),
            "positives": int(micro_labels.sum()),
            "negatives": int(len(micro_labels) - micro_labels.sum()),
            "auc": micro_auc,
            "auc_less_than_0_5": bool(micro_auc < 0.5),
        }
    )

    macro_auc = float(np.nanmean([row["auc"] for row in rows if row["target"] != "Micro average"]))
    rows.append(
        {
            "sport": "football",
            "target": "Macro average",
            "n": int(len(football)),
            "positives": np.nan,
            "negatives": np.nan,
            "auc": macro_auc,
            "auc_less_than_0_5": bool(macro_auc < 0.5),
        }
    )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#777777", label="Random AUC=0.500")
    ax.set_title("Football ROC Backtest")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "football_roc_auc.png")
    plt.close(fig)
    return rows


def nba_roc() -> list[dict]:
    nba = pd.read_csv(first_existing_path(NBA_BACKTEST_CANDIDATES, "NBA backtest CSV"))
    nba["home_win"] = nba["actual_winner"].eq(nba["home_team"]).astype(int)
    nba["away_win"] = nba["actual_winner"].eq(nba["away_team"]).astype(int)
    home_scores = pd.to_numeric(nba["predicted_home_margin"], errors="coerce")
    away_scores = -home_scores

    home_fpr, home_tpr, home_auc = roc_curve_auc(nba["home_win"], home_scores)
    away_fpr, away_tpr, away_auc = roc_curve_auc(nba["away_win"], away_scores)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(home_fpr, home_tpr, linewidth=2.5, label=f"Home win AUC={home_auc:.3f}")
    ax.plot(away_fpr, away_tpr, linewidth=2.5, label=f"Away win AUC={away_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#777777", label="Random AUC=0.500")
    ax.set_title("NBA ROC Backtest")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nba_roc_auc.png")
    plt.close(fig)

    positives = int(nba["home_win"].sum())
    return [
        {
            "sport": "NBA",
            "target": "Home win",
            "n": int(len(nba)),
            "positives": positives,
            "negatives": int(len(nba) - positives),
            "auc": home_auc,
            "auc_less_than_0_5": bool(home_auc < 0.5),
        },
        {
            "sport": "NBA",
            "target": "Away win",
            "n": int(len(nba)),
            "positives": int(nba["away_win"].sum()),
            "negatives": int(len(nba) - nba["away_win"].sum()),
            "auc": away_auc,
            "auc_less_than_0_5": bool(away_auc < 0.5),
        }
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = football_roc() + nba_roc()
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "roc_auc_summary.csv", index=False)

    display = summary.copy()
    display["auc"] = display["auc"].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    print(display.to_string(index=False))
    print()
    print(f"Saved summary: {OUTPUT_DIR / 'roc_auc_summary.csv'}")
    print(f"Saved football ROC plot: {OUTPUT_DIR / 'football_roc_auc.png'}")
    print(f"Saved NBA ROC plot: {OUTPUT_DIR / 'nba_roc_auc.png'}")


if __name__ == "__main__":
    main()
