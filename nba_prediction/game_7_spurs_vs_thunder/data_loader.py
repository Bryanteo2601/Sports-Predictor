"""Load CSV inputs for the NBA Game 7 project."""

import pandas as pd

from config import DATA_DIR
from data_fetcher import ensure_csv_files


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_csv_files()
    game_logs = pd.read_csv(DATA_DIR / "team_game_logs.csv")
    recent_form = pd.read_csv(DATA_DIR / "recent_form_features.csv")
    injuries = pd.read_csv(DATA_DIR / "injuries_manual.csv")
    odds = pd.read_csv(DATA_DIR / "odds_game7.csv")
    return game_logs, recent_form, injuries, odds
