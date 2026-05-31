"""Create or fetch data files for the NBA Game 7 project."""

import os
import pandas as pd

from config import DATA_DIR, ODDS_ROWS, PREVIOUS_SIX_GAMES, SPURS, THUNDER


NBA_TEAM_NAMES = {
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "LA Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
}


GAME_LOG_COLUMNS = [
    "game_id",
    "date",
    "team",
    "opponent",
    "home_away",
    "points_for",
    "points_against",
    "win_loss",
    "offensive_rating",
    "defensive_rating",
    "net_rating",
    "pace",
    "efg_pct",
    "turnover_pct",
    "offensive_rebound_pct",
    "free_throw_rate",
    "three_point_rate",
    "rest_days",
]


def build_seed_game_logs() -> pd.DataFrame:
    """Create team-game rows from the previous six games."""

    rows = []
    for game in PREVIOUS_SIX_GAMES:
        home = game["home"]
        away = game["away"]
        home_points = game["home_points"]
        away_points = game["away_points"]

        for team, opponent, home_away, points_for, points_against in [
            (home, away, "home", home_points, away_points),
            (away, home, "away", away_points, home_points),
        ]:
            possessions = 100.0
            offensive_rating = points_for / possessions * 100
            defensive_rating = points_against / possessions * 100
            rows.append(
                {
                    "date": game["date"],
                    "game_id": game["date"] + "_" + home.replace(" ", "_") + "_" + away.replace(" ", "_"),
                    "team": team,
                    "opponent": opponent,
                    "home_away": home_away,
                    "points_for": points_for,
                    "points_against": points_against,
                    "win_loss": "W" if points_for > points_against else "L",
                    "offensive_rating": offensive_rating,
                    "defensive_rating": defensive_rating,
                    "net_rating": offensive_rating - defensive_rating,
                    "pace": possessions,
                    "efg_pct": pd.NA,
                    "turnover_pct": pd.NA,
                    "offensive_rebound_pct": pd.NA,
                    "free_throw_rate": pd.NA,
                    "three_point_rate": pd.NA,
                    "rest_days": 2,
                }
            )

    return pd.DataFrame(rows, columns=GAME_LOG_COLUMNS)


def try_fetch_nba_api_game_logs() -> pd.DataFrame | None:
    """Try to fetch NBA logs with nba_api.

    The environment may not have internet access or nba_api installed. If this
    fails, the project falls back to the six-game seed dataset.
    """

    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except Exception:
        return None

    try:
        finder = leaguegamefinder.LeagueGameFinder(season_nullable="2025-26")
        raw_all = finder.get_data_frames()[0]
        raw = raw_all[raw_all["TEAM_NAME"].isin(NBA_TEAM_NAMES)].copy()
        if raw.empty:
            return None

        opponent_points = (
            raw_all[["GAME_ID", "TEAM_NAME", "PTS"]]
            .rename(columns={"TEAM_NAME": "opponent_team_name", "PTS": "points_against"})
        )
        raw = raw.merge(opponent_points, on="GAME_ID", how="left")
        raw = raw[raw["TEAM_NAME"].ne(raw["opponent_team_name"])].copy()
        raw = raw[raw["opponent_team_name"].isin(NBA_TEAM_NAMES)].copy()

        raw["date"] = pd.to_datetime(raw["GAME_DATE"]).dt.date.astype(str)
        raw["game_id"] = raw["GAME_ID"]
        raw["team"] = raw["TEAM_NAME"]
        raw["opponent"] = raw["opponent_team_name"]
        raw["home_away"] = raw["MATCHUP"].apply(lambda x: "home" if "vs." in x else "away")
        raw["points_for"] = raw["PTS"]
        raw["win_loss"] = raw["WL"]
        raw["offensive_rating"] = raw["points_for"]
        raw["defensive_rating"] = raw["points_against"]
        raw["net_rating"] = raw["offensive_rating"] - raw["defensive_rating"]
        raw["pace"] = pd.NA
        raw["efg_pct"] = (raw["FGM"] + 0.5 * raw["FG3M"]) / raw["FGA"].replace(0, pd.NA)
        raw["turnover_pct"] = raw["TOV"] / (raw["FGA"] + 0.44 * raw["FTA"] + raw["TOV"]).replace(0, pd.NA)
        raw["offensive_rebound_pct"] = pd.NA
        raw["free_throw_rate"] = raw["FTA"] / raw["FGA"].replace(0, pd.NA)
        raw["three_point_rate"] = raw["FG3A"] / raw["FGA"].replace(0, pd.NA)
        raw = raw.sort_values(["team", "date"])
        raw["rest_days"] = raw.groupby("team")["date"].transform(
            lambda s: pd.to_datetime(s).diff().dt.days
        )
        return raw[GAME_LOG_COLUMNS].drop_duplicates().reset_index(drop=True)
    except Exception:
        return None

    return None


def create_injury_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": SPURS, "player": "Manual input", "status": "available", "estimated_impact_points": 0.0, "notes": ""},
            {"team": THUNDER, "player": "Manual input", "status": "available", "estimated_impact_points": 0.0, "notes": ""},
        ]
    )


def create_recent_form_features(game_logs: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["points_for", "points_against", "net_rating", "pace", "rest_days"]
    rows = []
    for team, group in game_logs.groupby("team"):
        row = {"team": team, "games": len(group)}
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"avg_{col}"] = values.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_csv_files() -> None:
    """Create all input CSV files if they do not already exist."""

    game_logs_path = DATA_DIR / "team_game_logs.csv"
    refresh_data = os.getenv("REFRESH_NBA_DATA", "0") == "1"
    if refresh_data or not game_logs_path.exists():
        fetched = try_fetch_nba_api_game_logs()
        game_logs = fetched if fetched is not None and not fetched.empty else build_seed_game_logs()
        game_logs.to_csv(game_logs_path, index=False)

    game_logs = pd.read_csv(game_logs_path)
    recent_form_path = DATA_DIR / "recent_form_features.csv"
    if refresh_data or not recent_form_path.exists():
        create_recent_form_features(game_logs).to_csv(recent_form_path, index=False)

    injuries_path = DATA_DIR / "injuries_manual.csv"
    if not injuries_path.exists():
        create_injury_template().to_csv(injuries_path, index=False)

    odds_path = DATA_DIR / "odds_game7.csv"
    if not odds_path.exists():
        pd.DataFrame(ODDS_ROWS).to_csv(odds_path, index=False)

    print("Data files ready:")
    print("team_game_logs.csv")
    print("recent_form_features.csv")
    print("injuries_manual.csv")
    print("odds_game7.csv")


if __name__ == "__main__":
    ensure_csv_files()
