"""
FIFA World Cup 2026 Monte Carlo simulator.

This keeps the expected-goals / Poisson style from the Champions League model,
but shifts the inputs to World Cup-specific context:
- regional qualifier performance
- most recent confederation tournament or Nations League performance
- travel across USA, Mexico, and Canada
- rest days
- host nation advantage
- squad quality, elite league representation, big-club players, breakout players
- injuries and player availability
- explainable match-level predictions

The team context values are normalized scouting/model inputs on a 0-100 scale.
They are intentionally kept in one table so they can be replaced later with
FIFA ranking, Elo, Transfermarkt, FBref, FotMob, SofaScore, ClubElo, injury,
or odds-implied data without changing the tournament engine.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
BASE_CAMPS_PATH = DATA_DIR / "world_cup_2026_team_base_camps.csv"
MATCH_SCHEDULE_PATH = DATA_DIR / "world_cup_2026_match_schedule.csv"

RANDOM_SEED = 42
BASE_XG = 1.35
EXTRA_TIME_XG_SHARE = 0.32
DEFAULT_GROUP_REST_DAYS = 5
DEFAULT_KNOCKOUT_REST_DAYS = 4

HOST_COUNTRIES = {"United States", "Mexico", "Canada"}
HOST_TEAM_HOME_COUNTRY = {
    "United States": "United States",
    "Mexico": "Mexico",
    "Canada": "Canada",
}


@dataclass
class Player:
    name: str
    country: str
    club: str
    league: str
    position: str
    minutes_played: int = 0
    goals: int = 0
    assists: int = 0
    clean_sheets: int = 0
    expected_goals: float | None = None
    expected_assists: float | None = None
    player_rating: float | None = None
    club_strength_rating: float = 70.0
    league_strength_rating: float = 70.0
    form_rating: float = 70.0
    is_key_player: bool = False
    is_available: bool = True
    injury_status: str = "available"
    expected_minutes: int = 90
    fitness_level: float = 1.0


@dataclass
class Team:
    name: str
    group: str
    confederation: str
    home_country: str
    base_city: str
    base_country: str
    base_latitude: float
    base_longitude: float
    attack_rating: float = 75.0
    defence_rating: float = 75.0
    overall_rating: float = 75.0
    penalty_rating: float = 75.0
    fair_play_score: float = 0.0
    qualifier_score: float = 75.0
    confed_competition_score: float = 75.0
    elite_league_player_count: int = 0
    big_club_player_count: int = 0
    breakout_player_score: float = 50.0
    squad_depth_score: float = 75.0
    current_form_score: float = 75.0
    injury_impact: float = 0.0
    market_odds: float | None = None
    market_implied_probability: float = 0.0
    market_strength_score: float = 75.0
    manual_note: str = ""
    squad_players: list[Player] = field(default_factory=list)
    last_stadium: "Stadium | None" = None
    last_match_date: date | None = None
    cumulative_travel_distance: float = 0.0


@dataclass(frozen=True)
class Stadium:
    name: str
    city: str
    country: str
    latitude: float
    longitude: float


@dataclass
class MatchContext:
    stadium: Stadium
    match_date: date


@dataclass
class TeamMatchFactors:
    distance_from_base_km: float
    travel_from_previous_km: float
    estimated_travel_time_hours: float
    crossed_country_border: bool
    rest_days: int
    cumulative_travel_distance: float
    travel_fatigue_score: float
    home_advantage_multiplier: float
    player_quality_score: float
    adjusted_attack: float
    adjusted_defence: float
    xg: float = 0.0


@dataclass
class MatchResult:
    match_id: int
    stage: str
    team_1: str
    team_2: str
    goals_1: int
    goals_2: int
    winner: str | None
    loser: str | None
    decided_by: str
    stadium: str
    city: str
    country: str
    team_1_xg: float
    team_2_xg: float
    explanation: str
    team_1_factors: TeamMatchFactors
    team_2_factors: TeamMatchFactors


@dataclass
class GroupStanding:
    team: Team
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


STADIUMS = {
    "Mexico City": Stadium("Estadio Azteca", "Mexico City", "Mexico", 19.3029, -99.1505),
    "Guadalajara": Stadium("Estadio Akron", "Guadalajara", "Mexico", 20.6818, -103.4627),
    "Monterrey": Stadium("Estadio BBVA", "Monterrey", "Mexico", 25.6689, -100.2446),
    "Toronto": Stadium("BMO Field", "Toronto", "Canada", 43.6332, -79.4186),
    "Vancouver": Stadium("BC Place", "Vancouver", "Canada", 49.2768, -123.1119),
    "Atlanta": Stadium("Mercedes-Benz Stadium", "Atlanta", "United States", 33.7555, -84.4008),
    "Boston": Stadium("Gillette Stadium", "Boston", "United States", 42.0909, -71.2643),
    "Dallas": Stadium("AT&T Stadium", "Dallas", "United States", 32.7473, -97.0945),
    "Houston": Stadium("NRG Stadium", "Houston", "United States", 29.6847, -95.4107),
    "Kansas City": Stadium("Arrowhead Stadium", "Kansas City", "United States", 39.0489, -94.4839),
    "Los Angeles": Stadium("SoFi Stadium", "Los Angeles", "United States", 33.9535, -118.3392),
    "Miami": Stadium("Hard Rock Stadium", "Miami", "United States", 25.9580, -80.2389),
    "New York/New Jersey": Stadium("MetLife Stadium", "New York/New Jersey", "United States", 40.8135, -74.0745),
    "Philadelphia": Stadium("Lincoln Financial Field", "Philadelphia", "United States", 39.9008, -75.1675),
    "San Francisco Bay Area": Stadium("Levi's Stadium", "San Francisco Bay Area", "United States", 37.4030, -121.9700),
    "Seattle": Stadium("Lumen Field", "Seattle", "United States", 47.5952, -122.3316),
}


GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


TEAM_CONTEXT = {
    # Scores blend most recent qualifier strength and most recent regional
    # competition context: UEFA Nations League 2024/25, Copa America 2024,
    # CONCACAF Gold Cup/Nations League 2025, Asian Cup 2023, AFCON 2025,
    # and OFC Nations Cup 2024 where relevant.
    "Argentina": ("CONMEBOL", 96, 92, 95, 94, 96, 14, 10, 86, 93, 94, 0.02),
    "Brazil": ("CONMEBOL", 92, 91, 92, 87, 85, 18, 13, 82, 91, 88, 0.04),
    "Uruguay": ("CONMEBOL", 88, 88, 88, 89, 87, 11, 7, 84, 87, 88, 0.03),
    "Colombia": ("CONMEBOL", 86, 85, 86, 87, 91, 8, 4, 89, 84, 90, 0.02),
    "Ecuador": ("CONMEBOL", 82, 84, 83, 84, 80, 7, 3, 80, 81, 83, 0.02),
    "Paraguay": ("CONMEBOL", 76, 79, 77, 77, 68, 4, 1, 70, 75, 74, 0.03),
    "England": ("UEFA", 92, 90, 92, 89, 87, 19, 13, 83, 92, 88, 0.05),
    "France": ("UEFA", 93, 91, 93, 88, 86, 20, 14, 86, 93, 88, 0.04),
    "Spain": ("UEFA", 92, 90, 92, 91, 95, 18, 12, 85, 91, 94, 0.03),
    "Portugal": ("UEFA", 91, 88, 90, 92, 96, 15, 10, 84, 89, 94, 0.03),
    "Germany": ("UEFA", 88, 86, 88, 87, 84, 16, 10, 80, 88, 85, 0.04),
    "Netherlands": ("UEFA", 88, 86, 88, 86, 84, 15, 9, 82, 87, 84, 0.04),
    "Belgium": ("UEFA", 86, 84, 86, 83, 78, 13, 7, 79, 84, 80, 0.05),
    "Croatia": ("UEFA", 84, 84, 84, 85, 83, 9, 5, 78, 82, 82, 0.04),
    "Switzerland": ("UEFA", 81, 83, 82, 81, 76, 10, 4, 76, 81, 78, 0.03),
    "Austria": ("UEFA", 82, 80, 81, 84, 82, 8, 3, 78, 80, 84, 0.02),
    "Turkey": ("UEFA", 82, 78, 81, 83, 81, 9, 4, 82, 79, 84, 0.04),
    "Czechia": ("UEFA", 79, 78, 79, 80, 75, 7, 2, 76, 78, 78, 0.03),
    "Sweden": ("UEFA", 80, 78, 79, 79, 73, 8, 3, 74, 78, 76, 0.05),
    "Scotland": ("UEFA", 76, 77, 76, 76, 70, 7, 1, 72, 76, 73, 0.04),
    "Norway": ("UEFA", 84, 76, 81, 82, 74, 6, 3, 82, 75, 80, 0.04),
    "Bosnia and Herzegovina": ("UEFA", 73, 72, 73, 70, 62, 3, 0, 67, 70, 68, 0.05),
    "Morocco": ("CAF", 85, 85, 85, 87, 86, 10, 5, 84, 84, 88, 0.02),
    "Senegal": ("CAF", 83, 84, 83, 84, 82, 9, 4, 80, 83, 84, 0.03),
    "Ivory Coast": ("CAF", 82, 80, 82, 81, 90, 8, 3, 82, 80, 87, 0.03),
    "Egypt": ("CAF", 82, 78, 81, 83, 78, 6, 2, 78, 78, 81, 0.04),
    "Algeria": ("CAF", 80, 78, 80, 80, 74, 7, 2, 78, 78, 77, 0.04),
    "Ghana": ("CAF", 77, 76, 77, 76, 68, 7, 2, 75, 76, 72, 0.05),
    "Tunisia": ("CAF", 76, 78, 77, 77, 70, 5, 1, 70, 76, 73, 0.04),
    "South Africa": ("CAF", 75, 76, 75, 77, 80, 2, 0, 76, 74, 80, 0.02),
    "Cape Verde": ("CAF", 73, 74, 74, 78, 78, 3, 0, 82, 73, 81, 0.02),
    "DR Congo": ("CAF", 75, 76, 75, 77, 79, 4, 1, 77, 74, 79, 0.03),
    "Japan": ("AFC", 84, 82, 84, 91, 84, 12, 5, 86, 83, 90, 0.02),
    "South Korea": ("AFC", 83, 80, 82, 86, 80, 9, 4, 78, 81, 84, 0.04),
    "Australia": ("AFC", 78, 79, 79, 82, 76, 5, 1, 74, 79, 78, 0.03),
    "Iran": ("AFC", 81, 79, 80, 84, 78, 4, 1, 76, 79, 81, 0.03),
    "Saudi Arabia": ("AFC", 76, 76, 76, 76, 70, 3, 0, 73, 76, 73, 0.05),
    "Qatar": ("AFC", 75, 76, 76, 78, 87, 1, 0, 74, 75, 84, 0.02),
    "Uzbekistan": ("AFC", 75, 75, 75, 82, 78, 2, 0, 80, 74, 82, 0.02),
    "Iraq": ("AFC", 74, 74, 74, 81, 75, 1, 0, 77, 73, 79, 0.03),
    "Jordan": ("AFC", 73, 74, 73, 80, 82, 1, 0, 79, 72, 83, 0.02),
    "United States": ("CONCACAF", 82, 79, 81, 83, 74, 10, 5, 79, 80, 77, 0.05),
    "Mexico": ("CONCACAF", 81, 80, 81, 84, 83, 8, 3, 77, 80, 84, 0.03),
    "Canada": ("CONCACAF", 80, 77, 79, 82, 79, 8, 3, 78, 78, 81, 0.04),
    "Panama": ("CONCACAF", 75, 75, 75, 82, 81, 2, 0, 76, 74, 82, 0.02),
    "Haiti": ("CONCACAF", 72, 71, 72, 78, 70, 1, 0, 74, 71, 77, 0.03),
    "Curacao": ("CONCACAF", 72, 71, 72, 78, 72, 2, 0, 76, 71, 78, 0.02),
    "New Zealand": ("OFC", 74, 73, 74, 88, 86, 2, 0, 73, 73, 86, 0.02),
}


WINNER_MARKET_DECIMAL_ODDS = {
    "Spain": 5.50,
    "France": 6.00,
    "England": 7.00,
    "Argentina": 9.00,
    "Brazil": 9.00,
    "Portugal": 11.00,
    "Germany": 15.00,
    "Netherlands": 21.00,
    "Norway": 26.00,
    "Colombia": 34.00,
    "Belgium": 34.00,
    "Japan": 51.00,
    "Morocco": 51.00,
    "Uruguay": 67.00,
    "United States": 67.00,
    "Mexico": 81.00,
    "Croatia": 81.00,
    "Switzerland": 81.00,
    "Turkey": 81.00,
    "Ecuador": 101.00,
    "Senegal": 126.00,
    "Sweden": 126.00,
    "Austria": 151.00,
    "Paraguay": 151.00,
    "Canada": 151.00,
    "Scotland": 251.00,
    "Bosnia and Herzegovina": 251.00,
    "Egypt": 301.00,
    "Ivory Coast": 301.00,
    "Czechia": 301.00,
    "South Korea": 401.00,
    "Ghana": 401.00,
    "Algeria": 401.00,
    "Australia": 501.00,
    "Tunisia": 501.00,
    "Iran": 501.00,
    "DR Congo": 751.00,
    "Saudi Arabia": 1001.00,
    "South Africa": 1001.00,
    "Panama": 1501.00,
    "New Zealand": 1501.00,
    "Uzbekistan": 1501.00,
    "Iraq": 1501.00,
    "Qatar": 2001.00,
    "Cape Verde": 2001.00,
    "Jordan": 2501.00,
    "Haiti": 3001.00,
    "Curacao": 4001.00,
}


MANUAL_TEAM_NOTES = {
    # User opinion only. These notes are exported for transparency and are not
    # used in the model unless converted into an explicit injury/availability
    # or rating input.
    "England": "User note: personal view says England will not win; not model-active.",
    "Spain": "User note: concern about Lamine Yamal and Fermin availability; not model-active without confirmed availability input.",
}


BASE_CAMPS = {
    "A": ("Monterrey", "Mexico", 25.6866, -100.3161),
    "B": ("Toronto", "Canada", 43.6532, -79.3832),
    "C": ("Miami", "United States", 25.7617, -80.1918),
    "D": ("Dallas", "United States", 32.7767, -96.7970),
    "E": ("Atlanta", "United States", 33.7490, -84.3880),
    "F": ("Seattle", "United States", 47.6062, -122.3321),
    "G": ("New York/New Jersey", "United States", 40.7128, -74.0060),
    "H": ("Los Angeles", "United States", 34.0522, -118.2437),
    "I": ("Boston", "United States", 42.3601, -71.0589),
    "J": ("Houston", "United States", 29.7604, -95.3698),
    "K": ("Kansas City", "United States", 39.0997, -94.5786),
    "L": ("Philadelphia", "United States", 39.9526, -75.1652),
}


GROUP_STADIUM_ROTATION = {
    "A": ["Mexico City", "Guadalajara", "Monterrey"],
    "B": ["Toronto", "Vancouver", "Seattle"],
    "C": ["Miami", "Houston", "Dallas"],
    "D": ["Los Angeles", "San Francisco Bay Area", "Seattle"],
    "E": ["Atlanta", "Philadelphia", "New York/New Jersey"],
    "F": ["Vancouver", "Seattle", "Los Angeles"],
    "G": ["New York/New Jersey", "Boston", "Toronto"],
    "H": ["Dallas", "Kansas City", "Houston"],
    "I": ["Boston", "New York/New Jersey", "Philadelphia"],
    "J": ["Mexico City", "Houston", "Dallas"],
    "K": ["Kansas City", "Atlanta", "Miami"],
    "L": ["Philadelphia", "Toronto", "Boston"],
}


KNOCKOUT_STADIUM_ROTATION = [
    "Los Angeles",
    "Boston",
    "Dallas",
    "Houston",
    "Mexico City",
    "New York/New Jersey",
    "Atlanta",
    "Vancouver",
    "Philadelphia",
    "Seattle",
    "Toronto",
    "Miami",
    "Kansas City",
    "San Francisco Bay Area",
    "Monterrey",
    "Guadalajara",
]


def load_wikipedia_base_camps() -> dict[str, tuple[str, str, float, float]]:
    if not BASE_CAMPS_PATH.exists():
        return {}
    camps = pd.read_csv(BASE_CAMPS_PATH)
    return {
        str(row["team"]): (
            str(row["base_city"]),
            str(row["base_country"]),
            float(row["base_latitude"]),
            float(row["base_longitude"]),
        )
        for _, row in camps.dropna(subset=["base_latitude", "base_longitude"]).iterrows()
    }


def load_wikipedia_match_schedule() -> pd.DataFrame:
    if not MATCH_SCHEDULE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MATCH_SCHEDULE_PATH)


def stadium_from_schedule_row(row: pd.Series) -> Stadium:
    return Stadium(
        name=str(row["stadium"]),
        city=str(row["venue_city"]),
        country=str(row["venue_country"]),
        latitude=float(row["venue_latitude"]),
        longitude=float(row["venue_longitude"]),
    )


def match_context_from_schedule(match_id: int) -> MatchContext | None:
    schedule = load_wikipedia_match_schedule()
    if schedule.empty:
        return None
    row = schedule[schedule["match_id"].eq(match_id)]
    if row.empty:
        return None
    match = row.iloc[0]
    return MatchContext(stadium=stadium_from_schedule_row(match), match_date=pd.to_datetime(match["date"]).date())


def group_stage_fixtures(group: str) -> list[tuple[int, str, str, MatchContext]]:
    schedule = load_wikipedia_match_schedule()
    if schedule.empty:
        return []
    group_teams = set(GROUPS[group])
    fixtures = []
    for _, row in schedule[schedule["match_id"].le(72)].sort_values("match_id").iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        if home_team in group_teams and away_team in group_teams:
            fixtures.append(
                (
                    int(row["match_id"]),
                    home_team,
                    away_team,
                    MatchContext(stadium=stadium_from_schedule_row(row), match_date=pd.to_datetime(row["date"]).date()),
                )
            )
    return fixtures


ROUND_OF_32_TEMPLATE = [
    (73, "2A", "2B"),
    (74, "1E", "3:A/B/C/D/F"),
    (75, "1F", "2C"),
    (76, "1C", "2F"),
    (77, "1I", "3:C/D/F/G/H"),
    (78, "2E", "2I"),
    (79, "1A", "3:C/E/F/H/I"),
    (80, "1L", "3:E/H/I/J/K"),
    (81, "1D", "3:B/E/F/I/J"),
    (82, "1G", "3:A/E/H/I/J"),
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (85, "1B", "3:E/F/G/I/J"),
    (86, "1J", "2H"),
    (87, "1K", "3:D/E/I/J/L"),
    (88, "2D", "2G"),
]


NEXT_ROUNDS = {
    "Round of 16": [(89, 73, 75), (90, 74, 77), (91, 76, 78), (92, 79, 80), (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)],
    "Quarterfinal": [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)],
    "Semifinal": [(101, 97, 98), (102, 99, 100)],
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def no_vig_market_probabilities() -> dict[str, float]:
    raw = {team: 1 / odds for team, odds in WINNER_MARKET_DECIMAL_ODDS.items()}
    overround = sum(raw.values())
    return {team: probability / overround for team, probability in raw.items()}


def market_strength_scores() -> dict[str, float]:
    probabilities = no_vig_market_probabilities()
    if not probabilities:
        return {}

    min_probability = min(probabilities.values())
    max_probability = max(probabilities.values())
    if math.isclose(min_probability, max_probability):
        return {team: 75.0 for team in probabilities}

    return {
        team: 50 + 50 * (probability - min_probability) / (max_probability - min_probability)
        for team, probability in probabilities.items()
    }


def haversine_km(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    radius_km = 6371.0
    phi_1 = math.radians(lat_1)
    phi_2 = math.radians(lat_2)
    delta_phi = math.radians(lat_2 - lat_1)
    delta_lambda = math.radians(lon_2 - lon_1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_travel_distance(team: Team, stadium: Stadium) -> tuple[float, float]:
    distance_km = haversine_km(team.base_latitude, team.base_longitude, stadium.latitude, stadium.longitude)
    estimated_travel_time_hours = distance_km / 720 + 1.5
    return distance_km, estimated_travel_time_hours


def calculate_rest_days(team: Team, match_date: date, fallback: int) -> int:
    if team.last_match_date is None:
        return fallback
    return max(0, (match_date - team.last_match_date).days)


def calculate_travel_fatigue(team: Team, context: MatchContext, fallback_rest_days: int) -> TeamMatchFactors:
    base_distance, travel_hours = calculate_travel_distance(team, context.stadium)
    previous_distance = 0.0
    previous_country = team.base_country

    if team.last_stadium is not None:
        previous_distance = haversine_km(
            team.last_stadium.latitude,
            team.last_stadium.longitude,
            context.stadium.latitude,
            context.stadium.longitude,
        )
        previous_country = team.last_stadium.country

    rest_days = calculate_rest_days(team, context.match_date, fallback_rest_days)
    crossed_country_border = previous_country != context.stadium.country
    projected_cumulative = team.cumulative_travel_distance + previous_distance

    fatigue = (
        0.0005 * base_distance
        + 0.0007 * previous_distance
        + 0.00003 * projected_cumulative
        - 0.15 * rest_days
        + (0.30 if crossed_country_border else 0.0)
    )

    return TeamMatchFactors(
        distance_from_base_km=base_distance,
        travel_from_previous_km=previous_distance,
        estimated_travel_time_hours=travel_hours,
        crossed_country_border=crossed_country_border,
        rest_days=rest_days,
        cumulative_travel_distance=projected_cumulative,
        travel_fatigue_score=clamp(fatigue, 0.0, 1.0),
        home_advantage_multiplier=1.0,
        player_quality_score=0.0,
        adjusted_attack=0.0,
        adjusted_defence=0.0,
    )


def calculate_home_advantage(team: Team, stadium: Stadium) -> float:
    if team.name in HOST_TEAM_HOME_COUNTRY:
        if stadium.country == HOST_TEAM_HOME_COUNTRY[team.name]:
            return 1.05
        if stadium.country in HOST_COUNTRIES:
            return 1.02
    return 1.0


def calculate_player_form(player: Player) -> float:
    output = player.goals + 0.7 * player.assists + 0.2 * player.clean_sheets
    minutes_factor = clamp(player.minutes_played / 1800, 0.0, 1.0)
    base_rating = player.player_rating if player.player_rating is not None else player.form_rating
    return clamp(0.55 * base_rating + 15 * minutes_factor + 2.0 * output, 0.0, 100.0)


def calculate_breakout_player_score(team: Team) -> float:
    if not team.squad_players:
        return team.breakout_player_score
    breakout_bonus = 0.0
    for player in team.squad_players:
        form = calculate_player_form(player)
        if player.club_strength_rating < 75 and form > 85 and player.minutes_played > 900:
            breakout_bonus += 6.0
    return clamp(team.breakout_player_score + breakout_bonus, 0.0, 100.0)


def calculate_squad_depth(team: Team) -> float:
    if not team.squad_players:
        return team.squad_depth_score

    available = [p for p in team.squad_players if p.is_available]
    if not available:
        return team.squad_depth_score * 0.75

    qualities = sorted(
        [
            0.45 * (p.player_rating or p.form_rating)
            + 0.30 * p.club_strength_rating
            + 0.25 * p.league_strength_rating
            for p in available
        ],
        reverse=True,
    )
    top_11 = np.mean(qualities[:11]) if qualities[:11] else team.squad_depth_score
    next_7 = np.mean(qualities[11:18]) if qualities[11:18] else top_11 * 0.92
    return clamp(0.6 * top_11 + 0.3 * next_7 + 0.1 * team.squad_depth_score, 0.0, 100.0)


def calculate_player_quality_score(team: Team) -> float:
    elite_score = clamp(team.elite_league_player_count / 18 * 100, 0.0, 100.0)
    big_club_score = clamp(team.big_club_player_count / 12 * 100, 0.0, 100.0)
    breakout_score = calculate_breakout_player_score(team)
    squad_depth = calculate_squad_depth(team)
    return clamp(
        0.30 * elite_score
        + 0.25 * big_club_score
        + 0.20 * team.current_form_score
        + 0.15 * squad_depth
        + 0.10 * breakout_score,
        0.0,
        100.0,
    )


def calculate_adjusted_team_strength(team: Team, context: MatchContext, fallback_rest_days: int) -> TeamMatchFactors:
    factors = calculate_travel_fatigue(team, context, fallback_rest_days)
    player_quality_score = calculate_player_quality_score(team)
    home_multiplier = calculate_home_advantage(team, context.stadium)
    player_quality_multiplier = 0.90 + player_quality_score / 500
    qualifier_multiplier = 0.92 + team.qualifier_score / 625
    confed_multiplier = 0.94 + team.confed_competition_score / 850
    market_multiplier = 0.94 + team.market_strength_score / 900
    form_multiplier = 0.93 + team.current_form_score / 700
    depth_multiplier = 0.94 + calculate_squad_depth(team) / 900
    fatigue_attack_penalty = 1 - 0.04 * factors.travel_fatigue_score
    fatigue_defence_penalty = 1 - 0.03 * factors.travel_fatigue_score
    injury_attack_penalty = 1 - 0.06 * team.injury_impact
    injury_defence_penalty = 1 - 0.05 * team.injury_impact

    factors.home_advantage_multiplier = home_multiplier
    factors.player_quality_score = player_quality_score
    factors.adjusted_attack = (
        team.attack_rating
        * player_quality_multiplier
        * qualifier_multiplier
        * confed_multiplier
        * market_multiplier
        * form_multiplier
        * home_multiplier
        * fatigue_attack_penalty
        * injury_attack_penalty
    )
    factors.adjusted_defence = (
        team.defence_rating
        * player_quality_multiplier
        * qualifier_multiplier
        * confed_multiplier
        * market_multiplier
        * depth_multiplier
        * home_multiplier
        * fatigue_defence_penalty
        * injury_defence_penalty
    )
    return factors


def calculate_expected_goals(team_1: Team, team_2: Team, context: MatchContext, fallback_rest_days: int) -> tuple[TeamMatchFactors, TeamMatchFactors]:
    factors_1 = calculate_adjusted_team_strength(team_1, context, fallback_rest_days)
    factors_2 = calculate_adjusted_team_strength(team_2, context, fallback_rest_days)
    factors_1.xg = clamp(BASE_XG * (factors_1.adjusted_attack / factors_2.adjusted_defence), 0.15, 4.20)
    factors_2.xg = clamp(BASE_XG * (factors_2.adjusted_attack / factors_1.adjusted_defence), 0.15, 4.20)
    return factors_1, factors_2


def explain_match_prediction(team_1: Team, team_2: Team, context: MatchContext, factors_1: TeamMatchFactors, factors_2: TeamMatchFactors) -> str:
    notes = []
    quality_gap = factors_1.player_quality_score - factors_2.player_quality_score
    if abs(quality_gap) >= 5:
        better = team_1 if quality_gap > 0 else team_2
        notes.append(
            f"{better.name} has the stronger player-quality profile from elite-league, big-club, depth, and breakout inputs."
        )
    else:
        notes.append("The player-quality profiles are close, so travel, form, and finishing variance matter more.")

    fatigue_gap = factors_1.travel_fatigue_score - factors_2.travel_fatigue_score
    if abs(fatigue_gap) >= 0.15:
        more_tired = team_1 if fatigue_gap > 0 else team_2
        notes.append(f"{more_tired.name} carries the bigger travel/rest fatigue penalty for this venue.")

    for team, factors in [(team_1, factors_1), (team_2, factors_2)]:
        if factors.home_advantage_multiplier > 1.03:
            notes.append(f"{team.name} receives host-nation advantage because the match is in {context.stadium.country}.")
        elif factors.home_advantage_multiplier > 1.0:
            notes.append(f"{team.name} receives a smaller host-region crowd boost in {context.stadium.country}.")

    if team_1.injury_impact > 0.04 or team_2.injury_impact > 0.04:
        notes.append("Availability penalties are active for at least one side, so injury updates can materially move the forecast.")

    notes.append(f"Model xG: {team_1.name} {factors_1.xg:.2f}, {team_2.name} {factors_2.xg:.2f}.")
    return " ".join(notes)


def create_teams() -> list[Team]:
    teams = []
    market_probabilities = no_vig_market_probabilities()
    market_scores = market_strength_scores()
    wikipedia_base_camps = load_wikipedia_base_camps()
    for group, team_names in GROUPS.items():
        for name in team_names:
            base_city, base_country, base_latitude, base_longitude = wikipedia_base_camps.get(name, BASE_CAMPS[group])
            (
                confederation,
                attack,
                defence,
                overall,
                qualifier_score,
                confed_score,
                elite_count,
                big_club_count,
                breakout_score,
                depth_score,
                form_score,
                injury_impact,
            ) = TEAM_CONTEXT[name]
            teams.append(
                Team(
                    name=name,
                    group=group,
                    confederation=confederation,
                    home_country=HOST_TEAM_HOME_COUNTRY.get(name, ""),
                    base_city=base_city,
                    base_country=base_country,
                    base_latitude=base_latitude,
                    base_longitude=base_longitude,
                    attack_rating=attack,
                    defence_rating=defence,
                    overall_rating=overall,
                    penalty_rating=overall,
                    qualifier_score=qualifier_score,
                    confed_competition_score=confed_score,
                    elite_league_player_count=elite_count,
                    big_club_player_count=big_club_count,
                    breakout_player_score=breakout_score,
                    squad_depth_score=depth_score,
                    current_form_score=form_score,
                    injury_impact=injury_impact,
                    market_odds=WINNER_MARKET_DECIMAL_ODDS.get(name),
                    market_implied_probability=market_probabilities.get(name, 0.0),
                    market_strength_score=market_scores.get(name, 75.0),
                    manual_note=MANUAL_TEAM_NOTES.get(name, ""),
                )
            )
    return teams


def clone_teams() -> dict[str, Team]:
    return {team.name: Team(**{field.name: getattr(team, field.name) for field in Team.__dataclass_fields__.values() if field.name != "squad_players"}, squad_players=list(team.squad_players)) for team in create_teams()}


def create_groups(teams_by_name: dict[str, Team]) -> dict[str, list[Team]]:
    return {group: [teams_by_name[name] for name in names] for group, names in GROUPS.items()}


def update_team_travel_state(team: Team, context: MatchContext) -> None:
    if team.last_stadium is not None:
        team.cumulative_travel_distance += haversine_km(
            team.last_stadium.latitude,
            team.last_stadium.longitude,
            context.stadium.latitude,
            context.stadium.longitude,
        )
    team.last_stadium = context.stadium
    team.last_match_date = context.match_date


def simulate_match(
    team_1: Team,
    team_2: Team,
    match_id: int,
    stage: str,
    context: MatchContext,
    rng: np.random.Generator,
    allow_draws: bool,
    fallback_rest_days: int,
    update_travel: bool = True,
) -> MatchResult:
    factors_1, factors_2 = calculate_expected_goals(team_1, team_2, context, fallback_rest_days)
    goals_1 = int(rng.poisson(factors_1.xg))
    goals_2 = int(rng.poisson(factors_2.xg))
    decided_by = "normal_time"

    if allow_draws and goals_1 == goals_2:
        winner = None
        loser = None
        decided_by = "draw"
    elif goals_1 == goals_2:
        et_goals_1 = int(rng.poisson(factors_1.xg * EXTRA_TIME_XG_SHARE))
        et_goals_2 = int(rng.poisson(factors_2.xg * EXTRA_TIME_XG_SHARE))
        goals_1 += et_goals_1
        goals_2 += et_goals_2
        if goals_1 != goals_2:
            decided_by = "extra_time"
            winner = team_1.name if goals_1 > goals_2 else team_2.name
            loser = team_2.name if goals_1 > goals_2 else team_1.name
        else:
            decided_by = "penalties"
            shootout_strength_1 = team_1.penalty_rating * (1 - 0.05 * team_1.injury_impact)
            shootout_strength_2 = team_2.penalty_rating * (1 - 0.05 * team_2.injury_impact)
            p_team_1 = shootout_strength_1 / (shootout_strength_1 + shootout_strength_2)
            winner = team_1.name if rng.random() < p_team_1 else team_2.name
            loser = team_2.name if winner == team_1.name else team_1.name
    else:
        winner = team_1.name if goals_1 > goals_2 else team_2.name
        loser = team_2.name if goals_1 > goals_2 else team_1.name

    explanation = explain_match_prediction(team_1, team_2, context, factors_1, factors_2)

    if update_travel:
        update_team_travel_state(team_1, context)
        update_team_travel_state(team_2, context)

    return MatchResult(
        match_id=match_id,
        stage=stage,
        team_1=team_1.name,
        team_2=team_2.name,
        goals_1=goals_1,
        goals_2=goals_2,
        winner=winner,
        loser=loser,
        decided_by=decided_by,
        stadium=context.stadium.name,
        city=context.stadium.city,
        country=context.stadium.country,
        team_1_xg=factors_1.xg,
        team_2_xg=factors_2.xg,
        explanation=explanation,
        team_1_factors=factors_1,
        team_2_factors=factors_2,
    )


def update_standings(standings: dict[str, GroupStanding], result: MatchResult) -> None:
    row_1 = standings[result.team_1]
    row_2 = standings[result.team_2]
    row_1.played += 1
    row_2.played += 1
    row_1.goals_for += result.goals_1
    row_1.goals_against += result.goals_2
    row_2.goals_for += result.goals_2
    row_2.goals_against += result.goals_1

    if result.goals_1 > result.goals_2:
        row_1.wins += 1
        row_2.losses += 1
        row_1.points += 3
    elif result.goals_1 < result.goals_2:
        row_2.wins += 1
        row_1.losses += 1
        row_2.points += 3
    else:
        row_1.draws += 1
        row_2.draws += 1
        row_1.points += 1
        row_2.points += 1


def rank_group(standings: dict[str, GroupStanding]) -> list[GroupStanding]:
    return sorted(
        standings.values(),
        key=lambda row: (
            -row.points,
            -row.goal_difference,
            -row.goals_for,
            row.team.fair_play_score,
            row.team.name,
        ),
    )


def rank_third_place_teams(group_standings: dict[str, list[GroupStanding]]) -> list[GroupStanding]:
    third_place_rows = [rows[2] for rows in group_standings.values()]
    return sorted(
        third_place_rows,
        key=lambda row: (
            -row.points,
            -row.goal_difference,
            -row.goals_for,
            row.team.fair_play_score,
            row.team.name,
        ),
    )[:8]


def group_match_context(group: str, match_number: int) -> MatchContext:
    city = GROUP_STADIUM_ROTATION[group][match_number % len(GROUP_STADIUM_ROTATION[group])]
    match_date = date(2026, 6, 11) + timedelta(days=match_number * 3 + (ord(group) - ord("A")) % 4)
    return MatchContext(stadium=STADIUMS[city], match_date=match_date)


def simulate_group_stage(groups: dict[str, list[Team]], rng: np.random.Generator) -> tuple[list[MatchResult], dict[str, list[GroupStanding]], dict[str, Team | list[Team]]]:
    group_results = []
    group_standings = {}
    qualified: dict[str, Team | list[Team]] = {}
    match_id = 1

    for group, teams in groups.items():
        standings = {team.name: GroupStanding(team=team) for team in teams}
        official_fixtures = group_stage_fixtures(group)
        if official_fixtures:
            fixture_rows = [
                (
                    scheduled_match_id,
                    next(team for team in teams if team.name == team_1_name),
                    next(team for team in teams if team.name == team_2_name),
                    context,
                )
                for scheduled_match_id, team_1_name, team_2_name, context in official_fixtures
            ]
        else:
            fixture_rows = [
                (match_id + match_number, team_1, team_2, group_match_context(group, match_number))
                for match_number, (team_1, team_2) in enumerate(itertools.combinations(teams, 2))
            ]

        for scheduled_match_id, team_1, team_2, context in fixture_rows:
            result = simulate_match(
                team_1,
                team_2,
                match_id=scheduled_match_id,
                stage=f"Group {group}",
                context=context,
                rng=rng,
                allow_draws=True,
                fallback_rest_days=DEFAULT_GROUP_REST_DAYS,
            )
            update_standings(standings, result)
            group_results.append(result)
            match_id = max(match_id, scheduled_match_id + 1)

        ranked = rank_group(standings)
        group_standings[group] = ranked
        qualified[f"1{group}"] = ranked[0].team
        qualified[f"2{group}"] = ranked[1].team

    third_place_rows = rank_third_place_teams(group_standings)
    qualified["third_place_rows"] = third_place_rows
    return group_results, group_standings, qualified


def assign_third_place_teams(qualified_third_place_teams: list[GroupStanding], third_place_slots: Iterable[str]) -> dict[str, Team]:
    # TODO: Replace greedy third-place allocation with official FIFA third-place
    # allocation matrix if exact bracket assignment is required.
    available = qualified_third_place_teams.copy()
    assignments = {}
    for slot in third_place_slots:
        allowed_groups = slot.replace("3:", "").split("/")
        for row in available:
            if row.team.group in allowed_groups:
                assignments[slot] = row.team
                available.remove(row)
                break
        if slot not in assignments:
            assignments[slot] = available.pop(0).team
    return assignments


def build_round_of_32(qualified: dict[str, Team | list[GroupStanding]]) -> list[tuple[int, Team, Team]]:
    third_slots = [away for _, home, away in ROUND_OF_32_TEMPLATE if away.startswith("3:")]
    third_assignments = assign_third_place_teams(qualified["third_place_rows"], third_slots)  # type: ignore[arg-type]
    fixtures = []
    for match_id, home_slot, away_slot in ROUND_OF_32_TEMPLATE:
        team_1 = qualified[home_slot] if not home_slot.startswith("3:") else third_assignments[home_slot]
        team_2 = qualified[away_slot] if not away_slot.startswith("3:") else third_assignments[away_slot]
        fixtures.append((match_id, team_1, team_2))  # type: ignore[arg-type]
    return fixtures


def knockout_context(index: int) -> MatchContext:
    scheduled_context = match_context_from_schedule(73 + index)
    if scheduled_context is not None:
        return scheduled_context
    city = KNOCKOUT_STADIUM_ROTATION[index % len(KNOCKOUT_STADIUM_ROTATION)]
    return MatchContext(stadium=STADIUMS[city], match_date=date(2026, 6, 28) + timedelta(days=index * 2))


def simulate_knockout_stage(round_of_32_fixtures: list[tuple[int, Team, Team]], teams_by_name: dict[str, Team], rng: np.random.Generator) -> tuple[list[MatchResult], Team, Team, Team]:
    results = []
    winners: dict[int, Team] = {}
    losers: dict[int, Team] = {}
    context_index = 0

    for match_id, team_1, team_2 in round_of_32_fixtures:
        result = simulate_match(team_1, team_2, match_id, "Round of 32", knockout_context(context_index), rng, False, DEFAULT_KNOCKOUT_REST_DAYS)
        results.append(result)
        winners[match_id] = teams_by_name[result.winner]  # type: ignore[index]
        losers[match_id] = teams_by_name[result.loser]  # type: ignore[index]
        context_index += 1

    for stage, fixtures in NEXT_ROUNDS.items():
        for match_id, source_1, source_2 in fixtures:
            result = simulate_match(winners[source_1], winners[source_2], match_id, stage, knockout_context(context_index), rng, False, DEFAULT_KNOCKOUT_REST_DAYS)
            results.append(result)
            winners[match_id] = teams_by_name[result.winner]  # type: ignore[index]
            losers[match_id] = teams_by_name[result.loser]  # type: ignore[index]
            context_index += 1

    third_place_result = simulate_match(losers[101], losers[102], 103, "Third-place match", knockout_context(context_index), rng, False, DEFAULT_KNOCKOUT_REST_DAYS)
    results.append(third_place_result)
    context_index += 1

    final_result = simulate_match(winners[101], winners[102], 104, "Final", knockout_context(context_index), rng, False, DEFAULT_KNOCKOUT_REST_DAYS)
    results.append(final_result)

    champion = teams_by_name[final_result.winner]  # type: ignore[index]
    runner_up = teams_by_name[final_result.loser]  # type: ignore[index]
    third_place = teams_by_name[third_place_result.winner]  # type: ignore[index]
    return results, champion, runner_up, third_place


def simulate_tournament(seed: int | None = None) -> dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED if seed is None else seed)
    teams_by_name = clone_teams()
    groups = create_groups(teams_by_name)
    group_results, group_standings, qualified = simulate_group_stage(groups, rng)
    round_of_32 = build_round_of_32(qualified)
    knockout_results, champion, runner_up, third_place = simulate_knockout_stage(round_of_32, teams_by_name, rng)
    return {
        "teams_by_name": teams_by_name,
        "group_results": group_results,
        "group_standings": group_standings,
        "qualified": qualified,
        "round_of_32": round_of_32,
        "knockout_results": knockout_results,
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place,
    }


def mark_progress(progress: dict[str, set[str]], team_name: str, stage: str) -> None:
    progress[team_name].add(stage)


def run_monte_carlo(n_simulations: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counters = defaultdict(lambda: defaultdict(int))
    goals_for = defaultdict(int)
    goals_against = defaultdict(int)

    for _ in range(n_simulations):
        tournament = simulate_tournament(seed=int(rng.integers(0, 1_000_000_000)))
        progress = defaultdict(set)

        for row in tournament["qualified"]["third_place_rows"]:  # type: ignore[index]
            mark_progress(progress, row.team.name, "Round of 32")
        for key, team in tournament["qualified"].items():  # type: ignore[union-attr]
            if key.startswith(("1", "2")):
                mark_progress(progress, team.name, "Round of 32")  # type: ignore[union-attr]

        for result in tournament["knockout_results"]:  # type: ignore[union-attr]
            if result.stage == "Round of 32":
                mark_progress(progress, result.winner, "Round of 16")
            elif result.stage == "Round of 16":
                mark_progress(progress, result.winner, "Quarterfinal")
            elif result.stage == "Quarterfinal":
                mark_progress(progress, result.winner, "Semifinal")
            elif result.stage == "Semifinal":
                mark_progress(progress, result.winner, "Final")
            elif result.stage == "Final":
                mark_progress(progress, result.winner, "Champion")

        for result in tournament["group_results"] + tournament["knockout_results"]:  # type: ignore[operator]
            goals_for[result.team_1] += result.goals_1
            goals_against[result.team_1] += result.goals_2
            goals_for[result.team_2] += result.goals_2
            goals_against[result.team_2] += result.goals_1

        for team_name, stages in progress.items():
            for stage in stages:
                counters[team_name][stage] += 1

    rows = []
    for team in create_teams():
        champion_probability = counters[team.name]["Champion"] / n_simulations
        rows.append(
            {
                "Team": team.name,
                "Market odds": team.market_odds,
                "Market no-vig champion probability": team.market_implied_probability,
                "Champion probability": champion_probability,
                "Model minus market champion probability": champion_probability - team.market_implied_probability,
                "Final probability": counters[team.name]["Final"] / n_simulations,
                "Semifinal probability": counters[team.name]["Semifinal"] / n_simulations,
                "Quarterfinal probability": counters[team.name]["Quarterfinal"] / n_simulations,
                "Round of 16 probability": counters[team.name]["Round of 16"] / n_simulations,
                "Round of 32 probability": counters[team.name]["Round of 32"] / n_simulations,
                "Average goals scored": goals_for[team.name] / n_simulations,
                "Average goals conceded": goals_against[team.name] / n_simulations,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["Champion probability", "Final probability", "Semifinal probability"],
        ascending=False,
    )
    summary.to_csv(OUTPUT_DIR / "world_cup_2026_monte_carlo_summary.csv", index=False)
    return summary


def run_detailed_monte_carlo(n_simulations: int = 10000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counters = defaultdict(lambda: defaultdict(int))
    goals_for = defaultdict(int)
    goals_against = defaultdict(int)

    for _ in range(n_simulations):
        tournament = simulate_tournament(seed=int(rng.integers(0, 1_000_000_000)))
        qualified_names = set()

        for group, rows in tournament["group_standings"].items():  # type: ignore[union-attr]
            counters[rows[0].team.name]["Win group"] += 1
            counters[rows[1].team.name]["Runner-up group"] += 1
            counters[rows[0].team.name]["Top two group"] += 1
            counters[rows[1].team.name]["Top two group"] += 1

            for row in rows:
                counters[row.team.name]["Average group points total"] += row.points
                counters[row.team.name]["Average group goal difference total"] += row.goal_difference

        for row in tournament["qualified"]["third_place_rows"]:  # type: ignore[index]
            counters[row.team.name]["Best third-place qualifier"] += 1

        for key, team in tournament["qualified"].items():  # type: ignore[union-attr]
            if key.startswith(("1", "2")):
                qualified_names.add(team.name)  # type: ignore[union-attr]

        for row in tournament["qualified"]["third_place_rows"]:  # type: ignore[index]
            qualified_names.add(row.team.name)

        for team_name in qualified_names:
            counters[team_name]["Reach Round of 32"] += 1

        for result in tournament["knockout_results"]:  # type: ignore[union-attr]
            if result.stage == "Round of 32":
                counters[result.winner]["Reach Round of 16"] += 1
            elif result.stage == "Round of 16":
                counters[result.winner]["Reach Quarterfinal"] += 1
            elif result.stage == "Quarterfinal":
                counters[result.winner]["Reach Semifinal"] += 1
            elif result.stage == "Semifinal":
                counters[result.winner]["Reach Final"] += 1
            elif result.stage == "Third-place match":
                counters[result.winner]["Finish third"] += 1
            elif result.stage == "Final":
                counters[result.winner]["Win World Cup"] += 1
                counters[result.loser]["Finish runner-up"] += 1

        for result in tournament["group_results"] + tournament["knockout_results"]:  # type: ignore[operator]
            goals_for[result.team_1] += result.goals_1
            goals_against[result.team_1] += result.goals_2
            goals_for[result.team_2] += result.goals_2
            goals_against[result.team_2] += result.goals_1

    rows = []
    for team in create_teams():
        reach_round_of_32 = counters[team.name]["Reach Round of 32"] / n_simulations
        win_world_cup = counters[team.name]["Win World Cup"] / n_simulations
        rows.append(
            {
                "Team": team.name,
                "Group": team.group,
                "Market odds": team.market_odds,
                "Market no-vig champion probability": team.market_implied_probability,
                "Win group": counters[team.name]["Win group"] / n_simulations,
                "Runner-up group": counters[team.name]["Runner-up group"] / n_simulations,
                "Top two group": counters[team.name]["Top two group"] / n_simulations,
                "Best third-place qualifier": counters[team.name]["Best third-place qualifier"] / n_simulations,
                "Reach Round of 32": reach_round_of_32,
                "Eliminated in group stage": 1 - reach_round_of_32,
                "Reach Round of 16": counters[team.name]["Reach Round of 16"] / n_simulations,
                "Reach Quarterfinal": counters[team.name]["Reach Quarterfinal"] / n_simulations,
                "Reach Semifinal": counters[team.name]["Reach Semifinal"] / n_simulations,
                "Reach Final": counters[team.name]["Reach Final"] / n_simulations,
                "Finish runner-up": counters[team.name]["Finish runner-up"] / n_simulations,
                "Finish third": counters[team.name]["Finish third"] / n_simulations,
                "Win World Cup": win_world_cup,
                "Model minus market champion probability": win_world_cup - team.market_implied_probability,
                "Average group points": counters[team.name]["Average group points total"] / n_simulations,
                "Average group goal difference": counters[team.name]["Average group goal difference total"] / n_simulations,
                "Average tournament goals scored": goals_for[team.name] / n_simulations,
                "Average tournament goals conceded": goals_against[team.name] / n_simulations,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["Win World Cup", "Reach Final", "Reach Semifinal"],
        ascending=False,
    )
    summary.to_csv(OUTPUT_DIR / "world_cup_2026_detailed_monte_carlo_summary.csv", index=False)
    return summary


def result_rows(results: list[MatchResult]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        rows.append(
            {
                "match_id": result.match_id,
                "stage": result.stage,
                "team_1": result.team_1,
                "team_2": result.team_2,
                "goals_1": result.goals_1,
                "goals_2": result.goals_2,
                "winner": result.winner,
                "loser": result.loser,
                "decided_by": result.decided_by,
                "stadium": result.stadium,
                "city": result.city,
                "country": result.country,
                "team_1_xg": round(result.team_1_xg, 3),
                "team_2_xg": round(result.team_2_xg, 3),
                "team_1_distance_from_base_km": round(result.team_1_factors.distance_from_base_km, 1),
                "team_2_distance_from_base_km": round(result.team_2_factors.distance_from_base_km, 1),
                "team_1_travel_fatigue_score": round(result.team_1_factors.travel_fatigue_score, 3),
                "team_2_travel_fatigue_score": round(result.team_2_factors.travel_fatigue_score, 3),
                "team_1_home_advantage_multiplier": round(result.team_1_factors.home_advantage_multiplier, 3),
                "team_2_home_advantage_multiplier": round(result.team_2_factors.home_advantage_multiplier, 3),
                "team_1_player_quality_score": round(result.team_1_factors.player_quality_score, 2),
                "team_2_player_quality_score": round(result.team_2_factors.player_quality_score, 2),
                "team_1_adjusted_attack": round(result.team_1_factors.adjusted_attack, 2),
                "team_2_adjusted_attack": round(result.team_2_factors.adjusted_attack, 2),
                "team_1_adjusted_defence": round(result.team_1_factors.adjusted_defence, 2),
                "team_2_adjusted_defence": round(result.team_2_factors.adjusted_defence, 2),
                "explanation": result.explanation,
            }
        )
    return rows


def team_context_rows() -> list[dict[str, object]]:
    rows = []
    for team in create_teams():
        rows.append(
            {
                "team": team.name,
                "group": team.group,
                "confederation": team.confederation,
                "attack_rating": team.attack_rating,
                "defence_rating": team.defence_rating,
                "overall_rating": team.overall_rating,
                "qualifier_score": team.qualifier_score,
                "confed_competition_score": team.confed_competition_score,
                "elite_league_player_count": team.elite_league_player_count,
                "big_club_player_count": team.big_club_player_count,
                "breakout_player_score": team.breakout_player_score,
                "squad_depth_score": team.squad_depth_score,
                "current_form_score": team.current_form_score,
                "injury_impact": team.injury_impact,
                "market_odds": team.market_odds,
                "market_no_vig_implied_probability": team.market_implied_probability,
                "market_strength_score": team.market_strength_score,
                "manual_note": team.manual_note,
                "base_city": team.base_city,
                "base_country": team.base_country,
                "base_latitude": team.base_latitude,
                "base_longitude": team.base_longitude,
            }
        )
    return rows


def print_group_standings(group_standings: dict[str, list[GroupStanding]]) -> None:
    for group, rows in group_standings.items():
        print(f"\nGroup {group}")
        print("Team                      Pts  GD  GF  GA")
        for row in rows:
            print(f"{row.team.name:<24} {row.points:>3} {row.goal_difference:>3} {row.goals_for:>3} {row.goals_against:>3}")


def probability_value(summary_by_team: pd.DataFrame, team_name: str, column: str) -> float:
    return float(summary_by_team.loc[team_name, column])


def conditional_probability(summary_by_team: pd.DataFrame, team_name: str, numerator: str, denominator: str) -> float:
    denominator_value = probability_value(summary_by_team, team_name, denominator)
    if denominator_value <= 0:
        return 0.0
    return probability_value(summary_by_team, team_name, numerator) / denominator_value


def choose_projected_group_order(group_rows: pd.DataFrame) -> list[str]:
    winner = group_rows.sort_values(
        ["Win group", "Average group points", "Average group goal difference"],
        ascending=False,
    ).iloc[0]["Team"]

    remaining = group_rows[group_rows["Team"].ne(winner)]
    runner_up = remaining.sort_values(
        ["Top two group", "Average group points", "Average group goal difference"],
        ascending=False,
    ).iloc[0]["Team"]

    remaining = remaining[remaining["Team"].ne(runner_up)]
    third = remaining.sort_values(
        ["Reach Round of 32", "Best third-place qualifier", "Average group points"],
        ascending=False,
    ).iloc[0]["Team"]

    fourth = remaining[remaining["Team"].ne(third)].sort_values(
        ["Reach Round of 32", "Average group points", "Average group goal difference"],
        ascending=False,
    ).iloc[0]["Team"]

    return [winner, runner_up, third, fourth]


def build_probability_projected_round_of_32(summary: pd.DataFrame) -> tuple[list[tuple[int, Team, Team]], dict[str, list[str]], list[str]]:
    teams_by_name = {team.name: team for team in create_teams()}
    qualified: dict[str, Team | list[GroupStanding]] = {}
    projected_groups = {}
    projected_third_rows = []

    for group in GROUPS:
        group_rows = summary[summary["Group"].eq(group)]
        order = choose_projected_group_order(group_rows)
        projected_groups[group] = order
        qualified[f"1{group}"] = teams_by_name[order[0]]
        qualified[f"2{group}"] = teams_by_name[order[1]]

        third_team = teams_by_name[order[2]]
        third_probability = float(group_rows[group_rows["Team"].eq(order[2])]["Best third-place qualifier"].iloc[0])
        projected_third_rows.append((third_probability, GroupStanding(team=third_team)))

    projected_third_rows = sorted(projected_third_rows, key=lambda item: item[0], reverse=True)[:8]
    qualified["third_place_rows"] = [row for _, row in projected_third_rows]
    third_qualifiers = [row.team.name for _, row in projected_third_rows]

    return build_round_of_32(qualified), projected_groups, third_qualifiers


def choose_projected_knockout_winner(
    summary_by_team: pd.DataFrame,
    team_1: Team,
    team_2: Team,
    numerator: str,
    denominator: str,
) -> tuple[Team, Team, float, float]:
    team_1_chance = conditional_probability(summary_by_team, team_1.name, numerator, denominator)
    team_2_chance = conditional_probability(summary_by_team, team_2.name, numerator, denominator)
    if team_1_chance >= team_2_chance:
        return team_1, team_2, team_1_chance, team_2_chance
    return team_2, team_1, team_2_chance, team_1_chance


def print_probability_tournament_projection(summary: pd.DataFrame, n_simulations: int) -> None:
    summary_by_team = summary.set_index("Team")
    round_of_32, projected_groups, third_qualifiers = build_probability_projected_round_of_32(summary)

    print("\n" + "=" * 96)
    print(f"PROBABILITY-BASED TOURNAMENT BREAKDOWN ({n_simulations:,} simulations)")
    print("=" * 96)
    print(
        "This projection uses Monte Carlo percentages instead of one random draw: "
        "group order is selected from group-stage probabilities, then knockout "
        "winners are selected by conditional next-round probability."
    )

    print("\nGROUP STAGE PROJECTION")
    for group in GROUPS:
        print(f"\nGroup {group}")
        print("Rank  Team                      Win Grp  Top 2   Best 3rd  Qualify  Avg Pts  Avg GD")
        for rank, team_name in enumerate(projected_groups[group], start=1):
            row = summary_by_team.loc[team_name]
            marker = "Q" if rank <= 2 or team_name in third_qualifiers else "OUT"
            print(
                f"{rank:<5} {team_name:<25} "
                f"{row['Win group']:>7.1%} {row['Top two group']:>7.1%} "
                f"{row['Best third-place qualifier']:>9.1%} {row['Reach Round of 32']:>8.1%} "
                f"{row['Average group points']:>7.2f} {row['Average group goal difference']:>7.2f}  {marker}"
            )

    print("\nPROJECTED BEST THIRD-PLACE QUALIFIERS")
    for team_name in third_qualifiers:
        row = summary_by_team.loc[team_name]
        print(
            f"- {team_name:<22} Group {row['Group']} | "
            f"best-third probability {row['Best third-place qualifier']:.1%}, "
            f"overall qualify {row['Reach Round of 32']:.1%}"
        )

    winners: dict[int, Team] = {}
    losers: dict[int, Team] = {}

    print("\nROUND OF 32 PROJECTION")
    for match_id, team_1, team_2 in round_of_32:
        winner, loser, winner_chance, loser_chance = choose_projected_knockout_winner(
            summary_by_team,
            team_1,
            team_2,
            "Reach Round of 16",
            "Reach Round of 32",
        )
        winners[match_id] = winner
        losers[match_id] = loser
        print(
            f"Match {match_id}: {team_1.name} vs {team_2.name} -> "
            f"{winner.name} advances ({winner_chance:.1%} conditional vs {loser_chance:.1%})"
        )

    round_rules = [
        ("ROUND OF 16 PROJECTION", "Reach Quarterfinal", "Reach Round of 16", NEXT_ROUNDS["Round of 16"]),
        ("QUARTERFINAL PROJECTION", "Reach Semifinal", "Reach Quarterfinal", NEXT_ROUNDS["Quarterfinal"]),
        ("SEMIFINAL PROJECTION", "Reach Final", "Reach Semifinal", NEXT_ROUNDS["Semifinal"]),
    ]

    for title, numerator, denominator, fixtures in round_rules:
        print(f"\n{title}")
        for match_id, source_1, source_2 in fixtures:
            team_1 = winners[source_1]
            team_2 = winners[source_2]
            winner, loser, winner_chance, loser_chance = choose_projected_knockout_winner(
                summary_by_team,
                team_1,
                team_2,
                numerator,
                denominator,
            )
            winners[match_id] = winner
            losers[match_id] = loser
            print(
                f"Match {match_id}: {team_1.name} vs {team_2.name} -> "
                f"{winner.name} advances ({winner_chance:.1%} conditional vs {loser_chance:.1%})"
            )

    print("\nTHIRD-PLACE MATCH PROJECTION")
    third_winner, third_loser, third_winner_chance, third_loser_chance = choose_projected_knockout_winner(
        summary_by_team,
        losers[101],
        losers[102],
        "Finish third",
        "Reach Semifinal",
    )
    print(
        f"Match 103: {losers[101].name} vs {losers[102].name} -> "
        f"{third_winner.name} third ({third_winner_chance:.1%} conditional vs {third_loser_chance:.1%})"
    )

    print("\nFINAL PROJECTION")
    champion, runner_up, champion_chance, runner_up_chance = choose_projected_knockout_winner(
        summary_by_team,
        winners[101],
        winners[102],
        "Win World Cup",
        "Reach Final",
    )
    print(
        f"Match 104: {winners[101].name} vs {winners[102].name} -> "
        f"{champion.name} champion ({champion_chance:.1%} conditional vs {runner_up_chance:.1%})"
    )
    print(f"\nPROJECTED CHAMPION: {champion.name}")
    print(f"PROJECTED RUNNER-UP: {runner_up.name}")
    print(f"PROJECTED THIRD PLACE: {third_winner.name}")
    print("=" * 96)


def build_outright_value_table(summary: pd.DataFrame) -> pd.DataFrame:
    value = summary.copy()
    value["Raw market probability"] = 1 / value["Market odds"]
    value["No-vig edge"] = value["Win World Cup"] - value["Market no-vig champion probability"]
    value["Raw edge"] = value["Win World Cup"] - value["Raw market probability"]
    # Decimal odds include the returned stake. Example: odds 9.00 means a $1
    # winning bet returns $9 total, for $8 net profit.
    value["Expected total return per $1 stake"] = value["Win World Cup"] * value["Market odds"]
    value["Expected net profit per $1 stake"] = value["Expected total return per $1 stake"] - 1

    max_probability = value["Win World Cup"].max()
    value["Contender tier"] = np.select(
        [
            value["Win World Cup"] >= 0.05,
            value["Win World Cup"] >= 0.025,
            value["Win World Cup"] >= 0.01,
        ],
        [
            "true contender",
            "outside contender",
            "credible longshot",
        ],
        default="lottery longshot",
    )

    value["Probability index"] = value["Win World Cup"] / max_probability
    value["Contender value score"] = (
        0.60 * value["Probability index"]
        + 0.25 * (value["Expected total return per $1 stake"].clip(upper=2.0) / 2.0)
        + 0.15 * ((value["No-vig edge"] + 0.05).clip(lower=0, upper=0.10) / 0.10)
    )

    value = value.sort_values(
        ["Contender value score", "Win World Cup", "Expected net profit per $1 stake"],
        ascending=False,
    )
    value.to_csv(OUTPUT_DIR / "world_cup_2026_outright_value_comparison.csv", index=False)
    return value


def print_outright_contender_value(summary: pd.DataFrame) -> None:
    value = build_outright_value_table(summary)
    contender_columns = [
        "Team",
        "Market odds",
        "Win World Cup",
        "Raw market probability",
        "Market no-vig champion probability",
        "No-vig edge",
        "Expected total return per $1 stake",
        "Expected net profit per $1 stake",
        "Contender value score",
    ]

    true_contenders = value[value["Contender tier"].eq("true contender")].copy()
    print("\nOUTRIGHT WINNER VALUE: TRUE CONTENDERS ONLY")
    print(
        true_contenders[contender_columns]
        .sort_values(["Contender value score", "Win World Cup"], ascending=False)
        .to_string(
            index=False,
            formatters={
                "Win World Cup": "{:.2%}".format,
                "Raw market probability": "{:.2%}".format,
                "Market no-vig champion probability": "{:.2%}".format,
                "No-vig edge": "{:+.2%}".format,
                "Expected total return per $1 stake": "${:.3f}".format,
                "Expected net profit per $1 stake": "{:+.2%}".format,
                "Contender value score": "{:.3f}".format,
            },
        )
    )

    positive_ev = true_contenders[true_contenders["Expected net profit per $1 stake"] > 0]
    if positive_ev.empty:
        best = true_contenders.sort_values("Contender value score", ascending=False).iloc[0]
        print(
            "\nModel read: no true contender is positive raw EV at these prices. "
            f"The best contender/value compromise is {best['Team']} at {best['Market odds']:.2f}."
        )
    else:
        best = positive_ev.sort_values(["Contender value score", "Win World Cup"], ascending=False).iloc[0]
        print(
            f"\nModel read: best positive-EV true contender is {best['Team']} "
            f"at {best['Market odds']:.2f}."
        )

    print("\nBEST OUTSIDE CONTENDERS / LONGSHOTS")
    print(
        value[value["Contender tier"].isin(["outside contender", "credible longshot"])]
        .head(10)[contender_columns + ["Contender tier"]]
        .to_string(
            index=False,
            formatters={
                "Win World Cup": "{:.2%}".format,
                "Raw market probability": "{:.2%}".format,
                "Market no-vig champion probability": "{:.2%}".format,
                "No-vig edge": "{:+.2%}".format,
                "Expected total return per $1 stake": "${:.3f}".format,
                "Expected net profit per $1 stake": "{:+.2%}".format,
                "Contender value score": "{:.3f}".format,
            },
        )
    )


def main() -> None:
    tournament = simulate_tournament()
    group_results = tournament["group_results"]
    knockout_results = tournament["knockout_results"]

    pd.DataFrame(team_context_rows()).to_csv(OUTPUT_DIR / "world_cup_2026_team_context.csv", index=False)
    pd.DataFrame(result_rows(group_results)).to_csv(OUTPUT_DIR / "world_cup_2026_group_results.csv", index=False)
    pd.DataFrame(result_rows(knockout_results)).to_csv(OUTPUT_DIR / "world_cup_2026_knockout_results.csv", index=False)

    print("Saved one sample tournament to:")
    print(f"- {OUTPUT_DIR / 'world_cup_2026_group_results.csv'}")
    print(f"- {OUTPUT_DIR / 'world_cup_2026_knockout_results.csv'}")

    n_simulations = 10_000
    print(f"\nRunning detailed Monte Carlo projection ({n_simulations:,} tournaments)...")
    detailed_summary = run_detailed_monte_carlo(n_simulations=n_simulations)
    print_probability_tournament_projection(detailed_summary, n_simulations)

    top_columns = [
        "Team",
        "Win group",
        "Reach Round of 32",
        "Reach Round of 16",
        "Reach Quarterfinal",
        "Reach Semifinal",
        "Reach Final",
        "Win World Cup",
    ]
    probability_formatters = {
        column: "{:.1%}".format
        for column in top_columns
        if column != "Team"
    }
    print("\nTOP 16 TITLE CHANCES")
    print(detailed_summary[top_columns].head(16).to_string(index=False, formatters=probability_formatters))
    print_outright_contender_value(detailed_summary)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
