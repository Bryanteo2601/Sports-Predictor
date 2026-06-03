"""
Scrape 2026 World Cup venue and base-camp data from Wikipedia.

Outputs:
- data/world_cup_2026_team_base_camps.csv
- data/world_cup_2026_match_schedule.csv
"""

from __future__ import annotations

import io
import re
import ssl
import urllib.request
from pathlib import Path

import certifi
import pandas as pd
from bs4 import BeautifulSoup


URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
DATA_DIR = Path("data")
BASE_CAMPS_PATH = DATA_DIR / "world_cup_2026_team_base_camps.csv"
MATCH_SCHEDULE_PATH = DATA_DIR / "world_cup_2026_match_schedule.csv"

TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Türkiye": "Turkey",
}

CITY_COORDINATES = {
    "Alajuela": ("Costa Rica", 10.0162, -84.2116),
    "Alameda": ("United States", 37.7652, -122.2416),
    "Alexandria": ("United States", 38.8048, -77.0469),
    "Ann Arbor": ("United States", 42.2808, -83.7430),
    "Atlanta": ("United States", 33.7490, -84.3880),
    "Austin": ("United States", 30.2672, -97.7431),
    "Basking Ridge": ("United States", 40.7062, -74.5493),
    "Boca Raton": ("United States", 26.3683, -80.1289),
    "Carson": ("United States", 33.8314, -118.2820),
    "Charlotte": ("United States", 35.2271, -80.8431),
    "Chattanooga": ("United States", 35.0456, -85.3097),
    "Chester": ("United States", 39.8496, -75.3557),
    "Cincinnati": ("United States", 39.1031, -84.5120),
    "Columbus": ("United States", 39.9612, -82.9988),
    "Commerce City": ("United States", 39.8083, -104.9339),
    "Dallas": ("United States", 32.7767, -96.7970),
    "Dearborn": ("United States", 42.3223, -83.1763),
    "Denver": ("United States", 39.7392, -104.9903),
    "Fort Lauderdale": ("United States", 26.1224, -80.1373),
    "Frisco": ("United States", 33.1507, -96.8236),
    "Galloway Township": ("United States", 39.4928, -74.5597),
    "Green Bay": ("United States", 44.5133, -88.0133),
    "Greensboro": ("United States", 36.0726, -79.7920),
    "Guadalupe": ("Mexico", 25.6768, -100.2565),
    "Guadalajara": ("Mexico", 20.6597, -103.3496),
    "Hidalgo": ("Mexico", 20.0911, -98.7624),
    "Houston": ("United States", 29.7604, -95.3698),
    "Irvine": ("United States", 33.6846, -117.8265),
    "Kansas City": ("United States", 39.0997, -94.5786),
    "Kennesaw": ("United States", 34.0234, -84.6155),
    "Lawrence": ("United States", 38.9717, -95.2353),
    "Los Angeles": ("United States", 34.0522, -118.2437),
    "Mansfield": ("United States", 32.5632, -97.1417),
    "Marietta": ("United States", 33.9526, -84.5499),
    "Mesa": ("United States", 33.4152, -111.8315),
    "Mexico City": ("Mexico", 19.4326, -99.1332),
    "Miami": ("United States", 25.7617, -80.1918),
    "Minneapolis": ("United States", 44.9778, -93.2650),
    "Montreal": ("Canada", 45.5017, -73.5673),
    "Morristown": ("United States", 40.7968, -74.4815),
    "Nashville": ("United States", 36.1627, -86.7816),
    "New York": ("United States", 40.7128, -74.0060),
    "New Tecumseth": ("Canada", 44.0795, -79.7858),
    "Orlando": ("United States", 28.5383, -81.3792),
    "Palm Beach Gardens": ("United States", 26.8234, -80.1387),
    "Philadelphia": ("United States", 39.9526, -75.1652),
    "Piscataway": ("United States", 40.5549, -74.4643),
    "Playa del Carmen": ("Mexico", 20.6296, -87.0739),
    "Portland": ("United States", 45.5152, -122.6784),
    "Renton": ("United States", 47.4829, -122.2171),
    "Riverside": ("United States", 39.1775, -94.6130),
    "Salt Lake City": ("United States", 40.7608, -111.8910),
    "San Jose": ("United States", 37.3382, -121.8863),
    "San Agustin Tlaxiaca": ("Mexico", 20.1156, -98.8872),
    "Sandy": ("United States", 40.5649, -111.8389),
    "Santa Barbara": ("United States", 34.4208, -119.6982),
    "San Diego": ("United States", 32.7157, -117.1611),
    "San Francisco": ("United States", 37.7749, -122.4194),
    "Santiago": ("Mexico", 25.4251, -100.1414),
    "Santa Clara": ("United States", 37.3541, -121.9552),
    "Seattle": ("United States", 47.6062, -122.3321),
    "Smithfield": ("United States", 41.9220, -71.5495),
    "Spokane": ("United States", 47.6588, -117.4260),
    "St. Louis": ("United States", 38.6270, -90.1994),
    "Tampa": ("United States", 27.9506, -82.4572),
    "Tijuana": ("Mexico", 32.5149, -117.0382),
    "Toronto": ("Canada", 43.6532, -79.3832),
    "Vancouver": ("Canada", 49.2827, -123.1207),
    "Waltham": ("United States", 42.3765, -71.2356),
    "White Sulphur Springs": ("United States", 37.7965, -80.2976),
    "Winston-Salem": ("United States", 36.0999, -80.2442),
    "Zapopan": ("Mexico", 20.7236, -103.3848),
}

VENUE_CITY_ALIASES = {
    "Arlington": "Dallas",
    "East Rutherford": "New York",
    "Foxborough": "Boston",
    "Guadalupe": "Monterrey",
    "Inglewood": "Los Angeles",
    "Kansas City, Missouri": "Kansas City",
    "Miami Gardens": "Miami",
    "Monterrey": "Monterrey",
    "Santa Clara": "San Francisco Bay Area",
    "Zapopan": "Guadalajara",
}

VENUE_COORDINATES = {
    "Atlanta": ("United States", 33.7555, -84.4008),
    "Boston": ("United States", 42.0909, -71.2643),
    "Dallas": ("United States", 32.7473, -97.0945),
    "Guadalajara": ("Mexico", 20.6818, -103.4627),
    "Houston": ("United States", 29.6847, -95.4107),
    "Kansas City": ("United States", 39.0489, -94.4839),
    "Los Angeles": ("United States", 33.9535, -118.3392),
    "Mexico City": ("Mexico", 19.3029, -99.1505),
    "Miami": ("United States", 25.9580, -80.2389),
    "Monterrey": ("Mexico", 25.6689, -100.2446),
    "New York": ("United States", 40.8135, -74.0745),
    "Philadelphia": ("United States", 39.9008, -75.1675),
    "San Francisco Bay Area": ("United States", 37.4030, -121.9700),
    "Seattle": ("United States", 47.5952, -122.3316),
    "Toronto": ("Canada", 43.6332, -79.4186),
    "Vancouver": ("Canada", 49.2768, -123.1119),
}


def clean_text(value: object) -> str:
    text = str(value)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_team(value: str) -> str:
    value = clean_text(value)
    return TEAM_ALIASES.get(value, value)


def fetch_html() -> bytes:
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(URL, headers={"User-Agent": "Sports-Predictor/1.0"})
    with urllib.request.urlopen(request, context=context) as response:
        return response.read()


def last_known_city(text: str) -> str:
    clean = clean_text(text)
    matches = re.findall(r", ([A-Za-z .'-]+?)(?:,|$)", clean)
    if not matches:
        return ""
    return matches[-1].strip()


def base_city_from_training_site(training_site: str) -> str:
    if "San Agust" in training_site and "Tlaxiaca" in training_site:
        return "San Agustin Tlaxiaca"
    city = last_known_city(training_site)
    if city == "Missouri":
        return "Kansas City"
    return city


def scrape_base_camps(tables: list[pd.DataFrame]) -> pd.DataFrame:
    table = next(table for table in tables if {"Team", "Hotel", "Training site"}.issubset(set(map(str, table.columns))))
    rows = []
    for _, row in table.iterrows():
        team = normalize_team(row["Team"])
        training_site = clean_text(row["Training site"])
        city = base_city_from_training_site(training_site)
        country, latitude, longitude = CITY_COORDINATES.get(city, ("", None, None))
        rows.append(
            {
                "team": team,
                "hotel": clean_text(row["Hotel"]),
                "training_site": training_site,
                "base_city": city,
                "base_country": country,
                "base_latitude": latitude,
                "base_longitude": longitude,
            }
        )
    return pd.DataFrame(rows)


def scrape_match_schedule(soup: BeautifulSoup) -> pd.DataFrame:
    rows = []
    for box in soup.select("div.footballbox"):
        score = box.select_one(".fscore")
        if score is None:
            continue
        match_text = clean_text(score.get_text(" "))
        match = re.search(r"Match (\d+)", match_text)
        if not match:
            continue

        date_node = box.select_one(".bday")
        date_text = clean_text(date_node.get_text(" ")) if date_node else ""
        home_node = box.select_one(".fhome [itemprop='name']")
        away_node = box.select_one(".faway [itemprop='name']")
        venue_text = clean_text(box.select_one(".fright").get_text(" ")) if box.select_one(".fright") else ""
        stadium, raw_city = [part.strip() for part in venue_text.rsplit(",", 1)] if "," in venue_text else (venue_text, "")
        venue_city = VENUE_CITY_ALIASES.get(raw_city, raw_city)
        country, latitude, longitude = VENUE_COORDINATES.get(venue_city, ("", None, None))

        rows.append(
            {
                "match_id": int(match.group(1)),
                "date": date_text,
                "home_team": normalize_team(home_node.get_text(" ")) if home_node else "",
                "away_team": normalize_team(away_node.get_text(" ")) if away_node else "",
                "stadium": stadium,
                "raw_city": raw_city,
                "venue_city": venue_city,
                "venue_country": country,
                "venue_latitude": latitude,
                "venue_longitude": longitude,
            }
        )
    return pd.DataFrame(rows).sort_values("match_id")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    html = fetch_html()
    tables = pd.read_html(io.BytesIO(html))
    soup = BeautifulSoup(html, "html.parser")

    base_camps = scrape_base_camps(tables)
    schedule = scrape_match_schedule(soup)
    base_camps.to_csv(BASE_CAMPS_PATH, index=False)
    schedule.to_csv(MATCH_SCHEDULE_PATH, index=False)

    missing_bases = base_camps[base_camps["base_latitude"].isna()]
    missing_venues = schedule[schedule["venue_latitude"].isna()]
    print(f"Saved {len(base_camps)} base camps to {BASE_CAMPS_PATH}")
    print(f"Saved {len(schedule)} matches to {MATCH_SCHEDULE_PATH}")
    if not missing_bases.empty:
        print("Base camps missing coordinates:")
        print(missing_bases[["team", "training_site", "base_city"]].to_string(index=False))
    if not missing_venues.empty:
        print("Venues missing coordinates:")
        print(missing_venues[["match_id", "stadium", "raw_city", "venue_city"]].to_string(index=False))


if __name__ == "__main__":
    main()
