"""
Stadium coords + NOAA hourly wind forecast.

Backs the wind-UNDER totals play surfaced in app.py. NOAA is free, no API key,
US-only — perfect for the NFL. Forecast horizon is ~6.5 days; for games further
out we return None and the UI shows "forecast pending".
"""
from __future__ import annotations
import datetime as dt
import json
import re
import urllib.request
import urllib.error
from functools import lru_cache

USER_AGENT = "nfl-sports-model (oscar@paisalocal.com)"

# (lat, lon) for each NFL team's home stadium. Used by NOAA's /points endpoint
# to resolve forecast grid. Shared-stadium teams (LA/LAC at SoFi, NYG/NYJ at
# MetLife) share coords by design.
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.5275, -112.2625),  # State Farm Stadium (closed roof, weather irrelevant)
    "ATL": (33.7553,  -84.4006),  # Mercedes-Benz (retractable)
    "BAL": (39.2780,  -76.6227),  # M&T Bank
    "BUF": (42.7738,  -78.7867),  # Highmark
    "CAR": (35.2258,  -80.8528),  # Bank of America
    "CHI": (41.8623,  -87.6167),  # Soldier Field
    "CIN": (39.0954,  -84.5160),  # Paycor
    "CLE": (41.5060,  -81.6995),
    "DAL": (32.7473,  -97.0945),  # AT&T (retractable)
    "DEN": (39.7438, -105.0203),
    "DET": (42.3400,  -83.0456),  # Ford Field (dome)
    "GB":  (44.5013,  -88.0622),  # Lambeau
    "HOU": (29.6847,  -95.4107),  # NRG (retractable)
    "IND": (39.7601,  -86.1639),  # Lucas Oil (retractable)
    "JAX": (30.3239,  -81.6373),  # EverBank
    "KC":  (39.0489,  -94.4839),  # Arrowhead
    "LA":  (33.9534, -118.3387),  # SoFi (fixed-roof)
    "LAC": (33.9534, -118.3387),  # SoFi
    "LV":  (36.0909, -115.1830),  # Allegiant (dome)
    "MIA": (25.9580,  -80.2389),  # Hard Rock
    "MIN": (44.9737,  -93.2581),  # US Bank (dome)
    "NE":  (42.0909,  -71.2643),  # Gillette
    "NO":  (29.9509,  -90.0815),  # Caesars Superdome
    "NYG": (40.8135,  -74.0745),  # MetLife
    "NYJ": (40.8135,  -74.0745),  # MetLife
    "PHI": (39.9008,  -75.1675),  # Lincoln Financial
    "PIT": (40.4468,  -80.0158),  # Acrisure
    "SEA": (47.5952, -122.3316),  # Lumen
    "SF":  (37.4032, -121.9698),  # Levi's
    "TB":  (27.9759,  -82.5033),  # Raymond James
    "TEN": (36.1665,  -86.7713),  # Nissan
    "WAS": (38.9078,  -76.8645),  # Northwest
}


def _http_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


@lru_cache(maxsize=64)
def _forecast_hourly_url(lat: float, lon: float) -> str:
    """NOAA gives a stable per-gridpoint URL. Cache to avoid re-resolving points."""
    pts = _http_get(f"https://api.weather.gov/points/{lat},{lon}")
    return pts["properties"]["forecastHourly"]


def _parse_mph(s: str | None) -> float | None:
    """NOAA wind comes as e.g. '12 mph' or '5 to 10 mph' — take the upper bound."""
    if not s:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    return float(nums[-1]) if nums else None


def forecast_wind_at(lat: float, lon: float, kickoff_utc: dt.datetime) -> float | None:
    """Forecast wind in mph at the hour closest to kickoff. None if out of NOAA's window."""
    try:
        url = _forecast_hourly_url(lat, lon)
        data = _http_get(url)
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError):
        return None
    if kickoff_utc.tzinfo is None:
        kickoff_utc = kickoff_utc.replace(tzinfo=dt.timezone.utc)
    best, best_dt = None, None
    for p in data["properties"]["periods"]:
        start = dt.datetime.fromisoformat(p["startTime"]).astimezone(dt.timezone.utc)
        delta = abs((start - kickoff_utc).total_seconds())
        if best_dt is None or delta < best_dt:
            best_dt, best = delta, p
    # If the closest period is more than 2 hours away, the kickoff is outside NOAA's window
    if best is None or best_dt > 2 * 3600:
        return None
    return _parse_mph(best.get("windSpeed"))


def forecast_wind_for_game(home_team: str, kickoff_utc: dt.datetime) -> float | None:
    """Convenience wrapper: look up coords from STADIUM_COORDS by home team abbreviation."""
    coords = STADIUM_COORDS.get(home_team)
    if not coords:
        return None
    return forecast_wind_at(coords[0], coords[1], kickoff_utc)


def under_recommendation(wind_mph: float | None, roof: str | None) -> tuple[str, str]:
    """
    Map (wind, roof) → (recommendation, reason).
    Backed by 1999-2025 backtest in notebook 05: wind>=10 UNDER hits 54.7% (n=1,770),
    wind>=15 hits 56.2% (n=642). Indoor games get no weather play.
    """
    if roof in ("dome", "closed"):
        return ("— indoor", "controlled climate, weather edge does not apply")
    if wind_mph is None:
        return ("⏳ pending", "forecast not yet available (NOAA window is ~6.5 days)")
    if wind_mph >= 15:
        return ("🔥 STRONG UNDER", f"wind {wind_mph:.0f} mph — historical hit 56.2% (n=642)")
    if wind_mph >= 10:
        return ("📈 LEAN UNDER",   f"wind {wind_mph:.0f} mph — historical hit 54.7% (n=1,770)")
    return ("— pass", f"wind {wind_mph:.0f} mph — below 10 mph threshold")
