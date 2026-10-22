"""Historical Data Loader and Team Strength Calculator for QuantShield.

Downloads match results from Football-Data.co.uk, matches team names using RapidFuzz,
and calculates Poisson lambdas (Fuerza Ataque * Fuerza Defensa * Media de Goles).
Provides fallbacks for international cups and non-CSV leagues.
"""
from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from rapidfuzz import fuzz

CACHE_DIR_NAME = "data_cache"


def get_cache_dir() -> str:
    """Returns the path to the historical data cache directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_or_load_csv(csv_code: str) -> Optional[pd.DataFrame]:
    """Downloads or loads cached CSV from Football-Data.co.uk for a given league code (e.g., N1, D2, I2)."""
    cache_path = os.path.join(get_cache_dir(), f"{csv_code}.csv")

    # Try downloading the latest season (2526 or 2425)
    seasons = ["2526", "2425"]
    for season in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{csv_code}.csv"
        try:
            try:
                resp = requests.get(url, timeout=7)
            except requests.exceptions.SSLError:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                resp = requests.get(url, timeout=7, verify=False)

            if resp.status_code == 200 and len(resp.content) > 500:
                df = pd.read_csv(io.StringIO(resp.text), encoding="latin1")
                # Ensure essential columns exist
                if "HomeTeam" in df.columns and "AwayTeam" in df.columns and "FTHG" in df.columns:
                    # Save to local cache
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            f.write(resp.text)
                    except Exception:
                        pass
                    return df
        except Exception:
            continue

    # Fallback to local cache if offline or failed download
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path, encoding="utf-8")
        except Exception:
            try:
                return pd.read_csv(cache_path, encoding="latin1")
            except Exception:
                return None

    return None


def match_team_name(target_name: str, candidates: List[str], threshold: float = 70.0) -> Optional[str]:
    """Finds best matching team name using rapidfuzz token_sort_ratio > threshold."""
    if not candidates or not target_name:
        return None

    best_match = None
    highest_score = 0.0

    target_clean = target_name.strip()
    for cand in candidates:
        cand_clean = str(cand).strip()
        score = fuzz.token_sort_ratio(target_clean, cand_clean)
        if score > highest_score and score >= threshold:
            highest_score = score
            best_match = cand_clean

    return best_match


def calculate_poisson_lambdas_from_df(
    df: pd.DataFrame, home_team: str, away_team: str
) -> Optional[Tuple[float, float]]:
    """Calculates lambda_local and lambda_visita using Football-Data stats and RapidFuzz team matching:
    Fuerza Ataque * Fuerza Defensa * Media de Goles.
    """
    valid = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    valid["FTHG"] = pd.to_numeric(valid["FTHG"], errors="coerce")
    valid["FTAG"] = pd.to_numeric(valid["FTAG"], errors="coerce")
    valid = valid.dropna(subset=["FTHG", "FTAG"])

    if len(valid) < 10:
        return None

    all_teams = list(set(valid["HomeTeam"].unique()).union(set(valid["AwayTeam"].unique())))
    matched_home = match_team_name(home_team, all_teams, threshold=70.0)
    matched_away = match_team_name(away_team, all_teams, threshold=70.0)

    if not matched_home or not matched_away:
        return None

    # League averages
    total_matches = len(valid)
    total_home_goals = valid["FTHG"].sum()
    total_away_goals = valid["FTAG"].sum()

    league_avg_home = total_home_goals / total_matches
    league_avg_away = total_away_goals / total_matches

    # Home team stats as host
    home_matches_h = valid[valid["HomeTeam"] == matched_home]
    n_home_h = len(home_matches_h)
    if n_home_h == 0:
        return None
    home_attack = (home_matches_h["FTHG"].sum() / n_home_h) / max(0.5, league_avg_home)
    home_defense = (home_matches_h["FTAG"].sum() / n_home_h) / max(0.5, league_avg_away)

    # Away team stats as visitor
    away_matches_a = valid[valid["AwayTeam"] == matched_away]
    n_away_a = len(away_matches_a)
    if n_away_a == 0:
        return None
    away_attack = (away_matches_a["FTAG"].sum() / n_away_a) / max(0.5, league_avg_away)
    away_defense = (away_matches_a["FTHG"].sum() / n_away_a) / max(0.5, league_avg_home)

    # Expected goals
    lambda_local = home_attack * away_defense * league_avg_home
    lambda_visita = away_attack * home_defense * league_avg_away

    # Guard bounds for realistic Poisson rates
    lambda_local = max(0.40, min(lambda_local, 4.20))
    lambda_visita = max(0.35, min(lambda_visita, 3.80))

    return round(float(lambda_local), 3), round(float(lambda_visita), 3)


def derive_lambdas_from_oracle(
    p_home_clean: float,
    p_draw_clean: float,
    p_away_clean: float,
    tournament_avg_goals: float = 2.60,
    home_advantage: float = 1.15,
) -> Tuple[float, float]:
    """Derives lambda_local and lambda_visita directly from oracle vig-free probabilities
    weighted by tournament historical goal average and home advantage factor.
    Used for International Cups (Champions, Libertadores, Copa América) and leagues without direct CSV.
    """
    p_h = max(0.05, min(p_home_clean, 0.90))
    p_a = max(0.05, min(p_away_clean, 0.90))
    p_x = max(0.05, min(p_draw_clean, 0.60))

    # Adjust expected total goals: higher draw probability indicates tighter, lower-scoring game
    # Standard draw prob ~0.27; if draw prob is 0.35, expected goals is lower
    goal_factor = 1.0 - (p_x - 0.27) * 1.2
    expected_total = max(1.80, min(tournament_avg_goals * goal_factor, 3.50))

    # Power ratio between home and away win probabilities
    ratio = ((p_h / p_a) ** 0.5) * home_advantage
    ratio = max(0.25, min(ratio, 4.0))

    lambda_local = expected_total * (ratio / (1.0 + ratio))
    lambda_visita = expected_total * (1.0 / (1.0 + ratio))

    return round(float(max(0.40, lambda_local)), 3), round(float(max(0.35, lambda_visita)), 3)


def load_historical_forces(
    csv_code: Optional[str],
    home_team: str,
    away_team: str,
    oracle_probs: Optional[Tuple[float, float, float]] = None,
    tournament_avg_goals: float = 2.60,
) -> Tuple[float, float, str]:
    """Main loader function:
    1. Attempts to calculate forces from football-data CSV if csv_code is present.
    2. Falls back to deriving lambdas from oracle clean probabilities.

    Returns:
        (lambda_local, lambda_visita, calculation_source_description)
    """
    if csv_code:
        df = download_or_load_csv(csv_code)
        if df is not None:
            lambdas = calculate_poisson_lambdas_from_df(df, home_team, away_team)
            if lambdas:
                return lambdas[0], lambdas[1], f"CSV Histórico ({csv_code}) + RapidFuzz"

    # Fallback to oracle probabilities
    if oracle_probs is not None:
        p1, px, p2 = oracle_probs
        l_h, l_a = derive_lambdas_from_oracle(p1, px, p2, tournament_avg_goals=tournament_avg_goals)
        return l_h, l_a, "Modelo Oráculo Vig-Free + Media Torneo"

    # Default generic lambdas if no data available
    return 1.45, 1.15, "Valores Base de Referencia"
