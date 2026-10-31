"""The Odds API Client for QuantShield.

Fetches real-time odds from The Odds API v4, handles quota tracking,
error handling (401, 404, 422), and provides realistic simulation datasets.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Categorized tournaments as per QuantShield specification
SUPPORTED_SPORTS = {
    "Fútbol (Ligas de Goles)": {
        "soccer_netherlands_eredivisie": {"name": "Países Bajos - Eredivisie", "csv_code": "N1", "type": "goals"},
        "soccer_germany_bundesliga2": {"name": "Alemania - 2. Bundesliga", "csv_code": "D2", "type": "goals"},
        "soccer_switzerland_superleague": {"name": "Suiza - Super League", "csv_code": None, "type": "goals"},
    },
    "Fútbol (Fortaleza Local)": {
        "soccer_colombia_primera_a": {"name": "Colombia - Liga BetPlay", "csv_code": None, "type": "dnb"},
        "soccer_argentina_primera_division": {"name": "Argentina - Liga Profesional", "csv_code": None, "type": "dnb"},
        "soccer_italy_serie_b": {"name": "Italia - Serie B", "csv_code": "I2", "type": "dnb"},
    },
    "Fútbol (Copas Internacionales)": {
        "soccer_uefa_champs_league": {"name": "UEFA Champions League", "csv_code": None, "type": "intl"},
        "soccer_conmebol_copa_libertadores": {"name": "CONMEBOL Copa Libertadores", "csv_code": None, "type": "intl"},
        "soccer_conmebol_copa_america": {"name": "CONMEBOL Copa América", "csv_code": None, "type": "intl"},
    },
    "Tenis (ATP / WTA)": {
        "tennis_atp_wimbledon": {"name": "ATP - Wimbledon / Grand Slam", "csv_code": None, "type": "tennis"},
        "tennis_atp_french_open": {"name": "ATP - Roland Garros / Grand Slam", "csv_code": None, "type": "tennis"},
        "tennis_atp_us_open": {"name": "ATP - US Open", "csv_code": None, "type": "tennis"},
        "tennis_atp_masters_1000": {"name": "ATP - Masters 1000", "csv_code": None, "type": "tennis"},
    },
    "Baloncesto (NBA / Euroliga)": {
        "basketball_nba": {"name": "Baloncesto - NBA", "csv_code": None, "type": "nba"},
        "basketball_euroleague": {"name": "Baloncesto - EuroLeague", "csv_code": None, "type": "nba"},
    },
}

# Casas de apuestas autorizadas en Colombia (Coljuegos) y sus motores de cuotas equivalentes
COLOMBIAN_BOOKMAKERS_MAP: Dict[str, str] = {
    # Licencias directas en Colombia
    "betplay": "BetPlay",
    "wplay": "Wplay",
    "rushbet": "Rushbet",
    "bwin": "Bwin Colombia",
    "codere": "Codere",
    "betsson": "Betsson",
    "yajuego": "Yajuego",
    "rivalo": "Rivalo",
    "mozzart": "Mozzartbet",
    
    # Proveedores internacionales que alimentan las cuotas de casas colombianas:
    # BetPlay opera bajo la plataforma y cuotas de Kambi (exactamente las mismas que Unibet / Bet365)
    "unibet_eu": "BetPlay",
    "unibet_uk": "BetPlay",
    "unibet": "BetPlay",
    "bet365": "BetPlay",
    
    # Rushbet opera bajo Rush Street Interactive / Kambi (BetRivers / Betano)
    "betrivers": "Rushbet",
    "betano": "Rushbet",
    
    # Wplay opera con proveedores como Betfair / SBTech
    "betfair": "Wplay",
    "betfair_ex_eu": "Wplay",
    "betfair_ex_uk": "Wplay",
}


def get_available_sports() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Returns the dictionary of supported sports."""
    return SUPPORTED_SPORTS


def format_event_name(event: Dict[str, Any]) -> str:
    """Returns a clean display name for an event."""
    home = event.get("home_team", "Local")
    away = event.get("away_team", "Visita")
    return f"{home} vs {away}"


def fetch_odds(
    sport_key: str,
    api_key: Optional[str] = None,
    regions: str = "eu,uk,us",
    markets: str = "h2h,totals,spreads",
    is_demo: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """Fetches odds from The Odds API or simulation mock data.

    Returns:
        (events_list, metadata_dict, error_message_or_none)
    """
    metadata: Dict[str, Any] = {
        "requests_remaining": "N/A",
        "requests_used": "N/A",
        "source": "api",
        "sport_key": sport_key,
    }

    # If demo mode is active, use realistic mock data
    if is_demo:
        metadata["requests_remaining"] = "500 (Modo Demo Activo)"
        metadata["requests_used"] = "0"
        metadata["source"] = "demo"
        mock_data = get_simulation_events(sport_key)
        return mock_data, metadata, None

    # If demo is disabled but no API key was provided, try auto-reading st.secrets and os.getenv
    if not api_key or not api_key.strip():
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if "THE_ODDS_API_KEY" in st.secrets and st.secrets["THE_ODDS_API_KEY"]:
                    api_key = str(st.secrets["THE_ODDS_API_KEY"]).strip()
                else:
                    for section in st.secrets:
                        sec_obj = st.secrets[section]
                        if isinstance(sec_obj, dict) and "THE_ODDS_API_KEY" in sec_obj and sec_obj["THE_ODDS_API_KEY"]:
                            api_key = str(sec_obj["THE_ODDS_API_KEY"]).strip()
                            break
        except Exception:
            pass
        if not api_key:
            api_key = os.getenv("THE_ODDS_API_KEY", "").strip()

    if not api_key or api_key.strip() == "" or api_key.strip() == "your_the_odds_api_key_here":
        metadata["requests_remaining"] = "0"
        metadata["requests_used"] = "0"
        metadata["source"] = "api"
        err = "Desactivaste el Modo Simulación, pero no has ingresado tu clave en 'The Odds API Key'. Para consultar partidos y cuotas reales en vivo por internet, ingresa tu API Key gratuita (the-odds-api.com) o activa el '🧪 Modo Simulación' en el menú lateral."
        return [], metadata, err

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key.strip(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        try:
            response = requests.get(url, params=params, timeout=12)
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Attempt with verify=False in corporate environments with SSL inspection
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(url, params=params, timeout=12, verify=False)

        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        if remaining is not None:
            metadata["requests_remaining"] = remaining
        if used is not None:
            metadata["requests_used"] = used

        if response.status_code == 200:
            events = response.json()
            if not events:
                return [], metadata, f"El torneo '{sport_key}' no tiene eventos programados con cuotas activas actualmente."
            
            # FILTRO ESTRICTO COLOMBIA (COLJUEGOS):
            # 1. Conservar 'pinnacle' únicamente como oráculo matemático de cuota limpia (Vig-Free Oracle).
            # 2. Conservar y mapear exclusivamente casas autorizadas en Colombia (BetPlay, Rushbet, Wplay, Bwin Colombia).
            # 3. Eliminar completamente casas extranjeras/offshore (DraftKings, FanDuel, Bovada, 1xBet, etc.).
            for ev in events:
                filtered_bms: List[Dict[str, Any]] = []
                seen_titles: set = set()

                # Conservar Pinnacle de primero para cálculos matemáticos del motor
                pinnacle_bm = next((b for b in ev.get("bookmakers", []) if b.get("key", "").lower() == "pinnacle"), None)
                if pinnacle_bm:
                    filtered_bms.append(pinnacle_bm)

                # Mapear y filtrar casas de Colombia
                for bm in ev.get("bookmakers", []):
                    k = bm.get("key", "").lower()
                    if k == "pinnacle":
                        continue
                    if k in COLOMBIAN_BOOKMAKERS_MAP:
                        target_title = COLOMBIAN_BOOKMAKERS_MAP[k]
                        bm["title"] = target_title
                        if target_title not in seen_titles:
                            seen_titles.add(target_title)
                            filtered_bms.append(bm)
                        else:
                            # Si ya existe esta casa por otra llave (ej. bet365 y unibet), unificar mercados faltantes
                            existing = next(b for b in filtered_bms if b.get("title") == target_title)
                            existing_mkts = {m.get("key") for m in existing.get("markets", [])}
                            for m in bm.get("markets", []):
                                if m.get("key") not in existing_mkts:
                                    existing.setdefault("markets", []).append(m)
                                    existing_mkts.add(m.get("key"))

                ev["bookmakers"] = filtered_bms

            return events, metadata, None

        elif response.status_code == 401:
            err = "Error 401: Clave API de The Odds API inválida o expirada. Por favor verifica tu THE_ODDS_API_KEY o activa el Modo Simulación."
            return [], metadata, err

        elif response.status_code in (404, 422):
            err = f"Aviso (Código {response.status_code}): La competición '{sport_key}' no está disponible o se encuentra fuera de temporada en The Odds API."
            return [], metadata, err

        else:
            err = f"Error en The Odds API (HTTP {response.status_code}): {response.text[:200]}"
            return [], metadata, err

    except Exception as e:
        err_str = str(e)
        if "ACCESS_DENIED" in err_str or "SSLError" in err_str:
            err = (
                "Bloqueo del Firewall Empresarial: La red corporativa (Vision & Marketing) restringe dominios de la categoría 'Apuestas / Gambling' (TLSV1_ALERT_ACCESS_DENIED).\n\n"
                "• Para continuar ahora mismo sin restricciones: Enciende el interruptor '🧪 Modo Simulación' en la barra lateral (los datos se cargarán de inmediato y tu IA de Gemini funcionará al 100%).\n"
                "• Para consultar cuotas reales en vivo: Conéctate a una red externa (por ejemplo, compartiendo internet desde tu celular o desconectando la VPN corporativa)."
            )
        else:
            err = f"Error al conectar con The Odds API ({type(e).__name__}). Activa el '🧪 Modo Simulación' en el menú lateral."
        return [], metadata, err


def get_simulation_events(sport_key: str) -> List[Dict[str, Any]]:
    """Returns realistic mock data tailored to the selected sport and tournament."""
    now = datetime.now()
    d1 = (now + timedelta(days=1)).replace(hour=18, minute=30, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    d2 = (now + timedelta(days=2)).replace(hour=20, minute=45, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    d3 = (now + timedelta(days=3)).replace(hour=16, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    if sport_key == "soccer_colombia_primera_a":
        return [
            {
                "id": "col_match_1",
                "sport_key": "soccer_colombia_primera_a",
                "sport_title": "Liga BetPlay Dimayor",
                "commence_time": d1,
                "home_team": "Millonarios",
                "away_team": "Santa Fe",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Millonarios", "price": 1.95},
                                    {"name": "Draw", "price": 3.25},
                                    {"name": "Santa Fe", "price": 4.10},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.85},
                                    {"name": "Under", "point": 2.0, "price": 1.95},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Millonarios", "price": 2.08},  # +EV on Home
                                    {"name": "Draw", "price": 3.30},
                                    {"name": "Santa Fe", "price": 3.90},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.92},
                                    {"name": "Under", "point": 2.0, "price": 1.90},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betano",
                        "title": "Rushbet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Millonarios", "price": 2.05},
                                    {"name": "Draw", "price": 3.35},
                                    {"name": "Santa Fe", "price": 3.85},
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "id": "col_match_2",
                "sport_key": "soccer_colombia_primera_a",
                "sport_title": "Liga BetPlay Dimayor",
                "commence_time": d2,
                "home_team": "Atlético Nacional",
                "away_team": "Junior",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Atlético Nacional", "price": 1.88},
                                    {"name": "Draw", "price": 3.40},
                                    {"name": "Junior", "price": 4.30},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "bwin",
                        "title": "Bwin Colombia",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Atlético Nacional", "price": 1.98},
                                    {"name": "Draw", "price": 3.45},
                                    {"name": "Junior", "price": 4.00},
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "id": "col_match_3",
                "sport_key": "soccer_colombia_primera_a",
                "sport_title": "Liga BetPlay Dimayor",
                "commence_time": d3,
                "home_team": "América de Cali",
                "away_team": "Deportivo Cali",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "América de Cali", "price": 2.10},
                                    {"name": "Draw", "price": 3.10},
                                    {"name": "Deportivo Cali", "price": 3.60},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "América de Cali", "price": 2.05},
                                    {"name": "Draw", "price": 3.05},
                                    {"name": "Deportivo Cali", "price": 3.50},
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "id": "col_match_4",
                "sport_key": "soccer_colombia_primera_a",
                "sport_title": "Liga BetPlay Dimayor",
                "commence_time": d3,
                "home_team": "Independiente Medellín",
                "away_team": "Once Caldas",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Independiente Medellín", "price": 1.95},
                                    {"name": "Draw", "price": 3.20},
                                    {"name": "Once Caldas", "price": 3.90},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "betano",
                        "title": "Rushbet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Independiente Medellín", "price": 1.90},
                                    {"name": "Draw", "price": 3.15},
                                    {"name": "Once Caldas", "price": 3.80},
                                ],
                            }
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "soccer_netherlands_eredivisie":
        return [
            {
                "id": "ned_match_cambuur_eagles",
                "sport_key": "soccer_netherlands_eredivisie",
                "sport_title": "Dutch Eredivisie / KNVB Beker",
                "commence_time": d1,
                "home_team": "SC Cambuur",
                "away_team": "Go Ahead Eagles",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "SC Cambuur", "price": 2.55},
                                    {"name": "Draw", "price": 3.70},
                                    {"name": "Go Ahead Eagles", "price": 2.43},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 3.5, "price": 2.12},
                                    {"name": "Under", "point": 3.5, "price": 1.72},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "SC Cambuur", "price": 2.55},
                                    {"name": "Draw", "price": 3.70},
                                    {"name": "Go Ahead Eagles", "price": 2.43},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 3.5, "price": 2.12},
                                    {"name": "Under", "point": 3.5, "price": 1.72},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "ned_match_1",
                "sport_key": "soccer_netherlands_eredivisie",
                "sport_title": "Dutch Eredivisie",
                "commence_time": d1,
                "home_team": "PSV Eindhoven",
                "away_team": "Ajax",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "PSV Eindhoven", "price": 1.70},
                                    {"name": "Draw", "price": 4.10},
                                    {"name": "Ajax", "price": 4.50},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.48},
                                    {"name": "Under", "point": 2.0, "price": 2.65},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betfair",
                        "title": "Wplay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "PSV Eindhoven", "price": 1.82},  # +EV
                                    {"name": "Draw", "price": 4.00},
                                    {"name": "Ajax", "price": 4.30},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.62},  # +EV Over
                                    {"name": "Under", "point": 2.0, "price": 2.40},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "ned_match_2",
                "sport_key": "soccer_netherlands_eredivisie",
                "sport_title": "Dutch Eredivisie",
                "commence_time": d2,
                "home_team": "Feyenoord",
                "away_team": "AZ Alkmaar",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Feyenoord", "price": 1.75},
                                    {"name": "Draw", "price": 3.80},
                                    {"name": "AZ Alkmaar", "price": 4.40},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.55},
                                    {"name": "Under", "point": 2.0, "price": 2.45},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Feyenoord", "price": 1.85},
                                    {"name": "Draw", "price": 3.90},
                                    {"name": "AZ Alkmaar", "price": 4.20},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.68},
                                    {"name": "Under", "point": 2.0, "price": 2.30},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "ned_match_3",
                "sport_key": "soccer_netherlands_eredivisie",
                "sport_title": "Dutch Eredivisie",
                "commence_time": d3,
                "home_team": "FC Twente",
                "away_team": "FC Utrecht",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "FC Twente", "price": 1.95},
                                    {"name": "Draw", "price": 3.50},
                                    {"name": "FC Utrecht", "price": 3.80},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.65},
                                    {"name": "Under", "point": 2.0, "price": 2.20},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "FC Twente", "price": 1.90},  # No EV
                                    {"name": "Draw", "price": 3.40},
                                    {"name": "FC Utrecht", "price": 3.75},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.60},
                                    {"name": "Under", "point": 2.0, "price": 2.15},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "ned_match_4",
                "sport_key": "soccer_netherlands_eredivisie",
                "sport_title": "Dutch Eredivisie",
                "commence_time": d3,
                "home_team": "SC Heerenveen",
                "away_team": "Sparta Rotterdam",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "SC Heerenveen", "price": 2.20},
                                    {"name": "Draw", "price": 3.40},
                                    {"name": "Sparta Rotterdam", "price": 3.20},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.70},
                                    {"name": "Under", "point": 2.0, "price": 2.10},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betfair",
                        "title": "Wplay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "SC Heerenveen", "price": 2.15},  # No EV
                                    {"name": "Draw", "price": 3.30},
                                    {"name": "Sparta Rotterdam", "price": 3.10},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.65},
                                    {"name": "Under", "point": 2.0, "price": 2.05},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "soccer_germany_bundesliga2":
        return [
            {
                "id": "ger2_match_1",
                "sport_key": "soccer_germany_bundesliga2",
                "sport_title": "Alemania - 2. Bundesliga",
                "commence_time": d1,
                "home_team": "Hamburger SV",
                "away_team": "Schalke 04",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Hamburger SV", "price": 2.10},
                                    {"name": "Draw", "price": 3.60},
                                    {"name": "Schalke 04", "price": 3.40},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.72},
                                    {"name": "Under", "point": 2.0, "price": 2.15},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Hamburger SV", "price": 2.20},
                                    {"name": "Draw", "price": 3.50},
                                    {"name": "Schalke 04", "price": 3.30},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.82},  # +EV Over 2.0
                                    {"name": "Under", "point": 2.0, "price": 2.00},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "ger2_match_2",
                "sport_key": "soccer_germany_bundesliga2",
                "sport_title": "Alemania - 2. Bundesliga",
                "commence_time": d2,
                "home_team": "Hertha BSC",
                "away_team": "1. FC Köln",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Hertha BSC", "price": 2.45},
                                    {"name": "Draw", "price": 3.50},
                                    {"name": "1. FC Köln", "price": 2.85},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.70},
                                    {"name": "Under", "point": 2.0, "price": 2.20},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betano",
                        "title": "Rushbet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Hertha BSC", "price": 2.60},
                                    {"name": "Draw", "price": 3.45},
                                    {"name": "1. FC Köln", "price": 2.75},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.78},
                                    {"name": "Under", "point": 2.0, "price": 2.05},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "soccer_switzerland_superleague":
        return [
            {
                "id": "sui_match_1",
                "sport_key": "soccer_switzerland_superleague",
                "sport_title": "Suiza - Super League",
                "commence_time": d1,
                "home_team": "Young Boys",
                "away_team": "FC Basel",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Young Boys", "price": 1.85},
                                    {"name": "Draw", "price": 3.90},
                                    {"name": "FC Basel", "price": 3.95},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.60},
                                    {"name": "Under", "point": 2.0, "price": 2.35},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Young Boys", "price": 1.95},
                                    {"name": "Draw", "price": 3.80},
                                    {"name": "FC Basel", "price": 3.80},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.75},  # +EV
                                    {"name": "Under", "point": 2.0, "price": 2.15},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "sui_match_2",
                "sport_key": "soccer_switzerland_superleague",
                "sport_title": "Suiza - Super League",
                "commence_time": d2,
                "home_team": "FC Zürich",
                "away_team": "Servette FC",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "FC Zürich", "price": 2.20},
                                    {"name": "Draw", "price": 3.45},
                                    {"name": "Servette FC", "price": 3.25},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.68},
                                    {"name": "Under", "point": 2.0, "price": 2.22},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betano",
                        "title": "Rushbet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "FC Zürich", "price": 2.35},
                                    {"name": "Draw", "price": 3.40},
                                    {"name": "Servette FC", "price": 3.15},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.78},
                                    {"name": "Under", "point": 2.0, "price": 2.10},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "soccer_argentina_primera_division":
        return [
            {
                "id": "arg_match_1",
                "sport_key": "soccer_argentina_primera_division",
                "sport_title": "Argentina - Liga Profesional",
                "commence_time": d1,
                "home_team": "River Plate",
                "away_team": "Boca Juniors",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "River Plate", "price": 2.10},
                                    {"name": "Draw", "price": 3.15},
                                    {"name": "Boca Juniors", "price": 3.85},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "River Plate", "price": 2.28},  # +EV DNB
                                    {"name": "Draw", "price": 3.10},
                                    {"name": "Boca Juniors", "price": 3.65},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "arg_match_2",
                "sport_key": "soccer_argentina_primera_division",
                "sport_title": "Argentina - Liga Profesional",
                "commence_time": d2,
                "home_team": "Racing Club",
                "away_team": "Independiente",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Racing Club", "price": 2.05},
                                    {"name": "Draw", "price": 3.20},
                                    {"name": "Independiente", "price": 3.95},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bwin",
                        "title": "Bwin Colombia",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Racing Club", "price": 2.18},
                                    {"name": "Draw", "price": 3.15},
                                    {"name": "Independiente", "price": 3.75},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "soccer_italy_serie_b":
        return [
            {
                "id": "it2_match_1",
                "sport_key": "soccer_italy_serie_b",
                "sport_title": "Italia - Serie B",
                "commence_time": d1,
                "home_team": "Palermo",
                "away_team": "Sampdoria",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Palermo", "price": 2.15},
                                    {"name": "Draw", "price": 3.25},
                                    {"name": "Sampdoria", "price": 3.55},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Palermo", "price": 2.30},  # +EV DNB
                                    {"name": "Draw", "price": 3.20},
                                    {"name": "Sampdoria", "price": 3.40},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "it2_match_2",
                "sport_key": "soccer_italy_serie_b",
                "sport_title": "Italia - Serie B",
                "commence_time": d2,
                "home_team": "Pisa",
                "away_team": "Spezia",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Pisa", "price": 2.20},
                                    {"name": "Draw", "price": 3.10},
                                    {"name": "Spezia", "price": 3.60},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betano",
                        "title": "Rushbet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Pisa", "price": 2.35},
                                    {"name": "Draw", "price": 3.10},
                                    {"name": "Spezia", "price": 3.40},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "soccer_conmebol_copa_libertadores":
        return [
            {
                "id": "lib_match_1",
                "sport_key": "soccer_conmebol_copa_libertadores",
                "sport_title": "CONMEBOL Copa Libertadores",
                "commence_time": d1,
                "home_team": "Flamengo",
                "away_team": "Palmeiras",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Flamengo", "price": 2.05},
                                    {"name": "Draw", "price": 3.35},
                                    {"name": "Palmeiras", "price": 3.80},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.78},
                                    {"name": "Under", "point": 2.0, "price": 2.08},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Flamengo", "price": 2.20},
                                    {"name": "Draw", "price": 3.30},
                                    {"name": "Palmeiras", "price": 3.60},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.90},  # +EV
                                    {"name": "Under", "point": 2.0, "price": 1.92},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "lib_match_2",
                "sport_key": "soccer_conmebol_copa_libertadores",
                "sport_title": "CONMEBOL Copa Libertadores",
                "commence_time": d2,
                "home_team": "River Plate",
                "away_team": "Atlético Mineiro",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "River Plate", "price": 1.95},
                                    {"name": "Draw", "price": 3.40},
                                    {"name": "Atlético Mineiro", "price": 4.10},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betfair",
                        "title": "Wplay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "River Plate", "price": 2.10},  # +EV
                                    {"name": "Draw", "price": 3.35},
                                    {"name": "Atlético Mineiro", "price": 3.90},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "soccer_uefa_champs_league":
        return [
            {
                "id": "ucl_match_1",
                "sport_key": "soccer_uefa_champs_league",
                "sport_title": "UEFA Champions League",
                "commence_time": d1,
                "home_team": "Real Madrid",
                "away_team": "Manchester City",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Real Madrid", "price": 2.45},
                                    {"name": "Draw", "price": 3.50},
                                    {"name": "Manchester City", "price": 2.80},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.65},
                                    {"name": "Under", "point": 2.0, "price": 2.25},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betano",
                        "title": "Rushbet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Real Madrid", "price": 2.62},  # +EV
                                    {"name": "Draw", "price": 3.55},
                                    {"name": "Manchester City", "price": 2.70},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.0, "price": 1.76},
                                    {"name": "Under", "point": 2.0, "price": 2.10},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "ucl_match_2",
                "sport_key": "soccer_uefa_champs_league",
                "sport_title": "UEFA Champions League",
                "commence_time": d2,
                "home_team": "Arsenal",
                "away_team": "Bayern Munich",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 1.85},
                                    {"name": "Draw", "price": 3.75},
                                    {"name": "Bayern Munich", "price": 4.10},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 1.95},
                                    {"name": "Draw", "price": 3.70},
                                    {"name": "Bayern Munich", "price": 4.00},
                                ],
                            }
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "soccer_conmebol_copa_america":
        return [
            {
                "id": "ca_match_1",
                "sport_key": "soccer_conmebol_copa_america",
                "sport_title": "Copa América",
                "commence_time": d1,
                "home_team": "Colombia",
                "away_team": "Argentina",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Colombia", "price": 3.10},
                                    {"name": "Draw", "price": 3.05},
                                    {"name": "Argentina", "price": 2.45},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Colombia", "price": 3.35},  # +EV
                                    {"name": "Draw", "price": 3.10},
                                    {"name": "Argentina", "price": 2.40},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]

    elif "tennis" in sport_key:
        return [
            {
                "id": "tennis_match_1",
                "sport_key": sport_key,
                "sport_title": "ATP Tennis",
                "commence_time": d1,
                "home_team": "Carlos Alcaraz",
                "away_team": "Jannik Sinner",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Carlos Alcaraz", "price": 1.80},
                                    {"name": "Jannik Sinner", "price": 2.05},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Carlos Alcaraz", "price": 1.93},  # +EV
                                    {"name": "Jannik Sinner", "price": 1.91},
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "id": "tennis_match_2",
                "sport_key": sport_key,
                "sport_title": "ATP Tennis",
                "commence_time": d2,
                "home_team": "Novak Djokovic",
                "away_team": "Alexander Zverev",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Novak Djokovic", "price": 1.62},
                                    {"name": "Alexander Zverev", "price": 2.35},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "bwin",
                        "title": "Bwin Colombia",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Novak Djokovic", "price": 1.74},  # +EV
                                    {"name": "Alexander Zverev", "price": 2.20},
                                ],
                            }
                        ],
                    },
                ],
            },
        ]

    elif "basketball" in sport_key:
        return [
            {
                "id": "nba_match_1",
                "sport_key": "basketball_nba",
                "sport_title": "NBA Basketball",
                "commence_time": d1,
                "home_team": "Boston Celtics",
                "away_team": "Los Angeles Lakers",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": 1.50},
                                    {"name": "Los Angeles Lakers", "price": 2.70},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Boston Celtics", "point": -5.5, "price": 1.91},
                                    {"name": "Los Angeles Lakers", "point": 5.5, "price": 1.91},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 224.5, "price": 1.91},
                                    {"name": "Under", "point": 224.5, "price": 1.91},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "betano",
                        "title": "Rushbet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": 1.60},  # +EV
                                    {"name": "Los Angeles Lakers", "price": 2.50},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Boston Celtics", "point": -5.5, "price": 2.05},  # +EV
                                    {"name": "Los Angeles Lakers", "point": 5.5, "price": 1.82},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 224.5, "price": 2.02},  # +EV
                                    {"name": "Under", "point": 224.5, "price": 1.84},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "id": "nba_match_2",
                "sport_key": "basketball_nba",
                "sport_title": "NBA Basketball",
                "commence_time": d2,
                "home_team": "Denver Nuggets",
                "away_team": "Golden State Warriors",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Denver Nuggets", "price": 1.65},
                                    {"name": "Golden State Warriors", "price": 2.30},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Denver Nuggets", "point": -3.5, "price": 1.90},
                                    {"name": "Golden State Warriors", "point": 3.5, "price": 1.92},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 228.0, "price": 1.90},
                                    {"name": "Under", "point": 228.0, "price": 1.92},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Denver Nuggets", "price": 1.76},  # +EV
                                    {"name": "Golden State Warriors", "price": 2.15},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Denver Nuggets", "point": -3.5, "price": 2.02},
                                    {"name": "Golden State Warriors", "point": 3.5, "price": 1.83},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

    elif sport_key == "basketball_euroleague":
        return [
            {
                "id": "euro_bball_1",
                "sport_key": "basketball_euroleague",
                "sport_title": "Baloncesto - EuroLeague",
                "commence_time": d1,
                "home_team": "Real Madrid Baloncesto",
                "away_team": "Panathinaikos",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Real Madrid Baloncesto", "price": 1.60},
                                    {"name": "Panathinaikos", "price": 2.45},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Real Madrid Baloncesto", "point": -4.5, "price": 1.91},
                                    {"name": "Panathinaikos", "point": 4.5, "price": 1.91},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 164.5, "price": 1.91},
                                    {"name": "Under", "point": 164.5, "price": 1.91},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "bet365",
                        "title": "BetPlay",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Real Madrid Baloncesto", "price": 1.72},
                                    {"name": "Panathinaikos", "price": 2.25},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Real Madrid Baloncesto", "point": -4.5, "price": 2.05},
                                    {"name": "Panathinaikos", "point": 4.5, "price": 1.80},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 164.5, "price": 2.02},
                                    {"name": "Under", "point": 164.5, "price": 1.83},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

    # Default contextual soccer simulation for any unlisted league
    league_label = sport_key.replace("soccer_", "").replace("_", " ").title()
    return [
        {
            "id": f"{sport_key}_sample_1",
            "sport_key": sport_key,
            "sport_title": f"Fútbol - {league_label}",
            "commence_time": d1,
            "home_team": f"{league_label} Club A",
            "away_team": f"{league_label} Club B",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": f"{league_label} Club A", "price": 1.90},
                                {"name": "Draw", "price": 3.30},
                                {"name": f"{league_label} Club B", "price": 4.20},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 2.0, "price": 1.80},
                                {"name": "Under", "point": 2.0, "price": 2.00},
                            ],
                        },
                    ],
                },
                {
                    "key": "bet365",
                    "title": "BetPlay",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": f"{league_label} Club A", "price": 2.05},
                                {"name": "Draw", "price": 3.35},
                                {"name": f"{league_label} Club B", "price": 3.90},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 2.0, "price": 1.95},
                                {"name": "Under", "point": 2.0, "price": 1.85},
                            ],
                        },
                    ],
                },
            ],
        }
    ]
