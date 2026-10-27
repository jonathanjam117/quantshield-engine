"""QuantShield Multi-Sport +EV & Low-Variance Betting Engine.

Web application in Streamlit + SQLite for value betting detection,
mathematical risk management, bankroll accounting, and Telegram alerts.

Security Notes:
- API keys are read from st.secrets (Streamlit Cloud) with fallback to os.getenv (local).
- All SQL queries use parameterized statements (no SQL injection risk).
- HTML templates use only static variables from internal model outputs (no raw user input).
- XSRF protection is enabled in .streamlit/config.toml.
"""
from __future__ import annotations

import html
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


def _get_secret(key: str, default: str = "") -> str:
    """Reads a secret from st.secrets (Streamlit Cloud) with fallback to os.getenv (local .env).

    Priority: st.secrets → os.getenv → default
    This dual-source pattern makes the app compatible with both local development
    and Streamlit Community Cloud without code changes.
    """
    try:
        val = st.secrets.get(key, None)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


def _sanitize(text: str) -> str:
    """HTML-escapes user-supplied text before embedding in HTML templates.
    Prevents XSS injection from any user-controlled input field.
    """
    return html.escape(str(text), quote=True)


def format_to_local_time(iso_utc_str: str, tz_offset_hours: int = -5) -> str:
    """Convierte un timestamp ISO UTC (ej. 2026-10-20T18:30:00Z) a fecha y hora local legible (UTC-5 / Colombia / Perú)."""
    if not iso_utc_str:
        return "Horario por definir"
    try:
        clean_str = iso_utc_str.replace("Z", "")
        if "T" in clean_str:
            dt = datetime.fromisoformat(clean_str)
        else:
            dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
        dt_local = dt + timedelta(hours=tz_offset_hours)
        dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        meses = ["", "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        dia_nom = dias[dt_local.weekday()]
        mes_nom = meses[dt_local.month]
        hora_str = dt_local.strftime("%I:%M %p")
        return f"{dia_nom} {dt_local.day:02d}/{mes_nom}/{dt_local.year} a las {hora_str} (Hora Local UTC-5)"
    except Exception:
        return iso_utc_str


def format_short_local_time(iso_utc_str: str, tz_offset_hours: int = -5) -> str:
    """Versión concisa para tablas compactas (ej. 21/10 01:30 PM)."""
    if not iso_utc_str:
        return "Por definir"
    try:
        clean_str = iso_utc_str.replace("Z", "")
        if "T" in clean_str:
            dt = datetime.fromisoformat(clean_str)
        else:
            dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
        dt_local = dt + timedelta(hours=tz_offset_hours)
        return dt_local.strftime("%d/%m %I:%M %p")
    except Exception:
        return iso_utc_str


def format_money(amount: float, curr_code: str = "COP") -> str:
    """Formatea valores monetarios con separadores de miles y divisa legible."""
    if curr_code == "COP":
        return f"${amount:,.0f} COP"
    elif curr_code == "EUR":
        return f"€{amount:,.2f}"
    else:
        return f"${amount:,.2f} USD"


def get_ai_field(ai_obj: Any, field: str, default: Any = "") -> Any:
    """Helper universal de extracción segura para campos de análisis táctico / IA.
    Maneja con total seguridad: None, diccionarios (dict), objetos Pydantic y clases genéricas.
    """
    if ai_obj is None:
        return default

    alias_map = {
        "nivel_riesgo": ["riesgo", "risk", "nivel_riesgo"],
        "riesgo": ["nivel_riesgo", "risk", "riesgo"],
        "justificacion_tactica": ["justificacion", "justificacion_tactica", "justification"],
        "justificacion": ["justificacion_tactica", "justificacion", "justification"],
        "confianza_ia": ["confianza", "confianza_ia", "confidence"],
        "confianza": ["confianza_ia", "confianza", "confidence"],
        "factor_ajuste_local": ["factor_local", "factor_ajuste_local", "factor_h"],
        "factor_local": ["factor_ajuste_local", "factor_local", "factor_h"],
        "factor_ajuste_visita": ["factor_visita", "factor_ajuste_visita", "factor_a"],
        "factor_visita": ["factor_ajuste_visita", "factor_visita", "factor_a"],
    }
    candidates = alias_map.get(field, [field])

    if isinstance(ai_obj, dict):
        for c in candidates:
            if c in ai_obj and ai_obj[c] is not None:
                return ai_obj[c]
        return default

    for c in candidates:
        if hasattr(ai_obj, c):
            val = getattr(ai_obj, c, None)
            if val is not None:
                return val

    return default


# Ensure quantshield_engine directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load environment variables (local dev only; Streamlit Cloud uses st.secrets)
dotenv.load_dotenv(os.path.join(BASE_DIR, ".env"))

from database.db_manager import (
    init_db,
    add_bet,
    get_all_bets,
    update_bet_status,
    delete_bet,
    get_bankroll_metrics,
    get_bankroll_settings,
    set_initial_bankroll,
)
from etl.odds_api_client import fetch_odds, get_available_sports
from models.soccer_engine import analyze_soccer_match
from models.tennis_engine import analyze_tennis_match
from models.nba_engine import analyze_nba_match
from models.betplay_markets import (
    calculate_betplay_security_catalog,
    generate_betplay_same_game_parlays,
)
from models.parlay_engine import (
    calculate_parlay,
    find_best_auto_parlay,
    evaluate_corners_poisson,
    calculate_poisson_advanced_markets,
)
from models.ai_context_agent import (
    TacticalAdjustment,
    get_ai_field,
    evaluar_contexto_partido,
    verificar_conexion_gemini,
    GEMINI_MODELS,
)
from models.backtest_simulator import run_historical_backtest, load_historical_matches, AVAILABLE_BACKTEST_LEAGUES

# Page Configuration
st.set_page_config(
    page_title="QuantShield Engine | +EV & Low-Variance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ============================================================
# PREMIUM MOBILE-FIRST CSS — QuantShield v2
# ============================================================
st.markdown(
    """
    <style>
    /* === BASE MOBILE VIEWPORT === */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem 0.7rem 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
            overflow-x: auto;
            flex-wrap: nowrap;
            -webkit-overflow-scrolling: touch;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 7px 9px;
            font-size: 0.72rem;
            white-space: nowrap;
        }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1rem !important; }
    }

    /* === PREMIUM CARDS === */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 14px 16px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.35);
    }

    /* === EV BADGE === */
    .badge-ev {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.85em;
        letter-spacing: 0.4px;
        box-shadow: 0 2px 8px rgba(16,185,129,0.45);
        display: inline-block;
    }

    /* === DNB BADGE === */
    .badge-dnb {
        background: linear-gradient(90deg, #6366f1, #4f46e5);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(99,102,241,0.4);
        display: inline-block;
    }

    /* === ASIAN BADGE === */
    .badge-asian {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(245,158,11,0.4);
        display: inline-block;
    }

    /* === AI BADGE === */
    .badge-ai {
        background: linear-gradient(90deg, #7c3aed, #6d28d9);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78em;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(124,58,237,0.4);
        display: inline-block;
    }

    /* === OPPORTUNITY CARDS === */
    .opp-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 18px rgba(16,185,129,0.1);
        transition: box-shadow 0.2s ease;
    }
    .opp-card:hover {
        box-shadow: 0 6px 24px rgba(16,185,129,0.2);
    }

    /* === BUTTONS === */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.3px;
        transition: all 0.18s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(59,130,246,0.3);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* === SIDEBAR PREMIUM === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* === TAB STYLING === */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.12);
        border-bottom: 3px solid #3b82f6;
    }

    /* === PULSE ANIMATION (LIVE INDICATOR) === */
    @keyframes pulse-green {
        0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.45); }
        70%  { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .live-indicator { animation: pulse-green 2s infinite; }

    /* === HIDE STREAMLIT BRANDING (PRESERVING SIDEBAR TOGGLE ON MOBILE) === */
    #MainMenu  { display: none !important; }
    footer     { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }

    /* Ensure header remains functional for mobile sidebar navigation */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Make mobile sidebar button prominent on mobile devices */
    @media (max-width: 768px) {
        div[data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            background-color: #1e293b !important;
            border-radius: 8px !important;
            border: 1.5px solid #3b82f6 !important;
            box-shadow: 0 2px 10px rgba(59, 130, 246, 0.5) !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_odds(
    sport_key: str,
    api_key: Optional[str],
    is_demo: bool,
    force_nonce: int = 0,
):
    """Cached wrapper around The Odds API to strictly protect the 500 requests/month limit.
    Cached with TTL of 1 hour (3600 seconds) as required in Section 4.1.
    """
    return fetch_odds(
        sport_key=sport_key,
        api_key=api_key,
        is_demo=is_demo,
    )


def initialize_app():
    """Initializes local SQLite database automatically on startup."""
    init_db()


initialize_app()

# ==============================================================================
# SIDEBAR: CONFIGURATION & PARAMETERS
# ==============================================================================
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
        <span style="font-size: 2.2rem;">🛡️</span>
        <span style="font-size: 1.6rem; font-weight: 700; color: #3b82f6;">QuantShield</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Motores Cuantitativos +EV & Baja Varianza")

# Bankroll settings in database
curr_settings = get_bankroll_settings()
default_bankroll = curr_settings.get("initial_bankroll", 1000.0)

# Read secrets: st.secrets (Streamlit Cloud) with fallback to os.getenv (local .env)
env_api_key = _get_secret("THE_ODDS_API_KEY")
env_gemini_key = _get_secret("GEMINI_API_KEY")
env_tg_token = _get_secret("TELEGRAM_BOT_TOKEN")
env_tg_chat_id = _get_secret("TELEGRAM_CHAT_ID")

# 1. API Configuration & Demo Toggle
st.sidebar.markdown("### 🔑 Credenciales & Fuente de Datos")
has_env_key = bool(env_api_key and env_api_key.strip() != "your_the_odds_api_key_here")

# Simulation mode toggle: ON by default if no valid API key
demo_mode_toggle = st.sidebar.toggle(
    "🧪 Modo Simulación (Sin gastar API)",
    value=True if not has_env_key else False,
    help="Usa datos realistas de prueba para probar todos los algoritmos sin consumir tu cuota mensual de The Odds API.",
)

api_key_input = st.sidebar.text_input(
    "The Odds API Key",
    value=env_api_key if not demo_mode_toggle else "",
    type="password",
    help="Obtén tu clave gratuita de 500 llamadas/mes en the-odds-api.com",
    disabled=demo_mode_toggle,
)

active_api_key = api_key_input.strip() if not demo_mode_toggle else ""

# 2. Motor de IA (Gemini Pro)
st.sidebar.markdown("### 🧠 Motor de IA (Gemini Pro)")
gemini_api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value=env_gemini_key if env_gemini_key != "your_gemini_api_key_here" else "",
    type="password",
    help="Pega aquí la clave de Google AI Studio (comienza por AIzaSy...).",
)

st.sidebar.markdown(
    "[👉 Obtener API Key gratuita en Google AI Studio](https://aistudio.google.com/app/apikey)"
)
st.sidebar.caption("💡 *Si la página de Google se queda en blanco o no carga, abre el enlace en ventana de incógnito con tu cuenta personal @gmail.com.*")

selected_gemini_model = st.sidebar.selectbox(
    "Modelo Gemini",
    options=GEMINI_MODELS,
    index=0,
    help="gemini-2.0-flash: Modelo recomendado oficial (1,500 consultas gratis/día) | gemini-1.5-flash: Máxima estabilidad | gemini-2.0-flash-lite: Ultra rápido.",
)

col_ai_btn1, col_ai_btn2 = st.sidebar.columns([1.1, 0.9])
with col_ai_btn1:
    test_gemini_clicked = st.button("🧪 Probar Clave", use_container_width=True, help="Envía un ping a Google Gemini para verificar tu clave")
with col_ai_btn2:
    if gemini_api_key_input and gemini_api_key_input.strip():
        st.markdown("<div style='margin-top: 6px; text-align: center;'><span style='background-color: #059669; color: white; padding: 4px 6px; border-radius: 4px; font-size: 0.78em; font-weight: 700;'>🟢 Clave Lista</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top: 6px; text-align: center;'><span style='background-color: #4b5563; color: white; padding: 4px 6px; border-radius: 4px; font-size: 0.78em; font-weight: 600;'>⚪ Simulado</span></div>", unsafe_allow_html=True)

if test_gemini_clicked:
    if not gemini_api_key_input or not gemini_api_key_input.strip():
        st.sidebar.warning("⚠️ Pega primero tu clave de Gemini en el campo superior.")
    else:
        with st.sidebar.status("Conectando con Google Gemini...", expanded=True) as status:
            res = verificar_conexion_gemini(gemini_api_key_input.strip(), selected_gemini_model)
            if res["success"]:
                status.update(label="✅ ¡Conexión con Gemini Exitosa!", state="complete", expanded=False)
                st.sidebar.success(f"{res['message']}")
            else:
                status.update(label="❌ Error de Conexión", state="error", expanded=True)
                st.sidebar.error(res["message"])
                if res.get("error_detail"):
                    st.sidebar.caption(f"Detalle técnico: `{res['error_detail'][:150]}`")

ai_tactical_enabled = st.sidebar.toggle(
    "Activar Ajuste Táctico con IA",
    value=True,
    help="Ajusta cuantitativamente los parámetros matemáticos evaluando lesiones, rotaciones, fatiga y clima.",
)

# 2. Capital Management & Mathematical Constraints
st.sidebar.markdown("### ⚖️ Gestión de Capital & Kelly")

currency_options = ["COP (Pesos Colombianos)", "USD (Dólares)", "EUR (Euros)"]
saved_currency = curr_settings.get("currency", "USD")
curr_idx = 0 if saved_currency == "COP" else (2 if saved_currency == "EUR" else 1)

selected_currency_label = st.sidebar.selectbox(
    "Divisa / Moneda",
    options=currency_options,
    index=curr_idx,
    help="Define la moneda de tu cuenta de apuestas. Los algoritmos de Kelly calcularán los stakes proporcionalmente."
)

selected_curr_code = "COP" if "COP" in selected_currency_label else ("EUR" if "EUR" in selected_currency_label else "USD")

if selected_curr_code == "COP":
    min_b = 20000.0
    max_b = 500000000.0
    step_b = 50000.0
    init_val = float(default_bankroll) if default_bankroll >= 20000.0 else 1000000.0
else:
    min_b = 20.0
    max_b = 2000000.0
    step_b = 50.0
    init_val = float(default_bankroll) if default_bankroll <= 2000000.0 else 1000.0

bankroll_val = st.sidebar.number_input(
    f"Bankroll Total ({selected_curr_code})",
    min_value=min_b,
    max_value=max_b,
    value=init_val,
    step=step_b,
    format="%.0f" if selected_curr_code == "COP" else "%.2f",
    help="Capital total asignado a tu cuenta de apuestas.",
)

if bankroll_val != default_bankroll or selected_curr_code != saved_currency:
    set_initial_bankroll(bankroll_val, currency=selected_curr_code)

kelly_multiplier = st.sidebar.slider(
    "Fracción de Kelly (k)",
    min_value=0.05,
    max_value=0.50,
    value=0.20,
    step=0.05,
    help="0.20 representa Fifth Kelly (quinto de Kelly), estándar para minimizar el riesgo de ruina.",
)

min_ev_filter = st.sidebar.slider(
    "Filtrar EV Mínimo (%)",
    min_value=0.5,
    max_value=12.0,
    value=3.0,
    step=0.5,
    help="Solo se mostrarán operaciones con Esperanza Matemática Positiva superior o igual a este valor.",
)

odds_range = st.sidebar.slider(
    "Rango Eficiente de Cuotas",
    min_value=1.20,
    max_value=3.50,
    value=(1.40, 2.25),
    step=0.05,
    help="Rango óptimo de cuotas para maximizar valor y reducir la racha de varianza negativa.",
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"🔒 **Regla de Oro**: Tope máximo por operación fijado en **3.5% del Bankroll** ({format_money(round(bankroll_val * 0.035, 2), selected_curr_code)})."
)


# ==============================================================================
# GLOBAL DATA & TOURNAMENT LOADING
# ==============================================================================
sports_catalog = get_available_sports()
col_cat, col_tourn, col_btn = st.columns([1.2, 1.8, 1])

with col_cat:
    category_selected = st.selectbox("Categoría Deportiva", list(sports_catalog.keys()))

tournament_dict = sports_catalog[category_selected]
with col_tourn:
    tournament_key = st.selectbox(
        "Competición Objetivo",
        options=list(tournament_dict.keys()),
        format_func=lambda k: tournament_dict[k]["name"],
    )

with col_btn:
    st.write("")
    st.write("")
    force_refresh = st.button("🔄 Refrescar Escáner", use_container_width=True)

# Cache Nonce management
if "cache_nonce" not in st.session_state:
    st.session_state.cache_nonce = 0
if force_refresh:
    st.session_state.cache_nonce += 1
    st.cache_data.clear()

# Fetch data
selected_meta = tournament_dict[tournament_key]
selected_meta["sport_key"] = tournament_key

with st.spinner("Consultando cuotas y ejecutando modelos matemáticos..."):
    events, meta, error_msg = get_cached_odds(
        sport_key=tournament_key,
        api_key=active_api_key,
        is_demo=demo_mode_toggle,
        force_nonce=st.session_state.cache_nonce,
    )

# Display Quota information in sidebar
st.sidebar.markdown("### 📊 Cuota de The Odds API")
st.sidebar.metric(
    "Peticiones Restantes (Mes)",
    str(meta.get("requests_remaining", "N/A")),
    help="The Odds API ofrece 500 peticiones mensuales en el plan gratuito.",
)

if not demo_mode_toggle and not active_api_key:
    st.warning(
        "🔑 **Estás en Modo Real pero la casilla 'The Odds API Key' está vacía:**\n\n"
        "Para conectarte a internet y descargar las cuotas reales de BetPlay y del mundo, necesitas tu clave gratuita de acceso:\n\n"
        "1. Obtén tu clave gratis en 1 minuto en [the-odds-api.com](https://the-odds-api.com) (500 consultas gratis al mes, sin tarjetas ni pagos).\n"
        "2. Pégala en la casilla **'The Odds API Key'** en la barra lateral izquierda.\n"
        "3. Pulsa el botón **🔄 Refrescar Escáner**.\n\n"
        "*(Si deseas seguir explorando sin clave, simplemente vuelve a encender el interruptor **🧪 Modo Simulación** en la barra lateral).*",
        icon="ℹ️"
    )
elif error_msg:
    st.warning(f"⚠️ {error_msg}")

# Process events with mathematical models and Gemini AI context
detected_opportunities: List[Dict[str, Any]] = []
all_fixture_rows: List[Dict[str, Any]] = []

# Initialize session state for AI adjustments and breaking news
if "match_ai_adjustments" not in st.session_state:
    st.session_state.match_ai_adjustments = {}
if "match_user_notes" not in st.session_state:
    st.session_state.match_user_notes = {}

sport_type = selected_meta.get("type", "soccer")
for event in events:
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    match_key = f"{home_team} vs {away_team}"

    # Extract 1X2 odds for fixture overview prioritizing Colombian bookmakers
    h_price, x_price, a_price, ref_bm = "-", "-", "-", "BetPlay"
    bms = event.get("bookmakers", [])
    # Prefer Colombian recreational house over Pinnacle for the fixture table
    rec_bms = [b for b in bms if b.get("key", "").lower() != "pinnacle"]
    target_bm = rec_bms[0] if rec_bms else (bms[0] if bms else None)

    if target_bm:
        for m in target_bm.get("markets", []):
            if m.get("key") == "h2h":
                for o in m.get("outcomes", []):
                    if o.get("name") == home_team:
                        h_price = f"{o.get('price', 0):.2f}"
                    elif o.get("name") in ("Draw", "Empate", "X"):
                        x_price = f"{o.get('price', 0):.2f}"
                    elif o.get("name") == away_team:
                        a_price = f"{o.get('price', 0):.2f}"
                bm_t = target_bm.get("title", target_bm.get("key", "Ref"))
                ref_bm = bm_t if target_bm.get("key", "").lower() != "pinnacle" else f"{bm_t} (Ref)"
                break

    # Evaluate Gemini AI tactical context if enabled
    ai_adj = None
    if ai_tactical_enabled:
        if match_key in st.session_state.match_ai_adjustments:
            ai_adj = st.session_state.match_ai_adjustments[match_key]
        else:
            odds_snapshot = {}
            bms = event.get("bookmakers", [])
            if bms:
                for m in bms[0].get("markets", []):
                    if m.get("key") == "h2h":
                        for o in m.get("outcomes", []):
                            odds_snapshot[o.get("name", "")] = o.get("price", 0.0)

            user_notes = st.session_state.match_user_notes.get(match_key, "")
            ai_adj = evaluar_contexto_partido(
                deporte=category_selected,
                torneo=selected_meta.get("name", ""),
                local=home_team,
                visita=away_team,
                cuotas_actuales=odds_snapshot,
                contexto_usuario=user_notes,
                api_key=gemini_api_key_input.strip() if gemini_api_key_input else None,
                model_name=selected_gemini_model,
                is_demo=False,
            )
            st.session_state.match_ai_adjustments[match_key] = ai_adj

    event_opps: List[Dict[str, Any]] = []
    if sport_type in ("goals", "dnb", "intl"):
        event_opps = analyze_soccer_match(
            event=event,
            sport_meta=selected_meta,
            bankroll=bankroll_val,
            kelly_k=kelly_multiplier,
            min_ev=min_ev_filter,
            min_odds=odds_range[0],
            max_odds=odds_range[1],
            ai_adjustment=ai_adj,
        )
        detected_opportunities.extend(event_opps)

    elif sport_type == "tennis":
        event_opps = analyze_tennis_match(
            event=event,
            sport_meta=selected_meta,
            bankroll=bankroll_val,
            kelly_k=kelly_multiplier,
            min_ev=min_ev_filter,
            min_odds=odds_range[0],
            max_odds=odds_range[1],
            ai_adjustment=ai_adj,
        )
        detected_opportunities.extend(event_opps)

    elif sport_type == "nba":
        event_opps = analyze_nba_match(
            event=event,
            sport_meta=selected_meta,
            bankroll=bankroll_val,
            kelly_k=kelly_multiplier,
            min_ev=min_ev_filter,
            min_odds=odds_range[0],
            max_odds=odds_range[1],
            ai_adjustment=ai_adj,
        )
        detected_opportunities.extend(event_opps)

    if event_opps:
        best_ev = max(event_opps, key=lambda x: x["ev_percent"])
        verdict = f"🟢 +EV: {best_ev['market']} (+{best_ev['ev_percent']:.1f}%)"
        has_val = True
    else:
        verdict = "⚪ Sin Ventaja (Margen Casa)"
        has_val = False

    all_fixture_rows.append({
        "Fecha/Hora (Local)": format_short_local_time(event.get("commence_time", "")),
        "Partido": match_key,
        "1 (Local)": h_price,
        "X (Empate)": x_price,
        "2 (Visita)": a_price,
        "Casa Ref": ref_bm,
        "Diagnóstico QuantShield": verdict,
        "_has_value": has_val,
    })

# Sort by EV (%) descending
detected_opportunities.sort(key=lambda x: x["ev_percent"], reverse=True)


st.session_state.current_opportunities = detected_opportunities
st.session_state.current_events = events
st.session_state.current_fixtures = all_fixture_rows


# ==============================================================================
# MAIN TABS INTERFACE
# ==============================================================================
tab_betplay, tab_scanner, tab_audit, tab_telegram, tab_simulator = st.tabs([
    "📱 Centro de Partido & Combinadas BetPlay",
    "🎯 Cartelera General (+EV Finder)",
    "📈 Contabilidad & Auditoría (SQLite)",
    "📲 Alertas Móviles (Telegram)",
    "🧪 Simulador Histórico (Backtesting)",
])

# ------------------------------------------------------------------------------
# TAB 1: CENTRO DE PARTIDO & COMBINADAS BETPLAY (ESTILO UNIFICADO APP MÓVIL)
# ------------------------------------------------------------------------------
with tab_betplay:
    try:

        st.subheader("📱 Centro de Partido & Combinadas BetPlay")
        st.caption(
            "Experiencia unificada estilo app móvil BetPlay: Tarjetas de 'Combinadas para ti' (Same-Game Parlay de baja varianza) "
            "y sábana integral de mercados de seguridad calculados a partir de la matriz Poisson 9x9."
        )

        if not events:
            st.info("ℹ️ No hay partidos cargados para esta competición. Selecciona un torneo arriba o activa el Modo Simulación en la barra lateral.")
        else:
            # Match selector
            match_labels = []
            for ev in events:
                ht = ev.get("home_team", "Local")
                at = ev.get("away_team", "Visita")
                t_str = format_short_local_time(ev.get("commence_time", ""))
                match_labels.append(f"{ht} vs {at} ({t_str})")

            c_msel1, c_msel2 = st.columns([2.5, 1])
            with c_msel1:
                selected_match_label = st.selectbox(
                    "⚽ Seleccionar Partido a Analizar:",
                    options=match_labels,
                    index=0,
                    key="betplay_match_select",
                )
            selected_match_idx = match_labels.index(selected_match_label)
            curr_event = events[selected_match_idx]
            cur_home = curr_event.get("home_team", "Local")
            cur_away = curr_event.get("away_team", "Visita")
            cur_match_name = f"{cur_home} vs {cur_away}"
            cur_match_time = format_to_local_time(curr_event.get("commence_time", ""))

            with c_msel2:
                st.markdown(
                    "<div style='margin-top: 28px; text-align: right;'>"
                    "<span style='background-color: #1e3a5f; color: #38bdf8; padding: 6px 12px; border-radius: 8px; font-weight: 700; font-size: 0.88em;'>"
                    "⚡ BetPlay Verified</span></div>",
                    unsafe_allow_html=True,
                )

            # Expected Goals & Poisson resolution
            adv_props = curr_event.get("advanced_poisson_props", {})
            lam_h = float(curr_event.get("lambda_h", adv_props.get("lambda_h", 1.35)))
            lam_a = float(curr_event.get("lambda_a", adv_props.get("lambda_a", 1.05)))
            ai_adj = st.session_state.match_ai_adjustments.get(cur_match_name)

            # Extract Colombian retail odds from event bookmakers
            retail_odds_map = {}
            for bm in curr_event.get("bookmakers", []):
                if bm.get("key", "").lower() != "pinnacle":
                    for m in bm.get("markets", []):
                        if m.get("key") == "h2h":
                            for o in m.get("outcomes", []):
                                if o.get("name") == cur_home:
                                    retail_odds_map["1"] = o.get("price", 0.0)
                                elif o.get("name") in ("Draw", "Empate", "X"):
                                    retail_odds_map["X"] = o.get("price", 0.0)
                                elif o.get("name") == cur_away:
                                    retail_odds_map["2"] = o.get("price", 0.0)

            betplay_cat = calculate_betplay_security_catalog(
                home_team=cur_home,
                away_team=cur_away,
                lambda_h=lam_h,
                lambda_a=lam_a,
                bankroll=bankroll_val,
                kelly_multiplier=kelly_multiplier,
                retail_odds=retail_odds_map,
            )

            sgp_cards = generate_betplay_same_game_parlays(
                catalog=betplay_cat,
                bankroll=bankroll_val,
                kelly_multiplier=kelly_multiplier,
                ai_adjustment=ai_adj,
            )

            # Top Banner: Match Context & Expected Goals
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #0f192d 0%, #172554 100%); border: 1.5px solid #1e3a5f; border-radius: 12px; padding: 14px 18px; margin: 10px 0 16px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: 0.3px;">
                                🏟️ {cur_home} vs {cur_away}
                            </div>
                            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 3px;">
                                📅 <b>Programación Oficial:</b> {cur_match_time}
                            </div>
                        </div>
                        <div>
                            <span style="background: #0b1f3a; border: 1px solid #38bdf8; color: #38bdf8; padding: 5px 12px; border-radius: 8px; font-size: 0.88rem; font-weight: 700;">
                                ⚽ Goles Esperados (xG): {cur_home} {lam_h:.2f} | {cur_away} {lam_a:.2f} | Total: {lam_h+lam_a:.2f}
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # AI Context Box if available
            if ai_adj:
                nivel_riesgo = str(get_ai_field(ai_adj, "nivel_riesgo", "Bajo")).capitalize()
                risk_color = "#10b981" if nivel_riesgo == "Bajo" else ("#f59e0b" if nivel_riesgo == "Medio" else "#ef4444")
                confianza_val = float(get_ai_field(ai_adj, "confianza_ia", 0.80))
                justificacion_val = str(get_ai_field(ai_adj, "justificacion_tactica", "Análisis estadístico base sin alteraciones tácticas."))
                f_local_val = float(get_ai_field(ai_adj, "factor_ajuste_local", 1.0))
                f_visita_val = float(get_ai_field(ai_adj, "factor_ajuste_visita", 1.0))
                st.markdown(
                    f"""
                    <div style="background-color: #0f192d; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; border-left: 4px solid #8b5cf6;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 700; color: #c084fc; font-size: 0.9em;">🧠 Contexto Táctico Gemini ({selected_gemini_model})</span>
                            <span style="background-color: {risk_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.78em; font-weight: 700;">
                                Riesgo {nivel_riesgo} (Confianza: {int(confianza_val*100)}%)
                            </span>
                        </div>
                        <div style="font-size: 0.82em; margin-bottom: 4px; color: #94a3b8;">
                            Multiplicador Local: <code style="color: #60a5fa;">{f_local_val:.2f}x</code> | Multiplicador Visita: <code style="color: #f472b6;">{f_visita_val:.2f}x</code>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.85em; font-style: italic;">
                            "{justificacion_val}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ----------------------------------------------------------------------
            # SECCIÓN 1: CARDS "COMBINADAS PARA TI" (BETPLAY STYLE CONNECTED DOTS)
            # ----------------------------------------------------------------------
            st.markdown("### ⚡ Combinadas para ti (Mismo Partido - Baja Varianza)")
            st.caption(
                "Combinaciones inteligentes de 2 a 3 selecciones del mismo partido con correlación positiva y alta probabilidad individual. "
                "Para apostarlas, ingresa a BetPlay en tu celular ➔ abre el partido ➔ pestaña **'Crea tu Apuesta'**."
            )

            col_c1, col_c2, col_c3 = st.columns(3)
            cols_for_cards = [col_c1, col_c2, col_c3]

            for idx_card, (col_tgt, card) in enumerate(zip(cols_for_cards, sgp_cards)):
                with col_tgt:
                    badge_bg = "#065f46" if "BLINDADA" in card["badge"] else ("#92400e" if "RITMO" in card["badge"] else "#5b21b6")
                    badge_fg = "#a7f3d0" if "BLINDADA" in card["badge"] else ("#fde68a" if "RITMO" in card["badge"] else "#ddd6fe")
                    border_color = "#10b981" if "BLINDADA" in card["badge"] else ("#f59e0b" if "RITMO" in card["badge"] else "#8b5cf6")
                    card_ev_str = f"+{card['ev_pct']}%" if card["ev_pct"] >= 0 else f"{card['ev_pct']}%"
                    card_ev_color = "#10b981" if card["ev_pct"] >= 0 else "#94a3b8"

                    # Build connected timeline HTML
                    legs_html = ""
                    for leg_idx, leg in enumerate(card["legs"]):
                        is_last = (leg_idx == len(card["legs"]) - 1)
                        margin_b = "0px" if is_last else "14px"
                        legs_html += f"""
                        <div style="position: relative; margin-bottom: {margin_b};">
                            <div style="position: absolute; left: -21px; top: 3px; width: 10px; height: 10px; border-radius: 50%; background: #f59e0b; box-shadow: 0 0 6px rgba(245, 158, 11, 0.9);"></div>
                            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.4px;">
                                {leg['market']}
                            </div>
                            <div style="font-size: 0.88rem; color: #ffffff; font-weight: 700; line-height: 1.3;">
                                {leg['selection']} <span style="color: #fbbf24; font-size: 0.82rem; font-weight: 800;">@{leg['odds']:.2f}</span>
                            </div>
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div style="background: #0f192d; border: 1.5px solid {border_color}; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.45);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="background: {badge_bg}; color: {badge_fg}; font-size: 0.76rem; font-weight: 800; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.5px;">
                                    {card['badge']}
                                </span>
                                <span style="font-size: 0.73rem; color: #94a3b8; font-weight: 600;">{card['risk_label']}</span>
                            </div>

                            <!-- Connected dots timeline (BetPlay mobile) -->
                            <div style="position: relative; padding-left: 22px; margin: 14px 0 14px 4px;">
                                <div style="position: absolute; left: 4px; top: 6px; bottom: 8px; width: 2px; background: linear-gradient(180deg, #f59e0b, #fbbf24);"></div>
                                {legs_html}
                            </div>

                            <!-- Metrics Summary Box -->
                            <div style="background: #16243f; border-radius: 10px; padding: 10px 12px; margin: 12px 0;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <span style="color: #94a3b8; font-size: 0.82rem;">Cuota Combinada:</span>
                                    <span style="color: #fbbf24; font-weight: 800; font-size: 1.08rem;">@{card['odds']:.2f}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <span style="color: #94a3b8; font-size: 0.80rem;">Probabilidad Real:</span>
                                    <span style="color: #38bdf8; font-weight: 700; font-size: 0.88rem;">{card['prob_pct']}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <span style="color: #94a3b8; font-size: 0.80rem;">Ventaja (+EV):</span>
                                    <span style="color: {card_ev_color}; font-weight: 700; font-size: 0.88rem;">{card_ev_str}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1e3a5f; padding-top: 6px; margin-top: 6px;">
                                    <span style="color: #f1f5f9; font-weight: 700; font-size: 0.84rem;">Stake Seguro:</span>
                                    <span style="color: #34d399; font-weight: 800; font-size: 0.96rem;">{format_money(card['recommended_stake'], selected_curr_code)}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
                                    <span style="color: #94a3b8; font-size: 0.78rem;">Ganancia Neta:</span>
                                    <span style="color: #a7f3d0; font-size: 0.82rem; font-weight: 600;">+{format_money(card['potential_profit'], selected_curr_code)}</span>
                                </div>
                            </div>

                            <!-- BetPlay mobile instruction -->
                            <div style="background: #091120; border-left: 3px solid #f59e0b; padding: 7px 10px; border-radius: 4px; font-size: 0.76rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 10px;">
                                📲 <b>En BetPlay:</b> Abre {cur_home} vs {cur_away} ➔ Pestaña <b>'Crea tu Apuesta'</b> ➔ Activa estas selecciones.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.button(f"💾 Guardar {card['badge']} en SQLite", key=f"save_sgp_card_{selected_match_idx}_{idx_card}", use_container_width=True):
                        legs_summary = " + ".join([f"{l['selection']} (@{l['odds']:.2f})" for l in card['legs']])
                        new_id = add_bet(
                            sport="Combinada / BetPlay SGP",
                            tournament=curr_event.get("sport_title", selected_meta.get("name", "Fútbol")),
                            match=cur_match_name,
                            market=card["badge"],
                            selection=" & ".join([l["selection"] for l in card["legs"]]),
                            bookmaker="BetPlay",
                            odds=card["odds"],
                            fair_odds=card["fair_odds"],
                            ev_percent=card["ev_pct"],
                            stake=card["recommended_stake"],
                            status="Pendiente",
                            notes=f"BetPlay Same-Game Parlay. Legs: {legs_summary}",
                            ai_analysis=card.get("instructions", ""),
                        )
                        st.toast(f"✅ ¡{card['badge']} guardada en SQLite (ID #{new_id})!", icon="💾")
                        st.success(f"¡Combinada guardada con éxito en la base de datos (ID #{new_id})!")

            # ----------------------------------------------------------------------
            # SECCIÓN 2: SÁBANA UNIFICADA DE MERCADOS DE SEGURIDAD BETPLAY
            # ----------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### 📋 Sábana Unificada de Mercados de Seguridad BetPlay")
            st.caption(
                "Todos los mercados secundarios calculados con la matriz bivariada de Poisson (9x9). "
                "Las opciones marcadas con 🟢 ENTRAR tienen Esperanza Matemática Positiva (+EV)."
            )

            def _render_sec_item(item_data, key_suffix):
                with st.container(border=True):
                    col_i1, col_i2, col_i3, col_i4 = st.columns([2.3, 1.1, 1.1, 1.5])
                    with col_i1:
                        st.markdown(f"**{item_data['name']}**")
                        st.progress(float(min(1.0, max(0.0, item_data['prob_pct'] / 100.0))))
                        st.caption(f"Prob. Poisson: **{item_data['prob_pct']}%** | Cuota Justa: **@{item_data['fair_odds']:.2f}**")
                    with col_i2:
                        st.metric("Cuota BetPlay", f"@{item_data['retail_odds']:.2f}")
                    with col_i3:
                        st.metric("Ventaja (+EV)", f"{item_data['ev_pct']:+.2f}%")
                    with col_i4:
                        if item_data["has_value"]:
                            st.markdown(
                                f"<span style='background:#059669; color:white; padding:4px 8px; border-radius:6px; font-weight:700; font-size:0.82em;'>🟢 ENTRAR</span> "
                                f"<span style='color:#34d399; font-weight:700; font-size:0.85em;'>Stake: {format_money(item_data['rec_stake'], selected_curr_code)}</span>",
                                unsafe_allow_html=True,
                            )
                            if st.button("💾 Guardar", key=f"btn_sec_save_{key_suffix}", use_container_width=True):
                                m_id = add_bet(
                                    sport="Fútbol",
                                    tournament=curr_event.get("sport_title", selected_meta.get("name", "Fútbol")),
                                    match=cur_match_name,
                                    market="Mercado de Seguridad BetPlay",
                                    selection=item_data["name"],
                                    bookmaker="BetPlay",
                                    odds=item_data["retail_odds"],
                                    fair_odds=item_data["fair_odds"],
                                    ev_percent=item_data["ev_pct"],
                                    stake=item_data["rec_stake"],
                                    status="Pendiente",
                                    notes=f"Poisson 9x9. Prob: {item_data['prob_pct']}%. Fair: @{item_data['fair_odds']}",
                                    ai_analysis="",
                                )
                                st.toast(f"✅ ¡Apuesta #{m_id} registrada!", icon="💾")
                                st.success(f"¡Guardada en SQLite (ID #{m_id})!")
                        else:
                            st.markdown(
                                "<span style='background:#475569; color:#cbd5e1; padding:4px 8px; border-radius:6px; font-weight:600; font-size:0.82em;'>⚪ PASAR</span> "
                                "<span style='color:#94a3b8; font-size:0.80em;'>Margen Casa</span>",
                                unsafe_allow_html=True,
                            )

            # BLOQUE 1: Resultado Final (1X2) y Doble Oportunidad
            with st.expander("⚽ Bloque 1: Resultado Final (1X2) y Doble Oportunidad (1X, 12, X2)", expanded=True):
                col_b1_left, col_b1_right = st.columns(2)
                with col_b1_left:
                    st.markdown("##### 🏆 Resultado Final (1X2):")
                    _render_sec_item(betplay_cat["block_1x2"]["home"], f"{selected_match_idx}_1x2_h")
                    _render_sec_item(betplay_cat["block_1x2"]["draw"], f"{selected_match_idx}_1x2_x")
                    _render_sec_item(betplay_cat["block_1x2"]["away"], f"{selected_match_idx}_1x2_a")
                with col_b1_right:
                    st.markdown("##### 🛡️ Doble Oportunidad (Protección):")
                    _render_sec_item(betplay_cat["block_double_chance"]["1X"], f"{selected_match_idx}_dc_1x")
                    _render_sec_item(betplay_cat["block_double_chance"]["12"], f"{selected_match_idx}_dc_12")
                    _render_sec_item(betplay_cat["block_double_chance"]["X2"], f"{selected_match_idx}_dc_x2")

            # BLOQUE 2: Apuesta sin Empate (Draw No Bet)
            with st.expander("🛡️ Bloque 2: Apuesta sin Empate (Draw No Bet / Reembolso si Empatan)", expanded=True):
                st.caption("🟡 Si el partido termina en empate, la casa reembolsa el 100% de tu dinero apostado (Push).")
                col_b2_left, col_b2_right = st.columns(2)
                with col_b2_left:
                    _render_sec_item(betplay_cat["block_dnb"]["dnb_home"], f"{selected_match_idx}_dnb_h")
                with col_b2_right:
                    _render_sec_item(betplay_cat["block_dnb"]["dnb_away"], f"{selected_match_idx}_dnb_a")

            # BLOQUE 3: Total de Goles por Equipo (Local y Visita)
            with st.expander("🥅 Bloque 3: Total de Goles por Equipo (Individual Local / Visita)", expanded=False):
                col_b3_left, col_b3_right = st.columns(2)
                with col_b3_left:
                    st.markdown(f"##### 🏠 Goles de {cur_home} (Local):")
                    _render_sec_item(betplay_cat["block_team_goals_home"]["over_05"], f"{selected_match_idx}_h_o05")
                    _render_sec_item(betplay_cat["block_team_goals_home"]["over_15"], f"{selected_match_idx}_h_o15")
                    _render_sec_item(betplay_cat["block_team_goals_home"]["over_25"], f"{selected_match_idx}_h_o25")
                with col_b3_right:
                    st.markdown(f"##### ✈️ Goles de {cur_away} (Visita):")
                    _render_sec_item(betplay_cat["block_team_goals_away"]["over_05"], f"{selected_match_idx}_a_o05")
                    _render_sec_item(betplay_cat["block_team_goals_away"]["over_15"], f"{selected_match_idx}_a_o15")
                    _render_sec_item(betplay_cat["block_team_goals_away"]["over_25"], f"{selected_match_idx}_a_o25")

            # BLOQUE 4: Ambos Equipos Marcarán (BTTS) y Gol en Ambas Mitades
            with st.expander("⚽ Bloque 4: Ambos Equipos Marcarán & Gol en Ambas Mitades", expanded=False):
                col_b4_left, col_b4_right = st.columns(2)
                with col_b4_left:
                    st.markdown("##### 🤝 Ambos Equipos Anotarán:")
                    _render_sec_item(betplay_cat["block_btts"]["btts_yes"], f"{selected_match_idx}_btts_y")
                    _render_sec_item(betplay_cat["block_btts"]["btts_no"], f"{selected_match_idx}_btts_n")
                with col_b4_right:
                    st.markdown("##### ⏱️ Distribución Temporal de Goles:")
                    _render_sec_item(betplay_cat["block_both_halves"]["halves_yes"], f"{selected_match_idx}_half_y")
                    _render_sec_item(betplay_cat["block_both_halves"]["halves_no"], f"{selected_match_idx}_half_n")
                    _render_sec_item(betplay_cat["block_both_halves"]["first_half_over05"], f"{selected_match_idx}_1h_o05")

            # BLOQUE 5: Bandas de Goles y Combos Pre-armados
            with st.expander("🧩 Bloque 5: Número de Bandas de Goles & Combos Bet Builder", expanded=False):
                col_b5_left, col_b5_right = st.columns(2)
                with col_b5_left:
                    st.markdown("##### 🎯 Bandas de Goles Totales:")
                    _render_sec_item(betplay_cat["block_goal_bands"]["band_0_2"], f"{selected_match_idx}_band_02")
                    _render_sec_item(betplay_cat["block_goal_bands"]["band_3_4"], f"{selected_match_idx}_band_34")
                    _render_sec_item(betplay_cat["block_goal_bands"]["band_5_plus"], f"{selected_match_idx}_band_5p")
                    _render_sec_item(betplay_cat["block_goal_bands"]["over_15_total"], f"{selected_match_idx}_tot_o15")
                with col_b5_right:
                    st.markdown("##### 🧩 Combos Pre-armados del Mismo Partido:")
                    _render_sec_item(betplay_cat["block_combos"]["h_and_btts"], f"{selected_match_idx}_h_btts")
                    _render_sec_item(betplay_cat["block_combos"]["a_and_btts"], f"{selected_match_idx}_a_btts")
                    _render_sec_item(betplay_cat["block_combos"]["draw_under25"], f"{selected_match_idx}_draw_u25")


    except Exception as e:
        st.error("⚠️ Error en pestaña Centro de Partido & Combinadas BetPlay: " + str(e))

# ------------------------------------------------------------------------------
# TAB 2: CARTELERA GENERAL (+EV FINDER)
# ------------------------------------------------------------------------------
with tab_scanner:
    try:

        st.subheader("🎯 Cartelera General (+EV Finder)")
        st.markdown(
            "Detecta apuestas de **Esperanza Matemática Positiva** comparando las probabilidades limpias "
            "del oráculo de mercado (**Pinnacle Vig-Free**) con modelos probabilísticos bivariados de Poisson y distribución Gaussiana."
        )

        # View Mode Selector: Solo Oportunidades vs Cartelera Completa
        c_view1, c_view2 = st.columns([1.8, 1])
        with c_view1:
            view_mode = st.radio(
                "Seleccionar Vista:",
                options=["🎯 Solo Apuestas Recomendadas (+EV)", "📋 Cartelera Completa (Todos los Partidos de la Fecha)"],
                horizontal=True,
                label_visibility="collapsed",
            )
        with c_view2:
            val_count = len(detected_opportunities)
            tot_count = len(events)
            st.markdown(
                f"<div style='text-align: right; padding-top: 6px;'>"
                f"<span style='background-color: #059669; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 700; font-size: 0.85em;'>"
                f"🟢 {val_count} con Valor</span> "
                f"<span style='background-color: #475569; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85em;'>"
                f"⚪ {tot_count - val_count} Sin Ventaja</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if "Cartelera Completa" in view_mode:
            st.markdown("##### 📋 Cartelera Completa de la Fecha & Diagnóstico:")
            st.caption("A continuación se evalúan todos los partidos programados de la competición. Solo los partidos marcados con **🟢 +EV** tienen ventaja matemática a tu favor; los partidos marcados con **⚪ Sin Ventaja** tienen cuotas que favorecen a la casa.")
            clean_df = pd.DataFrame(all_fixture_rows).drop(columns=["_has_value"], errors="ignore")
            st.dataframe(clean_df, use_container_width=True, hide_index=True)

        if not detected_opportunities:
            st.info(
                "ℹ️ No se detectaron oportunidades con los filtros actuales "
                f"(EV >= {min_ev_filter}%, Cuotas entre {odds_range[0]} y {odds_range[1]}). "
                "Revisa la pestaña de 'Cartelera Completa' para ver los partidos evaluados o prueba relajando los filtros en la barra lateral."
            )
        else:
            if "Cartelera Completa" not in view_mode:
                st.success(
                    f"✅ Se encontraron **{len(detected_opportunities)} oportunidades** con Esperanza Matemática Positiva (+EV)."
                )

                # Overview Table
                summary_rows = []
                for i, opp in enumerate(detected_opportunities):
                    summary_rows.append({
                        "#": i + 1,
                        "Fecha/Hora (Local)": format_short_local_time(opp.get("commence_time", "")),
                        "Partido": opp["match"],
                        "Mercado": opp["market"],
                        "Selección": opp["selection"],
                        "Casa": opp["bookmaker"],
                        "Cuota": f"{opp['odds']:.2f}",
                        "Cuota Justa": f"{opp['fair_odds']:.2f}",
                        "Prob. Efectiva": f"{opp['effective_prob_pct']}%",
                        "EV (%)": f"+{opp['ev_percent']:.2f}%",
                        f"Stake ({selected_curr_code})": format_money(opp['recommended_stake'], selected_curr_code),
                        "% Bankroll": f"{opp['recommended_stake_pct']}%",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.markdown("#### 📋 Tarjetas de Ejecución Paso a Paso & Seguro Sintético")

            for idx, opp in enumerate(detected_opportunities):
                cap_badge = " 🛑 [TOPE 3.5% APLICADO]" if opp.get("is_capped") else ""
                local_time_full = format_to_local_time(opp.get("commence_time", ""))
                formatted_stake_str = format_money(opp['recommended_stake'], selected_curr_code)
                expander_title = (
                    f"#{idx+1} | {opp['match']} — {opp['market']} @ {opp['odds']} "
                    f"(EV: +{opp['ev_percent']}% | Stake: {formatted_stake_str}){cap_badge}"
                )

                with st.expander(expander_title, expanded=(idx == 0)):
                    market_type = opp.get("market", "")
                    match_str = opp.get("match", "")

                    # Explicación clara y didáctica: ¿A qué apostar y cómo funciona?
                    if "Over 2.0" in market_type:
                        profit_val = round(opp['recommended_stake'] * (opp['odds'] - 1), 2)
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1.5px solid #f59e0b; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
                                <div style="font-weight: 700; color: #fbbf24; font-size: 1.05rem; margin-bottom: 6px;">
                                    ⚽ ¿A QUÉ ESTÁS APOSTANDO EXACTAMENTE? ➔ <u>TOTAL DE GOLES DE AMBOS EQUIPOS</u>
                                </div>
                                <div style="color: #f1f5f9; font-size: 0.93rem; line-height: 1.5;">
                                    <b>NO apuestas a que gane ningún equipo en particular.</b> Apuestas a la <b>suma combinada de goles</b> entre ambos equipos (<code>{match_str}</code>):<br/>
                                    • 🟢 <b>Ganas la apuesta (+{format_money(profit_val, selected_curr_code)}):</b> Si entre ambos equipos anotan <b>3 o más goles</b> en el partido (ej. 2-1, 3-0, 1-2, 2-2).<br/>
                                    • 🟡 <b>Reembolso del 100% (Push / Ni ganas ni pierdes):</b> Si el partido termina con <b>exactamente 2 goles</b> (ej. 1-1, 2-0, 0-2). ¡Se te devuelve íntegro tu dinero apostado ({formatted_stake_str})!<br/>
                                    • 🔴 <b>Pierdes:</b> Únicamente si el partido termina con 0 o 1 gol (0-0, 1-0 o 0-1).
                                </div>
                                <div style="margin-top: 8px; color: #38bdf8; font-size: 0.88rem; font-weight: 600;">
                                    📅 Programación: {local_time_full}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif "Draw No Bet" in market_type:
                        home_name = match_str.split(" vs ")[0] if " vs " in match_str else "Local"
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1.5px solid #6366f1; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
                                <div style="font-weight: 700; color: #818cf8; font-size: 1.05rem; margin-bottom: 6px;">
                                    🛡️ ¿A QUÉ ESTÁS APOSTANDO EXACTAMENTE? ➔ <u>VICTORIA DE {home_name.upper()} (EMPATE REEMBOLSA)</u>
                                </div>
                                <div style="color: #f1f5f9; font-size: 0.93rem; line-height: 1.5;">
                                    Apuestas a favor de la victoria de <b>{home_name}</b>, pero con cobertura total ante un posible empate:<br/>
                                    • 🟢 <b>Ganas la apuesta:</b> Si <b>{home_name}</b> gana el partido.<br/>
                                    • 🟡 <b>Reembolso del 100% (Push):</b> Si el partido termina en <b>Empate</b> (0-0, 1-1, 2-2). ¡Se te devuelve todo el stake ({formatted_stake_str})!<br/>
                                    • 🔴 <b>Pierdes:</b> Si gana el equipo visitante.
                                </div>
                                <div style="margin-top: 8px; color: #38bdf8; font-size: 0.88rem; font-weight: 600;">
                                    📅 Programación: {local_time_full}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1.5px solid #3b82f6; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
                                <div style="font-weight: 700; color: #60a5fa; font-size: 1.05rem; margin-bottom: 6px;">
                                    🎯 SELECCIÓN RECOMENDADA: <u>{opp.get('selection', '')}</u>
                                </div>
                                <div style="color: #f1f5f9; font-size: 0.93rem; line-height: 1.5;">
                                    Apuestas en el mercado <b>{market_type}</b> a cuota <b>{opp['odds']}</b> en <b>{opp['bookmaker']}</b>.<br/>
                                    📅 <b>Programación Oficial:</b> {local_time_full}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    col_info1, col_info2, col_info3 = st.columns([1.5, 1.5, 1])

                    with col_info1:
                        st.markdown(f"**Evento:** `{opp['match']}`")
                        st.markdown(f"**📅 Horario Local (UTC-5):** `{local_time_full}`")
                        st.markdown(f"**Competición:** `{opp['tournament']}`")
                        st.markdown(f"**Mercado:** `{opp['market']}`")
                        st.markdown(f"**Selección:** `{opp['selection']}`")

                    with col_info2:
                        st.markdown(f"**Casa de Apuestas:** `{opp['bookmaker']}`")
                        st.markdown(f"**Cuota Actual:** `{opp['odds']}`")
                        st.markdown(f"**Cuota Mínima Aceptable:** `{opp['fair_odds']}`")
                        st.markdown(f"**Probabilidad Estimada:** `{opp['effective_prob_pct']}%`")

                    with col_info3:
                        st.metric("Esperanza Matemática", f"+{opp['ev_percent']}%")
                        st.metric("Stake Óptimo", formatted_stake_str)

                    # AI Context Agent Verdict Box
                    if opp.get("ai_adjustment"):
                        ai_res = opp["ai_adjustment"]
                        nivel_riesgo = str(get_ai_field(ai_res, "nivel_riesgo", "Bajo")).capitalize()
                        risk_color = "#10b981" if nivel_riesgo == "Bajo" else ("#f59e0b" if nivel_riesgo == "Medio" else "#ef4444")
                        confianza_pct = int(float(get_ai_field(ai_res, "confianza_ia", 0.80)) * 100)
                        factor_loc = float(get_ai_field(ai_res, "factor_ajuste_local", 1.0))
                        factor_vis = float(get_ai_field(ai_res, "factor_ajuste_visita", 1.0))
                        justificacion_txt = str(get_ai_field(ai_res, "justificacion_tactica", "Análisis estadístico base sin alteraciones tácticas."))
                        st.markdown(
                            f"""
                            <div style="background-color: #1e293b; border-radius: 8px; padding: 12px; margin: 10px 0; border-left: 4px solid #8b5cf6;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <span style="font-weight: 700; color: #c084fc;">🧠 Veredicto Táctico Gemini ({selected_gemini_model})</span>
                                    <span style="background-color: {risk_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 700;">
                                        Riesgo {nivel_riesgo} (Confianza: {confianza_pct}%)
                                    </span>
                                </div>
                                <div style="font-size: 0.88em; margin-bottom: 6px; color: #94a3b8;">
                                    Multiplicador Local: <code style="color: #60a5fa;">{factor_loc:.2f}x</code> | 
                                    Multiplicador Visita: <code style="color: #f472b6;">{factor_vis:.2f}x</code>
                                </div>
                                <div style="font-style: italic; color: #f1f5f9; font-size: 0.9em;">
                                    "{justificacion_txt}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Interactive Breaking News Injection
                    st.markdown("##### ⚡ Novedad de Última Hora & Recálculo en Vivo:")
                    c_n1, c_n2 = st.columns([3, 1])
                    match_name_str = opp["match"]
                    current_saved_note = st.session_state.match_user_notes.get(match_name_str, "")
                    with c_n1:
                        user_news_input = st.text_input(
                            "📝 Inyectar noticia de última hora:",
                            value=current_saved_note,
                            key=f"news_input_{idx}_{match_name_str}_{opp['market']}",
                            placeholder="Ej. Arquero titular lesionado en Santa Fe, juegan con suplentes...",
                            label_visibility="collapsed",
                        )
                    with c_n2:
                        if st.button("🔄 Recalcular con Gemini", key=f"recalc_btn_{idx}_{match_name_str}_{opp['market']}", use_container_width=True):
                            st.session_state.match_user_notes[match_name_str] = user_news_input
                            with st.spinner("Reevaluando partido con Gemini..."):
                                odds_snap = {"odds_actual": opp["odds"], "fair_odds": opp["fair_odds"]}
                                new_ai_adj = evaluar_contexto_partido(
                                    deporte=opp["sport"],
                                    torneo=opp["tournament"],
                                    local=match_name_str.split(" vs ")[0],
                                    visita=match_name_str.split(" vs ")[1] if " vs " in match_name_str else "",
                                    cuotas_actuales=odds_snap,
                                    contexto_usuario=user_news_input,
                                    api_key=gemini_api_key_input.strip() if gemini_api_key_input else None,
                                    model_name=selected_gemini_model,
                                    is_demo=False,
                                )
                                st.session_state.match_ai_adjustments[match_name_str] = new_ai_adj
                            st.toast(f"¡Factores de {match_name_str} actualizados con Gemini!", icon="🧠")
                            st.rerun()

                    st.markdown("---")
                    st.markdown("##### 📝 Instrucciones Paso a Paso para Meter la Apuesta:")
                    rendered_instructions = opp["instructions"]
                    if selected_curr_code == "COP":
                        rendered_instructions = rendered_instructions.replace(f"${opp['recommended_stake']}", format_money(opp['recommended_stake'], "COP"))
                    st.info(rendered_instructions)

                    # If Dutching Saver data is present, display visual breakdown
                    if opp.get("dutching"):
                        d = opp["dutching"]
                        st.markdown("##### 🛡️ Desglose del Seguro Sintético (Dutching Saver):")
                        c_d1, c_d2 = st.columns(2)
                        with c_d1:
                            st.markdown(
                                f"- **Apuesta 1 (Victoria Local):** `{format_money(d['stake_home'], selected_curr_code)}`\n"
                                f"- **Retorno si Gana Local:** `{format_money(d['return_if_home'], selected_curr_code)}` (**+{format_money(d['net_profit_if_home'], selected_curr_code)} ganancia neta**)"
                            )
                        with c_d2:
                            st.markdown(
                                f"- **Apuesta 2 (Cobertura Empate):** `{format_money(d['stake_draw'], selected_curr_code)}`\n"
                                f"- **Retorno si Empatan:** `{format_money(d['return_if_draw'], selected_curr_code)}` (**Reembolso exacto del 100%**)"
                            )

                    # 🇨🇴 Validador Rápido de Cuota en Casas de Colombia (BetPlay, Wplay, Rushbet, Bwin)
                    st.markdown("---")
                    st.markdown("##### 🇨🇴 Validador de Cuota en tu Casa de Apuestas (Colombia):")
                    st.caption("Verifica si la cuota que ves en tu app de BetPlay, Rushbet, Wplay o Bwin sigue teniendo valor matemático:")
                    c_co1, c_co2 = st.columns([1.2, 1.8])
                    with c_co1:
                        col_bm_selected = st.selectbox(
                            "Casa en Colombia:",
                            options=["BetPlay", "Rushbet", "Wplay", "Bwin Colombia", "Betsson", "Codere"],
                            index=0 if "BetPlay" in opp["bookmaker"] else (1 if "Rushbet" in opp["bookmaker"] else (2 if "Wplay" in opp["bookmaker"] else 3)),
                            key=f"col_bm_sel_{idx}_{opp['match']}_{opp['market']}"
                        )
                        user_live_odd = st.number_input(
                            f"Cuota en {col_bm_selected}:",
                            min_value=1.01,
                            max_value=15.0,
                            value=float(opp["odds"]),
                            step=0.01,
                            key=f"live_odd_input_{idx}_{opp['match']}_{opp['market']}"
                        )

                    with c_co2:
                        fair_odd_val = opp["fair_odds"]
                        eff_prob = (opp["effective_prob_pct"] / 100.0) if opp["effective_prob_pct"] > 0 else (1.0 / fair_odd_val)
                        recalc_ev = (eff_prob * user_live_odd - 1.0) * 100.0
                    
                        # Kelly stake recalculation
                        b_val = user_live_odd - 1.0
                        f_star = (b_val * eff_prob - (1.0 - eff_prob)) / b_val if b_val > 0 else 0.0
                        recalc_stake = max(0.0, f_star * kelly_multiplier * bankroll_val)
                        recalc_stake = min(recalc_stake, bankroll_val * 0.035)  # Capped at 3.5%
                    
                        if recalc_ev >= min_ev_filter:
                            st.success(
                                f"✅ **¡Excelente Valor en {col_bm_selected}!**\n\n"
                                f"• Cuota Mínima Aceptable: `{fair_odd_val:.2f}` | Cuota {col_bm_selected}: `{user_live_odd:.2f}`\n\n"
                                f"• Esperanza Matemática: **+{recalc_ev:.2f}%**\n\n"
                                f"• Stake Sugerido a apostar: **{format_money(recalc_stake, selected_curr_code)}**"
                            )
                        elif recalc_ev > 0:
                            st.warning(
                                f"⚠️ **Valor Marginal (+{recalc_ev:.2f}%):**\n\n"
                                f"Sigue teniendo ventaja sobre la casa, pero está por debajo de tu filtro (+{min_ev_filter}%). "
                                f"Stake prudente: **{format_money(recalc_stake, selected_curr_code)}**."
                            )
                        else:
                            st.error(
                                f"❌ **Sin Valor Matemático (-EV: {recalc_ev:.2f}%):**\n\n"
                                f"La cuota de `{user_live_odd:.2f}` en {col_bm_selected} no supera la cuota justa (`{fair_odd_val:.2f}`). "
                                f"La casa tiene ventaja matemática. **No se recomienda apostar.**"
                            )

                    # Button to register directly to SQLite
                    st.markdown("---")
                    btn_key = f"save_bet_{idx}_{opp['match']}_{opp['market']}_{opp['bookmaker']}"
                    if st.button(f"💾 Registrar Apuesta ({col_bm_selected} @ {user_live_odd})", key=btn_key):
                        ai_justification_text = str(get_ai_field(opp.get("ai_adjustment"), "justificacion_tactica", "")) if opp.get("ai_adjustment") else ""
                        combined_notes = opp.get("lambdas", "")
                        if ai_justification_text:
                            combined_notes = f"{combined_notes} | IA: {ai_justification_text}"

                        new_id = add_bet(
                            sport=opp["sport"],
                            tournament=opp["tournament"],
                            match=opp["match"],
                            market=opp["market"],
                            selection=opp["selection"],
                            bookmaker=col_bm_selected,
                            odds=user_live_odd,
                            fair_odds=opp["fair_odds"],
                            ev_percent=round(recalc_ev, 2),
                            stake=round(recalc_stake, 2),
                            status="Pendiente",
                            notes=combined_notes,
                            ai_analysis=ai_justification_text,
                        )
                        st.toast(f"✅ ¡Apuesta #{new_id} ({col_bm_selected}) registrada en quantshield.db!", icon="💾")
                        st.success(f"Apuesta guardada con éxito en la base de datos (ID #{new_id} - {col_bm_selected}). Puedes auditarla en la pestaña de Contabilidad.")


        if len(detected_opportunities) >= 2:
            st.markdown('---')
            st.info(
                f"💡 **Multiplicador de Retorno:** Se detectaron **{len(detected_opportunities)} apuestas con ventaja matemática**. "
                "¡Pasa a la pestaña **'📱 Centro de Partido & Combinadas BetPlay'** para ver las combinadas sugeridas (Same-Game Parlay)!"
            )


    except Exception as e:
        st.error("⚠️ Error en pestaña Cartelera General (+EV Finder): " + str(e))

# TAB 2: CONTABILIDAD, BANKROLL Y AUDITORÍA (SQLite + Plotly)
# ------------------------------------------------------------------------------
with tab_audit:
    try:

        st.subheader("📈 Auditoría, Control de Bankroll y Rendimiento")
        st.markdown("Seguimiento del historial de operaciones, cálculo de Yield, ROI y evolución patrimonial.")

        metrics = get_bankroll_metrics(initial_bankroll=bankroll_val)

        # Top KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        kpi1.metric("Bankroll Inicial", format_money(metrics['initial_bankroll'], selected_curr_code))
        kpi2.metric(
            "Bankroll Actual",
            format_money(metrics['current_bankroll'], selected_curr_code),
            delta=f"{metrics['net_profit']:+,.0f} {selected_curr_code}" if selected_curr_code == "COP" else f"{metrics['net_profit']:+,.2f} {selected_curr_code}",
        )
        kpi3.metric("Beneficio Neto", format_money(metrics['net_profit'], selected_curr_code))
        kpi4.metric("Yield", f"{metrics['yield_pct']:.2f}%")
        kpi5.metric("ROI", f"{metrics['roi_pct']:.2f}%")
        kpi6.metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%")

        st.markdown("---")

        # Plotly Charts
        history_df = metrics.get("history_df")
        if history_df is not None and not history_df.empty:
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("##### 📈 Curva de Evolución del Bankroll ($)")
                fig_curve = px.line(
                    history_df,
                    x="id",
                    y="bankroll_curve",
                    labels={"id": "Operación #", "bankroll_curve": "Bankroll ($)"},
                    title="Crecimiento Patrimonial Acumulado",
                    markers=True,
                )
                fig_curve.add_hline(
                    y=metrics["initial_bankroll"],
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="Bankroll Inicial",
                )
                fig_curve.update_traces(line_color="#10b981", line_width=3)
                fig_curve.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_curve, use_container_width=True)

            with col_chart2:
                st.markdown("##### 📉 Drawdown de la Cartera (%)")
                fig_dd = px.area(
                    history_df,
                    x="id",
                    y="drawdown_pct",
                    labels={"id": "Operación #", "drawdown_pct": "Drawdown (%)"},
                    title="Caída Máxima desde el Pico (Drawdown)",
                )
                fig_dd.update_traces(line_color="#ef4444", fillcolor="rgba(239, 68, 68, 0.2)")
                fig_dd.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.info("ℹ️ Cierra al menos una apuesta como 'Ganada' o 'Perdida' para visualizar la curva de Bankroll y Drawdown.")

        st.markdown("---")
        st.markdown("#### 📋 Historial de Apuestas Registradas en SQLite")

        all_bets_df = get_all_bets()
        if all_bets_df.empty:
            st.info("No hay apuestas registradas aún en `quantshield.db`. Registra apuestas desde el Escáner o añade una manualmente abajo.")
        else:
            st.caption("Actualiza el estado de cualquier apuesta para recalcular automáticamente las ganancias y el bankroll.")

            # Interactive updates
            for _, row in all_bets_df.iterrows():
                bet_id = int(row["id"])
                col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns([2.5, 1.5, 1.2, 1.5, 0.8])

                with col_b1:
                    st.markdown(
                        f"**#{bet_id} | {row['match']}** ({row['market']})\n\n"
                        f"<small>📅 {row['timestamp']} | 🏢 {row['bookmaker']} | 🏷️ {row['selection']}</small>",
                        unsafe_allow_html=True,
                    )

                with col_b2:
                    st.markdown(
                        f"Cuota: **{row['odds']:.2f}** | Stake: **${row['stake']:.2f}**\n\n"
                        f"<small>EV: +{row['ev_percent']:.2f}%</small>",
                        unsafe_allow_html=True,
                    )

                with col_b3:
                    status_options = ["Pendiente", "Ganada", "Perdida", "Reembolso / Push"]
                    curr_status_idx = status_options.index(row["status"]) if row["status"] in status_options else 0
                    new_status = st.selectbox(
                        "Estado",
                        options=status_options,
                        index=curr_status_idx,
                        key=f"status_select_{bet_id}",
                        label_visibility="collapsed",
                    )

                    if new_status != row["status"]:
                        update_bet_status(bet_id, new_status)
                        st.rerun()

                with col_b4:
                    profit = float(row["net_profit"])
                    color = "#10b981" if profit > 0 else ("#ef4444" if profit < 0 else "#94a3b8")
                    st.markdown(
                        f"<div style='color:{color}; font-weight:700; font-size:1.1em; padding-top:8px;'>"
                        f"{profit:+,.2f} USD"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with col_b5:
                    if st.button("🗑️", key=f"del_bet_{bet_id}", help="Eliminar apuesta"):
                        delete_bet(bet_id)
                        st.rerun()

                st.divider()

        # Manual Bet Insertion Expander
        with st.expander("➕ Registrar Apuesta Manualmente en SQLite"):
            with st.form("manual_bet_form"):
                c_m1, c_m2, c_m3 = st.columns(3)
                with c_m1:
                    m_sport = st.selectbox("Deporte", ["Fútbol", "Tenis", "Baloncesto", "Otro"])
                    m_tourn = st.text_input("Competición", value="Liga BetPlay")
                    m_match = st.text_input("Partido", value="Millonarios vs Santa Fe")
                with c_m2:
                    m_market = st.text_input("Mercado", value="Draw No Bet (Local)")
                    m_selection = st.text_input("Selección", value="Millonarios")
                    m_bm = st.selectbox("Casa de Apuestas", ["BetPlay", "Rushbet", "Wplay", "Bwin Colombia", "Codere", "Betsson", "Yajuego", "Otra"])
                with c_m3:
                    m_odds = st.number_input("Cuota", min_value=1.01, value=1.85, step=0.01)
                    m_ev = st.number_input("EV Estimado (%)", min_value=0.0, value=4.5, step=0.1)
                    m_stake = st.number_input("Stake Invertido ($)", min_value=1.0, value=25.0, step=5.0)

                m_status = st.selectbox("Estado Inicial", ["Pendiente", "Ganada", "Perdida", "Reembolso / Push"])
                submitted = st.form_submit_button("Guardar en SQLite")
                if submitted:
                    add_bet(
                        sport=m_sport,
                        tournament=m_tourn,
                        match=m_match,
                        market=m_market,
                        selection=m_selection,
                        bookmaker=m_bm,
                        odds=m_odds,
                        fair_odds=None,
                        ev_percent=m_ev,
                        stake=m_stake,
                        status=m_status,
                    )
                    st.success("¡Apuesta registrada manualmente con éxito!")
                    st.rerun()


    except Exception as e:
        st.error("⚠️ Error en pestaña Contabilidad & Auditoría (SQLite): " + str(e))

# ------------------------------------------------------------------------------
# TAB 3: ALERTAS MÓVILES (Telegram Bot)
# ------------------------------------------------------------------------------
with tab_telegram:
    try:

        st.subheader("📲 Alertas Móviles Instantáneas vía Telegram Bot")
        st.markdown(
            "Recibe en tu celular las oportunidades con Esperanza Matemática Positiva detectadas por QuantShield en tiempo real."
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tg_token_input = st.text_input(
                "Telegram Bot Token",
                value=env_tg_token if env_tg_token != "your_telegram_bot_token_here" else "",
                type="password",
                help="Token otorgado por @BotFather al crear tu bot de Telegram.",
            )
        with col_t2:
            tg_chat_id_input = st.text_input(
                "Telegram Chat ID",
                value=env_tg_chat_id if env_tg_chat_id != "your_telegram_chat_id_here" else "",
                help="Tu ID numérico o el del canal/grupo (puedes obtenerlo enviando un mensaje a @userinfobot).",
            )

        # Validate chat_id: must be numeric or start with - (group/channel IDs can be negative)
        _tg_chat_id_clean = tg_chat_id_input.strip()
        _tg_chat_id_valid = bool(re.fullmatch(r"-?\d+", _tg_chat_id_clean)) if _tg_chat_id_clean else False
        if _tg_chat_id_clean and not _tg_chat_id_valid:
            st.warning("⚠️ El Telegram Chat ID debe ser un número entero (positivo o negativo). Ej: `123456789` o `-100123456789`.")

        # Test Telegram connection
        col_tb1, col_tb2 = st.columns(2)
        with col_tb1:
            if st.button("🧪 Probar Conexión con Telegram", use_container_width=True):
                if not tg_token_input or not _tg_chat_id_valid:
                    st.error("Por favor ingresa un Bot Token y un Chat ID numérico válido para probar la conexión.")
                else:
                    test_url = f"https://api.telegram.org/bot{tg_token_input}/sendMessage"
                    test_payload = {
                        "chat_id": _tg_chat_id_clean,
                        "text": "🛡️ *QuantShield Engine*: ¡Conexión establecida exitosamente con tu Bot de Alertas!",
                        "parse_mode": "Markdown",
                    }
                    try:
                        resp = requests.post(test_url, json=test_payload, timeout=8)
                        if resp.status_code == 200:
                            st.success("✅ ¡Mensaje de prueba enviado exitosamente a tu Telegram!")
                        else:
                            st.error(f"Fallo al enviar (Código {resp.status_code}): Verifica tu token y chat ID.")
                    except Exception as e:
                        st.error(f"Error de red: {str(e)}")

        with col_tb2:
            send_signals_btn = st.button("🚀 Enviar Señales Activas a mi Telegram", use_container_width=True)

        if send_signals_btn:
            if not tg_token_input or not _tg_chat_id_valid:
                st.error("Ingresa un Bot Token y un Chat ID numérico válido antes de emitir las alertas.")
            elif not detected_opportunities:
                st.warning("No hay oportunidades +EV detectadas actualmente en el escáner para enviar.")
            else:
                with st.spinner("Formateando reporte y enviando señales..."):
                    header_msg = (
                        f"🛡️ *QUANTSHIELD +EV ALERTS* 🛡️\n"
                        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                        f"Competición: *{selected_meta.get('name', 'General')}*\n"
                        f"Oportunidades Detectadas: *{len(detected_opportunities)}*\n"
                        f"----------------------------------------\n\n"
                    )

                    body_lines = []
                    for idx, opp in enumerate(detected_opportunities[:8]):
                        body_lines.append(
                            f"🎯 *#{idx+1} {opp['match']}*\n"
                            f"🏆 Torneo: {opp['tournament']}\n"
                            f"📊 Mercado: *{opp['market']}*\n"
                            f"🏷️ Selección: `{opp['selection']}`\n"
                            f"🏢 Casa: *{opp['bookmaker']}*\n"
                            f"💰 Cuota: *{opp['odds']}* (Mínima: {opp['fair_odds']})\n"
                            f"📈 EV: *+{opp['ev_percent']}%* | Prob: {opp['effective_prob_pct']}%\n"
                            f"💵 Stake Sugerido: *${opp['recommended_stake']}* ({opp['recommended_stake_pct']}% Bankroll)\n"
                            f"----------------------------------------"
                        )

                    full_message = header_msg + "\n".join(body_lines)

                    tg_url = f"https://api.telegram.org/bot{tg_token_input}/sendMessage"
                    try:
                        resp = requests.post(
                            tg_url,
                            json={"chat_id": _tg_chat_id_clean, "text": full_message, "parse_mode": "Markdown"},
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            st.success(f"🚀 ¡{len(detected_opportunities)} señales enviadas exitosamente a Telegram!")
                        else:
                            st.error(f"Error al enviar a Telegram ({resp.status_code}): {resp.text}")
                    except Exception as e:
                        st.error(f"Error de red con Telegram: {str(e)}")


    except Exception as e:
        st.error("⚠️ Error en pestaña Alertas Móviles (Telegram): " + str(e))

# ------------------------------------------------------------------------------
# TAB 4: SIMULADOR HISTÓRICO (BACKTESTING)
# ------------------------------------------------------------------------------
with tab_simulator:
    try:

        st.subheader("🧪 Simulador de Rendimiento Histórico (Backtesting Walk-Forward)")
        st.markdown(
            "Evalúa empíricamente la efectividad de los modelos matemáticos (+EV, Poisson Bivariado y Draw No Bet / Over 2.0) "
            "sobre partidos reales ya disputados, simulando la evolución del Bankroll fecha a fecha sin sesgo de anticipación (*lookahead bias*)."
        )

        col_sim1, col_sim2, col_sim3, col_sim4 = st.columns([1.5, 2.0, 1.2, 1.3])

        with col_sim1:
            sim_bankroll = st.number_input(
                f"Bankroll Inicial ({selected_curr_code})",
                min_value=20000.0 if selected_curr_code == "COP" else 50.0,
                max_value=500000000.0 if selected_curr_code == "COP" else 10000000.0,
                value=float(bankroll_val),
                step=50000.0 if selected_curr_code == "COP" else 100.0,
                format="%.0f" if selected_curr_code == "COP" else "%.2f",
                help="Capital inicial de prueba para la simulación histórica.",
            )

        with col_sim2:
            selected_leagues = st.multiselect(
                "Ligas a Simular",
                options=list(AVAILABLE_BACKTEST_LEAGUES.keys()),
                default=["N1", "D2", "I2"],
                format_func=lambda x: AVAILABLE_BACKTEST_LEAGUES.get(x, x),
                help="Selecciona una o varias competiciones para combinar en el portafolio histórico.",
            )

        with col_sim3:
            sim_min_ev = st.slider(
                "Filtro EV Mínimo (%)",
                min_value=1.5,
                max_value=8.0,
                value=2.5,
                step=0.5,
                help="Umbral mínimo de Esperanza Matemática para autorizar la entrada.",
            )

        with col_sim4:
            stake_selection = st.selectbox(
                "Gestión de Stake",
                options=[
                    "Kelly Fraccionado (k=0.20)",
                    "Kelly Agresivo (k=0.30)",
                    "Stake Fijo (2.0%)",
                    "Stake Fijo (2.5%)",
                    "Stake Fijo (3.0%)",
                ],
                index=0,
            )

        # Parse stake selection
        if "k=0.20" in stake_selection:
            sim_stake_mode = "kelly"
            sim_kelly_k = 0.20
            sim_fixed_pct = 0.025
        elif "k=0.30" in stake_selection:
            sim_stake_mode = "kelly"
            sim_kelly_k = 0.30
            sim_fixed_pct = 0.03
        elif "2.0%" in stake_selection:
            sim_stake_mode = "fixed"
            sim_kelly_k = 0.20
            sim_fixed_pct = 0.02
        elif "2.5%" in stake_selection:
            sim_stake_mode = "fixed"
            sim_kelly_k = 0.20
            sim_fixed_pct = 0.025
        else:
            sim_stake_mode = "fixed"
            sim_kelly_k = 0.20
            sim_fixed_pct = 0.03

        col_btn_sim, col_opts1, col_opts2 = st.columns([1.5, 1.2, 1.2])
        with col_btn_sim:
            run_sim_btn = st.button("▶️ Ejecutar Simulación Histórica", use_container_width=True, type="primary")
        with col_opts1:
            inc_dnb = st.checkbox("Incluir Draw No Bet (Local)", value=True)
        with col_opts2:
            inc_over2 = st.checkbox("Incluir Over 2.0 Asiático", value=True)

        # Execution logic
        if run_sim_btn:
            if not selected_leagues:
                st.error("Por favor selecciona al menos una liga para ejecutar el backtesting.")
            else:
                with st.spinner("Cargando base de datos histórica y ejecutando modelo iterativo..."):
                    hist_matches = load_historical_matches(selected_leagues)
                    if hist_matches.empty:
                        st.warning("No se encontraron registros de partidos para las ligas seleccionadas en el repositorio de caché.")
                    else:
                        sim_results = run_historical_backtest(
                            matches_df=hist_matches,
                            initial_bankroll=sim_bankroll,
                            min_ev=sim_min_ev,
                            min_odds=1.35,
                            max_odds=odds_range[1],
                            stake_mode=sim_stake_mode,
                            kelly_k=sim_kelly_k,
                            fixed_stake_pct=sim_fixed_pct,
                            include_dnb=inc_dnb,
                            include_over2=inc_over2,
                        )
                        st.session_state["last_sim_results"] = sim_results

        # Render Simulation Results if available
        if "last_sim_results" in st.session_state:
            res = st.session_state["last_sim_results"]
            summary = res["summary"]
            bets_df = res["bets_df"]
            hist_df = res["history_df"]

            st.markdown("---")
            st.markdown("#### 📊 Balance y Métricas del Backtesting con Partidos Reales")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Bankroll Inicial", format_money(summary['initial_bankroll'], selected_curr_code))
            m2.metric(
                "Bankroll Final",
                format_money(summary['final_bankroll'], selected_curr_code),
                delta=f"{summary['net_profit']:+,.0f} {selected_curr_code}" if selected_curr_code == "COP" else f"{summary['net_profit']:+,.2f} {selected_curr_code}",
            )
            m3.metric("Beneficio Neto", format_money(summary['net_profit'], selected_curr_code))
            m4.metric("ROI", f"{summary['roi_pct']:+.2f}%")
            m5.metric("Yield", f"{summary['yield_pct']:.2f}%")
            m6.metric("Win Rate", f"{summary['win_rate_pct']:.1f}%")

            # Result breakdown
            c_br1, c_br2, c_br3, c_br4 = st.columns(4)
            c_br1.markdown(f"**Total Entradas:** `{summary['total_bets']}`")
            c_br2.markdown(f"✅ **Ganadas:** `{summary['won_bets']}` | ❌ **Perdidas:** `{summary['lost_bets']}`")
            c_br3.markdown(f"🔄 **Reembolsos / Push:** `{summary['push_bets']}`")
            c_br4.markdown(f"🚫 **Descartadas por Filtro:** `{summary['discarded_matches']}`")

            # Specific match inspector
            if not bets_df.empty:
                st.markdown("---")
                st.markdown("##### 🔍 Consultar un Partido Real Específico ('¿Cuánto hubiera ganado?'):")
                st.caption("Selecciona cualquier partido real disputado para ver el marcador final oficial, la recomendación de QuantShield y la ganancia exacta:")

                match_list = bets_df["partido"].unique().tolist()
                c_sel1, c_sel2 = st.columns([2.5, 1.5])
                with c_sel1:
                    chosen_match = st.selectbox(
                        "Elige un partido jugado:",
                        options=match_list,
                        key="inspect_backtest_match",
                        help="Lista de partidos reales donde se detectó ventaja matemática (+EV)."
                    )
                with c_sel2:
                    schalke_matches = [m for m in match_list if "schalke" in m.lower() or "hamburger" in m.lower()]
                    if schalke_matches:
                        quick_pick = st.selectbox("Acceso Rápido (Alemania):", options=schalke_matches, key="quick_pick_ger")
                        if quick_pick:
                            chosen_match = quick_pick

                match_entries = bets_df[bets_df["partido"] == chosen_match]
                for _, r in match_entries.iterrows():
                    res_color = "#10b981" if r["resultado"] == "Ganada" else ("#f59e0b" if r["resultado"] == "Reembolso / Push" else "#ef4444")
                    st.markdown(
                        f"""
                        <div style="background-color: #1e293b; border-left: 5px solid {res_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span style="font-weight: 700; font-size: 1.15rem; color: #f8fafc;">⚽ {r['partido']} — {r['fecha']}</span>
                                <span style="background-color: {res_color}; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 700;">
                                    {r['resultado'].upper()}
                                </span>
                            </div>
                            <div style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 8px;">
                                <b>Competición:</b> {r['liga']} | <b>Marcador Oficial Real:</b> <code style="font-size: 1.05rem; color: #38bdf8; font-weight: 700;">{r['marcador']}</code> | <b>Mercado Jugado:</b> {r['mercado']} @ Cuota <b>{r['cuota']}</b>
                            </div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: {res_color};">
                                💰 Stake Invertido: {format_money(r['stake'], selected_curr_code)} ➔ Ganancia Neta Obtenida: {format_money(r['beneficio'], selected_curr_code)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Plotly Evolution Chart
            if not hist_df.empty:
                st.markdown("##### 📈 Curva de Crecimiento del Capital Simulado (Walk-Forward)")
                fig_sim = px.line(
                    hist_df,
                    x="id",
                    y="bankroll",
                    labels={"id": "Operación #", "bankroll": f"Bankroll ({selected_curr_code})"},
                    title="Evolución del Bankroll Partido a Partido sobre Resultados Reales",
                    hover_data=["fecha", "partido", "mercado", "resultado", "beneficio"],
                    markers=True,
                )
                fig_sim.add_hline(
                    y=summary["initial_bankroll"],
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Capital Inicial",
                )
                fig_sim.update_traces(line_color="#10b981", line_width=3)
                fig_sim.update_layout(template="plotly_dark", height=360)
                st.plotly_chart(fig_sim, use_container_width=True)

                # Detailed bets table
                st.markdown("##### 📋 Registro Detallado Partido a Partido")
                table_to_show = bets_df.copy()
                table_to_show["Stake"] = table_to_show["stake"].apply(lambda x: format_money(x, selected_curr_code))
                table_to_show["Beneficio"] = table_to_show["beneficio"].apply(lambda x: format_money(x, selected_curr_code))
                table_to_show["Bankroll"] = table_to_show["bankroll"].apply(lambda x: format_money(x, selected_curr_code))

                display_cols = ["id", "fecha", "liga", "partido", "mercado", "marcador", "cuota", "ev_pct", "Stake", "resultado", "Beneficio", "Bankroll"]
                table_to_show = table_to_show[display_cols].rename(columns={
                    "id": "#",
                    "fecha": "Fecha",
                    "liga": "Competición",
                    "partido": "Partido",
                    "mercado": "Mercado",
                    "marcador": "Marcador Real",
                    "cuota": "Cuota",
                    "ev_pct": "EV (%)",
                    "resultado": "Resultado",
                })
                st.dataframe(table_to_show, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ No se generaron apuestas con los filtros actuales. Prueba relajando el EV mínimo o habilitando más mercados.")

    except Exception as e:
        st.error("⚠️ Error en pestaña Simulador Histórico (Backtesting): " + str(e))

