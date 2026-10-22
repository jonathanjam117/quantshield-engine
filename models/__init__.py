"""Mathematical and risk modeling modules for QuantShield."""
from .soccer_engine import analyze_soccer_match, calculate_poisson_matrix, calculate_dutching_saver
from .tennis_engine import analyze_tennis_match
from .nba_engine import analyze_nba_match
from .kelly_risk import calculate_kelly_stake, evaluate_opportunity_filters
from .ai_context_agent import (
    TacticalAdjustment,
    evaluar_contexto_partido,
    verificar_conexion_gemini,
    GEMINI_MODELS,
)
from .backtest_simulator import run_historical_backtest, load_historical_matches, AVAILABLE_BACKTEST_LEAGUES

__all__ = [
    "analyze_soccer_match",
    "calculate_poisson_matrix",
    "calculate_dutching_saver",
    "analyze_tennis_match",
    "analyze_nba_match",
    "calculate_kelly_stake",
    "evaluate_opportunity_filters",
    "TacticalAdjustment",
    "evaluar_contexto_partido",
    "verificar_conexion_gemini",
    "GEMINI_MODELS",
    "run_historical_backtest",
    "load_historical_matches",
    "AVAILABLE_BACKTEST_LEAGUES",
]
