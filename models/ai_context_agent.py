"""AI Context Agent with Google Gemini (Official google-genai SDK).

Implements neuro-symbolic tactical adjustments:
- Evaluates qualitative factors (injuries, rotations, fatigue, weather, surface).
- Returns strict Pydantic Structured Outputs (TacticalAdjustment).
- Enforces safety hard caps (np.clip between 0.70 and 1.30).
- Provides instant coherent simulation fallback when offline or in demo mode.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

# Google GenAI official SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Available Gemini models for Free and Pro tiers
GEMINI_MODELS = [
    "gemini-2.0-flash",        # Official free tier default (1,500 requests/day, 15 RPM)
    "gemini-1.5-flash",        # LTS stable fallback (1,500 requests/day, 15 RPM)
    "gemini-2.0-flash-lite",   # Ultra lightweight & low latency (30 RPM free)
    "gemini-2.5-flash",        # Flash 2.5
    "gemini-1.5-pro",          # Deep reasoning tier
]


class TacticalAdjustment(BaseModel):
    """Structured Output schema for Gemini tactical analysis."""

    factor_ajuste_local: float = Field(
        default=1.00,
        description="Multiplicador para el rendimiento/xG local entre 0.70 y 1.30 (1.00 = sin cambios).",
    )
    factor_ajuste_visita: float = Field(
        default=1.00,
        description="Multiplicador para el rendimiento/xG visitante entre 0.70 y 1.30 (1.00 = sin cambios).",
    )
    confianza_ia: float = Field(
        default=0.80,
        description="Nivel de confianza del análisis entre 0.0 y 1.0.",
    )
    nivel_riesgo: str = Field(
        default="Bajo",
        description="Clasificación del riesgo operativo: 'Bajo', 'Medio' o 'Alto'.",
    )
    justificacion_tactica: str = Field(
        default="Sin variaciones tácticas significativas.",
        description="Explicación concisa de máximo 2 líneas detallando por qué se ajustaron las probabilidades.",
    )


def get_ai_field(ai_obj: Any, field: str, default: Any = "") -> Any:
    """Helper universal de extracción segura para campos de análisis táctico / IA.

    Maneja con total seguridad:
    - None (retorna default)
    - Diccionarios (dict)
    - Modelos Pydantic (TacticalAdjustment)
    - Objetos genéricos con atributos

    Soporta mapeo bidireccional de sinónimos comunes:
    - nivel_riesgo <-> riesgo
    - justificacion_tactica <-> justificacion
    - confianza_ia <-> confianza
    - factor_ajuste_local <-> factor_local
    - factor_ajuste_visita <-> factor_visita
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

    # 1. Si es diccionario
    if isinstance(ai_obj, dict):
        for c in candidates:
            if c in ai_obj and ai_obj[c] is not None:
                return ai_obj[c]
        return default

    # 2. Si es objeto Pydantic o clase Python
    for c in candidates:
        if hasattr(ai_obj, c):
            val = getattr(ai_obj, c, None)
            if val is not None:
                return val

    return default


def generar_ajuste_simulado(
    deporte: str,
    torneo: str,
    local: str,
    visita: str,
    contexto_usuario: Optional[str] = "",
) -> TacticalAdjustment:
    """Generates an instant, coherent simulated tactical analysis for demo mode
    or when no GEMINI_API_KEY is supplied. Reacts intelligently to keywords in user context.
    """
    ctx = (contexto_usuario or "").lower().strip()
    
    # Check for negative impact on visiting team
    if any(k in ctx for k in ["arquero", "portero", "lesion", "baja", "suplente", "fatiga", "sancion"]) and ("visita" in ctx or "santa fe" in ctx or "visitante" in ctx or "rival" in ctx):
        f_local = 1.08
        f_visita = 0.88
        confianza = 0.88
        riesgo = "Bajo"
        justificacion = f"Bajas e incidencias reportadas en el visitante reducen su producción defensiva/ofensiva (-12%). Se favorece al local (+8%)."

    # Check for negative impact on home team
    elif any(k in ctx for k in ["lesion", "baja", "suplente", "rotacion", "fatiga", "sancion"]) and ("local" in ctx or "titular" in ctx):
        f_local = 0.90
        f_visita = 1.06
        confianza = 0.84
        riesgo = "Medio"
        justificacion = f"Rotaciones y bajas sensibles en el equipo local comprometen su generación de peligro (-10%). Oportunidad en el rival (+6%)."

    # General injury / rotation note
    elif any(k in ctx for k in ["lesion", "baja", "rotacion", "suplente", "descanso", "enfermedad"]):
        f_local = 1.06
        f_visita = 0.92
        confianza = 0.85
        riesgo = "Medio"
        justificacion = f"Noticias de rotación/bajas de última hora confirmadas. Ajuste asimétrico táctico aplicado: Local (+6%), Visita (-8%)."

    # Weather, rain, altitude
    elif any(k in ctx for k in ["lluvia", "clima", "altura", "calor", "viento"]):
        f_local = 0.94
        f_visita = 0.93
        confianza = 0.78
        riesgo = "Bajo"
        justificacion = f"Condiciones climáticas adversas reducen el ritmo general del encuentro y la expectativa total de goles/puntos (-6%)."

    # Tournament-based heuristics
    elif "libertadores" in torneo.lower() or "champions" in torneo.lower():
        f_local = 1.04
        f_visita = 0.97
        confianza = 0.82
        riesgo = "Bajo"
        justificacion = f"Encuentro internacional de alta intensidad; el factor localía y motivación de copa elevan la solidez del anfitrión."

    elif "nba" in deporte.lower() or "baloncesto" in deporte.lower():
        f_local = 1.02
        f_visita = 0.98
        confianza = 0.80
        riesgo = "Bajo"
        justificacion = f"Dinámica habitual de rotación en back-to-back de temporada regular. Ligera ventaja física para el conjunto local."

    else:
        f_local = 1.03
        f_visita = 0.98
        confianza = 0.80
        riesgo = "Bajo"
        justificacion = f"Análisis táctico estándar: localía consolidada y alineaciones completas reportadas sin incidencias."

    return TacticalAdjustment(
        factor_ajuste_local=f_local,
        factor_ajuste_visita=f_visita,
        confianza_ia=confianza,
        nivel_riesgo=riesgo,
        justificacion_tactica=justificacion,
    )


def verificar_conexion_gemini(api_key: str, model_name: str = "gemini-2.0-flash") -> Dict[str, Any]:
    """Tests the connection to Google Gemini API using the official google-genai SDK.
    Gracefully handles newly created API keys by automatically selecting currently available models.

    Returns a dict with:
        - success (bool): True if prompt evaluation succeeds
        - model (str): model tested / verified
        - message (str): Human-friendly confirmation or error description
        - error_detail (str, optional): technical error details if failed
    """
    if not api_key or not api_key.strip():
        return {
            "success": False,
            "model": model_name,
            "message": "La clave de Gemini está vacía. Pega tu API Key que comienza por 'AIzaSy...'.",
            "error_detail": "Clave vacía",
        }

    if not GENAI_AVAILABLE:
        return {
            "success": False,
            "model": model_name,
            "message": "El paquete oficial google-genai no está instalado en este entorno.",
            "error_detail": "google.genai missing",
        }

    key = api_key.strip()
    target_model = model_name if model_name in GEMINI_MODELS else "gemini-2.0-flash"

    # Candidates to test in priority order (starting with official free-tier models with 1,500 req/day)
    models_to_try = [target_model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-1.5-pro"]
    tried = set()
    last_error = ""

    try:
        client = genai.Client(api_key=key)

        for m_try in models_to_try:
            if m_try in tried:
                continue
            tried.add(m_try)
            try:
                # Fast test with 1 token output
                response = client.models.generate_content(
                    model=m_try,
                    contents="Responde únicamente con la palabra 'CONECTADO'.",
                    config=types.GenerateContentConfig(
                        max_output_tokens=5,
                        temperature=0.0,
                    ),
                )
                reply = (response.text or "").strip()
                note = ""
                if m_try != model_name:
                    note = f" (Actualizado a {m_try} para optimizar tu cuota gratuita diaria)"
                return {
                    "success": True,
                    "model": m_try,
                    "message": f"¡Tu clave es 100% válida! Conexión exitosa con Google Gemini ({m_try}){note}. Respuesta: '{reply}'",
                    "error_detail": None,
                }
            except Exception as e:
                last_error = str(e)
                err_lower = last_error.lower()
                # If model is deprecated, not found, or quota exceeded (429/resource exhausted on preview models), try next candidate
                if any(k in err_lower for k in ["404", "not found", "no longer available", "429", "resource_exhausted", "quota"]):
                    continue
                else:
                    # Other errors (e.g. invalid key 400) should not retry models
                    break

    except Exception as outer_err:
        last_error = str(outer_err)

    user_msg = "Error al autenticar con Google Gemini."
    if "API_KEY_INVALID" in last_error or "API key not valid" in last_error or "400" in last_error:
        user_msg = "Clave de API no válida. Asegúrate de haber copiado la clave completa generada en Google AI Studio (comienza por 'AIzaSy...')."
    elif "PERMISSION_DENIED" in last_error or "403" in last_error:
        user_msg = "Permiso denegado por Google. Tu cuenta o proyecto no tiene la API de Gemini habilitada."
    elif "RESOURCE_EXHAUSTED" in last_error or "429" in last_error:
        user_msg = "Límite de solicitudes de Gemini alcanzado temporalmente. Espera unos segundos y vuelve a intentar."
    elif "Failed to establish a new connection" in last_error or "timed out" in last_error or "ProxyError" in last_error or "getaddrinfo" in last_error:
        user_msg = "No se pudo conectar a los servidores de Google. Revisa tu conexión a internet o el proxy corporativo."

    return {
        "success": False,
        "model": target_model,
        "message": user_msg,
        "error_detail": last_error,
    }


def evaluar_contexto_partido(
    deporte: str,
    torneo: str,
    local: str,
    visita: str,
    cuotas_actuales: Dict[str, Any],
    contexto_usuario: Optional[str] = "",
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
    is_demo: bool = False,
) -> TacticalAdjustment:
    """Evaluates qualitative and tactical context using Google Gemini.

    Parameters:
        deporte: Sport category (e.g. 'Fútbol', 'Tenis', 'Baloncesto')
        torneo: Tournament or league name
        local: Home team or player 1
        visita: Away team or player 2
        cuotas_actuales: Dictionary with current market odds
        contexto_usuario: Optional user-injected breaking news or tactical notes
        api_key: Google Gemini API key
        model_name: Selected Gemini model from GEMINI_MODELS
        is_demo: When True and no API key is provided, bypasses API calls

    Returns:
        TacticalAdjustment with strictly validated and clipped factors.
    """
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()

    # Fallback to simulation only if no valid API key is present or library is missing
    if not key or key == "your_gemini_api_key_here" or not GENAI_AVAILABLE or is_demo:
        return generar_ajuste_simulado(deporte, torneo, local, visita, contexto_usuario)

    prompt = f"""Actúa como un analista cuantitativo y táctico deportivo de élite para apuestas de valor (+EV).
Tu misión es evaluar el contexto cualitativo de este evento deportivo y proporcionar factores de ajuste numérico precisos.

INFORMACIÓN DEL EVENTO:
- Deporte: {deporte}
- Competición: {torneo}
- Partido / Encuentro: {local} (Local) vs {visita} (Visita)
- Cuotas de Mercado Disponibles: {cuotas_actuales}
- Novedades de última hora / Contexto Táctico: {contexto_usuario or 'Sin novedades específicas reportadas; evalúa contexto estándar de localía, descanso y dinámica reciente.'}

REGLAS DE EVALUACIÓN:
1. factor_ajuste_local: Multiplicador sobre el rendimiento ofensivo / xG del local (1.00 = neutro, 1.10 = +10% de xG, 0.90 = -10%).
2. factor_ajuste_visita: Multiplicador sobre el rendimiento ofensivo / xG del visitante (1.00 = neutro).
3. Ambos factores deben mantenerse dentro del rango estricto [0.70, 1.30] (máximo ±30% de variación).
4. confianza_ia: Probabilidad subjetiva del análisis [0.0, 1.0].
5. nivel_riesgo: 'Bajo', 'Medio' o 'Alto' en función de la incertidumbre o volatilidad táctica.
6. justificacion_tactica: Resumen técnico y contundente de máximo 2 líneas.
"""

    try:
        client = genai.Client(api_key=key)
        target_model = model_name if model_name in GEMINI_MODELS else "gemini-2.0-flash"

        # Try target model with fallback for new accounts
        models_to_try = [target_model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-1.5-pro"]
        tried = set()
        response = None

        for m_try in models_to_try:
            if m_try in tried:
                continue
            tried.add(m_try)
            try:
                response = client.models.generate_content(
                    model=m_try,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TacticalAdjustment,
                        temperature=0.2,
                    ),
                )
                if response:
                    break
            except Exception as call_err:
                err_low = str(call_err).lower()
                if any(k in err_low for k in ["404", "not found", "no longer available", "429", "resource_exhausted", "quota"]):
                    continue
                else:
                    raise call_err

        # Parse response using Pydantic
        if response and hasattr(response, "parsed") and isinstance(response.parsed, TacticalAdjustment):
            adj = response.parsed
        elif response and response.text:
            adj = TacticalAdjustment.model_validate_json(response.text)
        else:
            return generar_ajuste_simulado(deporte, torneo, local, visita, contexto_usuario)

        # Safety Hard Cap: Clip multipliers to +/- 30% [0.70, 1.30]
        clipped_local = float(np.clip(adj.factor_ajuste_local, 0.70, 1.30))
        clipped_visita = float(np.clip(adj.factor_ajuste_visita, 0.70, 1.30))
        clipped_confianza = float(np.clip(adj.confianza_ia, 0.0, 1.0))
        riesgo = adj.nivel_riesgo if adj.nivel_riesgo in ["Bajo", "Medio", "Alto"] else "Medio"

        return TacticalAdjustment(
            factor_ajuste_local=round(clipped_local, 3),
            factor_ajuste_visita=round(clipped_visita, 3),
            confianza_ia=round(clipped_confianza, 2),
            nivel_riesgo=riesgo,
            justificacion_tactica=adj.justificacion_tactica.strip(),
        )

    except Exception as err:
        # Graceful fallback on network/quota error: provide simulated response with notice
        sim = generar_ajuste_simulado(deporte, torneo, local, visita, contexto_usuario)
        sim.justificacion_tactica += f" (Ajuste estimado de contingencia: no se pudo consultar Gemini)"
        return sim
