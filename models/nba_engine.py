"""Basketball / NBA Engine for QuantShield.

Evaluates binary totals (Over/Under) and spreads (Handicap) using
Gaussian Normal Distribution (scipy.stats.norm) and Pinnacle consensus.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from scipy.stats import norm
from models.kelly_risk import calculate_kelly_stake, evaluate_opportunity_filters

# Standard deviations of points in basketball
NBA_TOTALS_STD = 11.5
NBA_SPREADS_STD = 11.0
EUROLEAGUE_TOTALS_STD = 9.5
EUROLEAGUE_SPREADS_STD = 9.0


def extract_pinnacle_baseline(bookmakers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Finds Pinnacle (or consensus) benchmark lines for Totals and Spreads."""
    pinnacle = next((b for b in bookmakers if b.get("key") == "pinnacle"), None)
    target = pinnacle or (bookmakers[0] if bookmakers else None)
    if not target:
        return {}

    baseline: Dict[str, Any] = {"totals": None, "spreads": None}
    for m in target.get("markets", []):
        if m.get("key") == "totals" and not baseline["totals"]:
            outs = m.get("outcomes", [])
            if outs:
                point = float(outs[0].get("point", 0.0))
                baseline["totals"] = point
        elif m.get("key") == "spreads" and not baseline["spreads"]:
            outs = m.get("outcomes", [])
            for out in outs:
                name = out.get("name", "")
                point = float(out.get("point", 0.0))
                baseline["spreads"] = {"team": name, "point": point}
                break
    return baseline


def analyze_nba_match(
    event: Dict[str, Any],
    sport_meta: Dict[str, Any],
    bankroll: float = 1000.0,
    kelly_k: float = 0.20,
    min_ev: float = 3.0,
    min_odds: float = 1.40,
    max_odds: float = 2.25,
    ai_adjustment: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Analyzes NBA / Euroleague match for +EV opportunities on Totals and Spreads.
    Applies Gemini AI tactical adjustments if provided.
    """
    opportunities: List[Dict[str, Any]] = []
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return opportunities

    home = event.get("home_team", "Local")
    away = event.get("away_team", "Visita")
    match_name = f"{home} vs {away}"

    is_euroleague = "euroleague" in sport_meta.get("sport_key", "").lower()
    std_totals = EUROLEAGUE_TOTALS_STD if is_euroleague else NBA_TOTALS_STD
    std_spreads = EUROLEAGUE_SPREADS_STD if is_euroleague else NBA_SPREADS_STD

    baseline = extract_pinnacle_baseline(bookmakers)
    base_total = baseline.get("totals", 220.0)

    # Adjust base total with Gemini AI pace and scoring factor
    if ai_adjustment is not None:
        f_local = getattr(ai_adjustment, "factor_ajuste_local", 1.0)
        f_visita = getattr(ai_adjustment, "factor_ajuste_visita", 1.0)
        pace_factor = (f_local + f_visita) / 2.0
        base_total = round(base_total * pace_factor, 1)

    # Evaluate recreational bookmakers
    for bm in bookmakers:
        bm_key = bm.get("key", "")
        bm_title = bm.get("title", bm_key.capitalize())
        if bm_key == "pinnacle":
            continue
        # Filtrar exclusivamente casas autorizadas en Colombia
        if not any(h in bm_title for h in ("BetPlay", "Rushbet", "Wplay", "Bwin", "Codere", "Betsson", "Yajuego", "Rivalo", "Mozzart")):
            continue

        for m in bm.get("markets", []):
            # 1. TOTALS (Over / Under)
            if m.get("key") == "totals":
                for out in m.get("outcomes", []):
                    name = out.get("name", "")
                    point = float(out.get("point", base_total))
                    price = float(out.get("price", 0.0))

                    if price <= 1.0:
                        continue

                    # Gaussian calculation:
                    # Implied mean = base_total
                    if name.lower() == "over":
                        # P(Total > point) = 1 - CDF(point, base_total, std)
                        prob = float(norm.sf(point, loc=base_total, scale=std_totals))
                    else:
                        # P(Total < point) = CDF(point, base_total, std)
                        prob = float(norm.cdf(point, loc=base_total, scale=std_totals))

                    if prob <= 0.0 or prob >= 1.0:
                        continue

                    fair_odds = 1.0 / prob
                    ev = (prob * price) - 1.0
                    ev_pct = ev * 100.0

                    passes, _ = evaluate_opportunity_filters(
                        odds=price,
                        ev_percent=ev_pct,
                        min_ev=min_ev,
                        min_odds=min_odds,
                        max_odds=max_odds,
                    )

                    if passes:
                        kelly_res = calculate_kelly_stake(
                            odds=price,
                            effective_prob=prob,
                            bankroll=bankroll,
                            kelly_fraction=kelly_k,
                        )
                        stake = kelly_res["recommended_stake"]

                        instructions = (
                            f"1. Accede a tu casa de apuestas ({bm_title}).\n"
                            f"2. Busca el partido de Baloncesto '{match_name}'.\n"
                            f"3. Entra a la pestaña 'Totales / Puntos del Partido'.\n"
                            f"4. Selecciona '{name} {point}' puntos a cuota actual {price} (Mínimo aceptable: {round(fair_odds, 2)}).\n"
                            f"5. Coloca un stake de ${stake} ({round(kelly_res['recommended_stake_pct'] * 100, 1)}% del bankroll).\n"
                            f"6. Modelo Gaussiano: Media esperada {base_total} pts, Desv. Estándar {std_totals} pts. Probabilidad de acierto: {round(prob * 100, 1)}%."
                        )

                        opportunities.append({
                            "sport": "Baloncesto",
                            "tournament": sport_meta.get("name", "Baloncesto"),
                            "match": match_name,
                            "commence_time": event.get("commence_time", ""),
                            "market": f"Total Puntos ({name})",
                            "selection": f"{name} {point} pts",
                            "bookmaker": bm_title,
                            "odds": round(price, 2),
                            "fair_odds": round(fair_odds, 2),
                            "effective_prob_pct": round(prob * 100.0, 1),
                            "ev_percent": round(ev_pct, 2),
                            "recommended_stake": stake,
                            "recommended_stake_pct": round(kelly_res["recommended_stake_pct"] * 100.0, 2),
                            "is_capped": kelly_res["is_capped"],
                            "instructions": instructions,
                            "dutching": None,
                            "lambdas": f"Línea Base: {base_total} | σ: {std_totals}",
                            "ai_adjustment": {
                                "factor_local": getattr(ai_adjustment, "factor_ajuste_local", 1.0),
                                "factor_visita": getattr(ai_adjustment, "factor_ajuste_visita", 1.0),
                                "confianza": getattr(ai_adjustment, "confianza_ia", 0.8),
                                "riesgo": getattr(ai_adjustment, "nivel_riesgo", "Bajo"),
                                "justificacion": getattr(ai_adjustment, "justificacion_tactica", ""),
                            } if ai_adjustment is not None else None,
                        })

            # 2. SPREADS (Handicap)
            elif m.get("key") == "spreads":
                for out in m.get("outcomes", []):
                    team_name = out.get("name", "")
                    spread = float(out.get("point", 0.0))
                    price = float(out.get("price", 0.0))

                    if price <= 1.0:
                        continue

                    # If base spread available for this team
                    base_spread = 0.0
                    if baseline.get("spreads") and baseline["spreads"]["team"] == team_name:
                        base_spread = baseline["spreads"]["point"]

                    # Cover probability via Gaussian CDF
                    diff = spread - base_spread
                    # Adjust margin with AI factors (up to +/- 3 points)
                    if ai_adjustment is not None:
                        f_local = getattr(ai_adjustment, "factor_ajuste_local", 1.0)
                        f_visita = getattr(ai_adjustment, "factor_ajuste_visita", 1.0)
                        shift = (f_local - f_visita) * 3.5
                        if team_name == home:
                            diff += shift
                        else:
                            diff -= shift

                    # Every +1 point in spread gives ~3.6% extra probability in NBA around mean
                    prob = float(norm.cdf(diff / std_spreads, loc=0.0, scale=1.0) * 0.5 + 0.5)
                    prob = max(0.10, min(prob, 0.90))

                    fair_odds = 1.0 / prob
                    ev = (prob * price) - 1.0
                    ev_pct = ev * 100.0

                    passes, _ = evaluate_opportunity_filters(
                        odds=price,
                        ev_percent=ev_pct,
                        min_ev=min_ev,
                        min_odds=min_odds,
                        max_odds=max_odds,
                    )

                    if passes:
                        kelly_res = calculate_kelly_stake(
                            odds=price,
                            effective_prob=prob,
                            bankroll=bankroll,
                            kelly_fraction=kelly_k,
                        )
                        stake = kelly_res["recommended_stake"]

                        instructions = (
                            f"1. Accede a tu casa de apuestas ({bm_title}).\n"
                            f"2. Localiza '{match_name}'.\n"
                            f"3. En 'Hándicap / Spread', selecciona '{team_name} ({spread:+.1f})' a cuota {price}.\n"
                            f"4. Cuota justa estimada: {round(fair_odds, 2)}. EV estimado: +{round(ev_pct, 2)}%.\n"
                            f"5. Apuesta recomendada: ${stake} ({round(kelly_res['recommended_stake_pct'] * 100, 1)}% del bankroll)."
                        )

                        opportunities.append({
                            "sport": "Baloncesto",
                            "tournament": sport_meta.get("name", "Baloncesto"),
                            "match": match_name,
                            "commence_time": event.get("commence_time", ""),
                            "market": "Hándicap / Spread",
                            "selection": f"{team_name} ({spread:+.1f})",
                            "bookmaker": bm_title,
                            "odds": round(price, 2),
                            "fair_odds": round(fair_odds, 2),
                            "effective_prob_pct": round(prob * 100.0, 1),
                            "ev_percent": round(ev_pct, 2),
                            "recommended_stake": stake,
                            "recommended_stake_pct": round(kelly_res["recommended_stake_pct"] * 100.0, 2),
                            "is_capped": kelly_res["is_capped"],
                            "instructions": instructions,
                            "dutching": None,
                            "lambdas": f"Línea Spread: {spread:+.1f} | σ: {std_spreads}",
                            "ai_adjustment": {
                                "factor_local": getattr(ai_adjustment, "factor_ajuste_local", 1.0),
                                "factor_visita": getattr(ai_adjustment, "factor_ajuste_visita", 1.0),
                                "confianza": getattr(ai_adjustment, "confianza_ia", 0.8),
                                "riesgo": getattr(ai_adjustment, "nivel_riesgo", "Bajo"),
                                "justificacion": getattr(ai_adjustment, "justificacion_tactica", ""),
                            } if ai_adjustment is not None else None,
                        })

    return opportunities
