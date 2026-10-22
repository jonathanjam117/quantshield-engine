"""Tennis Engine for QuantShield.

Evaluates binary head-to-head (Match Winner) markets:
- Calculates Pinnacle Vig-Free clean probabilities for Player 1 and Player 2.
- Compares recreational bookmaker odds against fair oracle probability.
- Computes EV, Kelly stake with 3.5% cap, and step-by-step instructions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from models.kelly_risk import calculate_kelly_stake, evaluate_opportunity_filters


def extract_vig_free_tennis(bookmaker_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extracts vig-free 2-way odds and probabilities for tennis players."""
    for m in bookmaker_dict.get("markets", []):
        if m.get("key") == "h2h":
            outcomes = m.get("outcomes", [])
            if len(outcomes) == 2:
                p1_name = outcomes[0].get("name", "")
                p1_price = float(outcomes[0].get("price", 0.0))
                p2_name = outcomes[1].get("name", "")
                p2_price = float(outcomes[1].get("price", 0.0))

                if p1_price <= 1.0 or p2_price <= 1.0:
                    return None

                inv1 = 1.0 / p1_price
                inv2 = 1.0 / p2_price
                tot_inv = inv1 + inv2

                p1_clean = inv1 / tot_inv
                p2_clean = inv2 / tot_inv

                return {
                    "player_1": p1_name,
                    "price_1": p1_price,
                    "prob_1": p1_clean,
                    "player_2": p2_name,
                    "price_2": p2_price,
                    "prob_2": p2_clean,
                }
    return None


def analyze_tennis_match(
    event: Dict[str, Any],
    sport_meta: Dict[str, Any],
    bankroll: float = 1000.0,
    kelly_k: float = 0.20,
    min_ev: float = 3.0,
    min_odds: float = 1.40,
    max_odds: float = 2.25,
    ai_adjustment: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Analyzes a tennis match for +EV opportunities on match winner.
    Applies Gemini AI qualitative weighting if provided.
    """
    opportunities: List[Dict[str, Any]] = []
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return opportunities

    home = event.get("home_team", "Jugador 1")
    away = event.get("away_team", "Jugador 2")
    match_name = f"{home} vs {away}"

    # Locate Pinnacle as oracle
    pinnacle_bm = next((b for b in bookmakers if b.get("key") == "pinnacle"), None)
    oracle_data = None
    if pinnacle_bm:
        oracle_data = extract_vig_free_tennis(pinnacle_bm)

    # Fallback to consensus if Pinnacle missing
    if not oracle_data:
        for b in bookmakers:
            od = extract_vig_free_tennis(b)
            if od:
                oracle_data = od
                break

    if not oracle_data:
        return opportunities

    p1_clean = oracle_data["prob_1"]
    p2_clean = oracle_data["prob_2"]

    # Apply AI tactical weighting (surface adaptation, fatigue, recent form)
    if ai_adjustment is not None:
        f1 = getattr(ai_adjustment, "factor_ajuste_local", 1.0)
        f2 = getattr(ai_adjustment, "factor_ajuste_visita", 1.0)
        w1 = p1_clean * f1
        w2 = p2_clean * f2
        tot_w = w1 + w2
        p1_final = w1 / tot_w if tot_w > 0 else p1_clean
        p2_final = w2 / tot_w if tot_w > 0 else p2_clean
    else:
        p1_final = p1_clean
        p2_final = p2_clean

    players_oracle = {
        oracle_data["player_1"]: p1_final,
        oracle_data["player_2"]: p2_final,
    }

    # Compare against recreational books
    for bm in bookmakers:
        bm_key = bm.get("key", "")
        bm_title = bm.get("title", bm_key.capitalize())
        if bm_key == "pinnacle":
            continue

        for m in bm.get("markets", []):
            if m.get("key") == "h2h":
                for out in m.get("outcomes", []):
                    player_name = out.get("name", "")
                    rec_price = float(out.get("price", 0.0))

                    if rec_price <= 1.0 or player_name not in players_oracle:
                        continue

                    p_fair = players_oracle[player_name]
                    fair_odds = 1.0 / p_fair if p_fair > 0 else 999.0

                    # EV = (P_clean * Cuota_rec) - 1
                    ev = (p_fair * rec_price) - 1.0
                    ev_pct = ev * 100.0

                    passes, _ = evaluate_opportunity_filters(
                        odds=rec_price,
                        ev_percent=ev_pct,
                        min_ev=min_ev,
                        min_odds=min_odds,
                        max_odds=max_odds,
                    )

                    if passes:
                        kelly_res = calculate_kelly_stake(
                            odds=rec_price,
                            effective_prob=p_fair,
                            bankroll=bankroll,
                            kelly_fraction=kelly_k,
                        )
                        stake = kelly_res["recommended_stake"]

                        instructions = (
                            f"1. Abre tu cuenta en {bm_title}.\n"
                            f"2. Localiza el torneo '{sport_meta.get('name', 'ATP')}' y el partido '{match_name}'.\n"
                            f"3. Selecciona 'Ganador del Partido' a favor de '{player_name}'.\n"
                            f"4. Cuota tomada: {rec_price} (Cuota mínima aceptable: {round(fair_odds, 2)}).\n"
                            f"5. Coloca un stake de ${stake} ({round(kelly_res['recommended_stake_pct'] * 100, 1)}% del bankroll).\n"
                            f"6. Valor matemático estimado: +{round(ev_pct, 2)}% frente a cuotas sin margen de Pinnacle."
                        )

                        opportunities.append({
                            "sport": "Tenis",
                            "tournament": sport_meta.get("name", "Tenis"),
                            "match": match_name,
                            "commence_time": event.get("commence_time", ""),
                            "market": "Ganador de Partido (H2H)",
                            "selection": player_name,
                            "bookmaker": bm_title,
                            "odds": round(rec_price, 2),
                            "fair_odds": round(fair_odds, 2),
                            "effective_prob_pct": round(p_fair * 100.0, 1),
                            "ev_percent": round(ev_pct, 2),
                            "recommended_stake": stake,
                            "recommended_stake_pct": round(kelly_res["recommended_stake_pct"] * 100.0, 2),
                            "is_capped": kelly_res["is_capped"],
                            "instructions": instructions,
                            "dutching": None,
                            "lambdas": f"Prob. Oráculo: {round(p_fair*100, 1)}%",
                            "ai_adjustment": {
                                "factor_local": getattr(ai_adjustment, "factor_ajuste_local", 1.0),
                                "factor_visita": getattr(ai_adjustment, "factor_ajuste_visita", 1.0),
                                "confianza": getattr(ai_adjustment, "confianza_ia", 0.8),
                                "riesgo": getattr(ai_adjustment, "nivel_riesgo", "Bajo"),
                                "justificacion": getattr(ai_adjustment, "justificacion_tactica", ""),
                            } if ai_adjustment is not None else None,
                        })

    return opportunities
