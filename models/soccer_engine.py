"""Soccer Engine for QuantShield.

Implements:
1. Vig-Free probability extraction from Pinnacle / Market Consensus.
2. 9x9 Bivariate Poisson probability matrix via scipy.stats.poisson.
3. Bayesian blending: 60% Poisson + 40% Oracle.
4. Low-variance protected markets:
   - Draw No Bet (Apuesta sin Empate) Local with synthetic odds calculation.
   - Asian Total Goals Over 2.0 (refund on exactly 2 goals).
5. Synthetic Insurance Dutching Saver calculation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import poisson

from etl.historical_loader import load_historical_forces
from models.ai_context_agent import get_ai_field
from models.kelly_risk import calculate_kelly_stake, evaluate_opportunity_filters


def extract_vig_free_h2h(bookmaker_dict: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extracts vig-free (no-margin) probabilities for Home, Draw, Away (1, X, 2)
    from a bookmaker market using formula:
    P_oraculo = (1 / Cuota) / sum(1 / Cuota_i)
    """
    for m in bookmaker_dict.get("markets", []):
        if m.get("key") == "h2h":
            outcomes = m.get("outcomes", [])
            if len(outcomes) >= 3:
                # Find home, draw, away
                prices = {}
                for o in outcomes:
                    name = o.get("name", "").strip()
                    price = float(o.get("price", 0.0))
                    if price <= 1.0:
                        return None
                    prices[name] = price

                # Identify draw
                draw_key = next((k for k in prices.keys() if "draw" in k.lower() or "empate" in k.lower() or k == "X"), None)
                if not draw_key:
                    return None

                # Non-draw keys
                team_keys = [k for k in prices.keys() if k != draw_key]
                if len(team_keys) < 2:
                    return None

                k_home, k_away = team_keys[0], team_keys[1]
                inv_h = 1.0 / prices[k_home]
                inv_x = 1.0 / prices[draw_key]
                inv_a = 1.0 / prices[k_away]
                total_inv = inv_h + inv_x + inv_a

                return {
                    "home_team": k_home,
                    "away_team": k_away,
                    "draw_key": draw_key,
                    "odds_h": prices[k_home],
                    "odds_x": prices[draw_key],
                    "odds_a": prices[k_away],
                    "p_clean_h": inv_h / total_inv,
                    "p_clean_x": inv_x / total_inv,
                    "p_clean_a": inv_a / total_inv,
                }
    return None


def resolve_event_lambdas(
    event: Dict[str, Any],
    sport_meta: Optional[Dict[str, Any]] = None,
    ai_adjustment: Optional[Any] = None,
) -> Tuple[float, float, str]:
    """Resolves and caches dynamic Poisson lambdas (lambda_h, lambda_a) for an event
    using Football-Data CSV, Oracle vig-free probabilities, or league averages,
    adjusted by Gemini AI if provided.
    """
    if "lambda_h" in event and "lambda_a" in event:
        return float(event["lambda_h"]), float(event["lambda_a"]), str(event.get("lambda_source", "Cache"))

    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    bookmakers = event.get("bookmakers", [])

    oracle_data = None
    if bookmakers:
        pinnacle_bm = next((b for b in bookmakers if b.get("key") == "pinnacle"), None)
        if pinnacle_bm:
            oracle_data = extract_vig_free_h2h(pinnacle_bm)
        if not oracle_data:
            for b in bookmakers:
                od = extract_vig_free_h2h(b)
                if od:
                    oracle_data = od
                    break

    oracle_probs = (oracle_data["p_clean_h"], oracle_data["p_clean_x"], oracle_data["p_clean_a"]) if oracle_data else None
    csv_code = sport_meta.get("csv_code") if sport_meta else None

    lh, la, src = load_historical_forces(
        csv_code=csv_code,
        home_team=home_team,
        away_team=away_team,
        oracle_probs=oracle_probs,
    )

    # Enforce market reality: if market odds clearly favor away or home, lambdas MUST match
    if oracle_data:
        p_clean_h = float(oracle_data["p_clean_h"])
        p_clean_a = float(oracle_data["p_clean_a"])
        tot_lam = (lh + la) if (lh + la) > 0 else 2.60
        # If Away is favorite by market odds (p_clean_a > p_clean_h), away lambda must be higher
        if p_clean_a > p_clean_h and lh >= la:
            ratio = (p_clean_h / p_clean_a) ** 0.65
            lh = tot_lam * (ratio / (1.0 + ratio))
            la = tot_lam * (1.0 / (1.0 + ratio))
            src += " [Calibrado a Favorito Mercado Visita]"
        elif p_clean_h > p_clean_a and la >= lh:
            ratio = (p_clean_h / p_clean_a) ** 0.65
            lh = tot_lam * (ratio / (1.0 + ratio))
            la = tot_lam * (1.0 / (1.0 + ratio))
            src += " [Calibrado a Favorito Mercado Local]"

    if ai_adjustment is not None:
        f_h = float(get_ai_field(ai_adjustment, "factor_ajuste_local", 1.0))
        f_a = float(get_ai_field(ai_adjustment, "factor_ajuste_visita", 1.0))
        lh = float(np.clip(lh * f_h, 0.35, 4.50))
        la = float(np.clip(la * f_a, 0.35, 4.50))
        src += f" + IA Gemini (H: {f_h:.2f}x | A: {f_a:.2f}x)"

    lh = round(float(lh), 2)
    la = round(float(la), 2)
    event["lambda_h"] = lh
    event["lambda_a"] = la
    event["lambda_source"] = src
    return lh, la, src


def calculate_poisson_matrix(lambda_h: float, lambda_a: float, max_goals: int = 9) -> np.ndarray:
    """Generates a 9x9 Bivariate Poisson probability matrix (scores 0 to 8).
    M[i, j] = P(Home = i) * P(Away = j)
    """
    home_probs = [poisson.pmf(i, lambda_h) for i in range(max_goals)]
    away_probs = [poisson.pmf(j, lambda_a) for j in range(max_goals)]
    matrix = np.outer(home_probs, away_probs)
    # Normalize to ensure sum is exactly 1.0
    total = np.sum(matrix)
    if total > 0:
        matrix = matrix / total
    return matrix


def calculate_dutching_saver(
    total_investment: float,
    odds_home: float,
    odds_draw: float,
) -> Dict[str, Any]:
    """Calculates synthetic insurance breakdown:
    - Monto_Empate = Inversion_Total / Cuota_X
    - Monto_Local = Inversion_Total - Monto_Empate
    """
    if odds_draw <= 1.0 or odds_home <= 1.0 or total_investment <= 0:
        return {
            "stake_draw": 0.0,
            "stake_home": 0.0,
            "return_if_draw": 0.0,
            "profit_if_draw": 0.0,
            "return_if_home": 0.0,
            "net_profit_if_home": 0.0,
        }

    stake_draw = round(total_investment / odds_draw, 2)
    stake_home = round(total_investment - stake_draw, 2)
    return_if_draw = round(stake_draw * odds_draw, 2)
    profit_if_draw = round(return_if_draw - total_investment, 2)
    return_if_home = round(stake_home * odds_home, 2)
    net_profit_if_home = round(return_if_home - total_investment, 2)

    return {
        "stake_draw": stake_draw,
        "stake_home": stake_home,
        "return_if_draw": return_if_draw,
        "profit_if_draw": profit_if_draw,
        "return_if_home": return_if_home,
        "net_profit_if_home": net_profit_if_home,
    }


def analyze_soccer_match(
    event: Dict[str, Any],
    sport_meta: Dict[str, Any],
    bankroll: float = 1000.0,
    kelly_k: float = 0.20,
    min_ev: float = 3.0,
    min_odds: float = 1.40,
    max_odds: float = 2.25,
    ai_adjustment: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Analyzes a soccer match to detect +EV opportunities in low-variance markets:
    - Draw No Bet (Local)
    - Asian Total Goals (Over 2.0)
    Applies Gemini AI tactical adjustments if provided.
    """
    opportunities: List[Dict[str, Any]] = []
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return opportunities

    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    match_name = f"{home_team} vs {away_team}"

    # 1. Look for Pinnacle or Market Consensus for Oracle Probabilities
    pinnacle_bm = next((b for b in bookmakers if b.get("key") == "pinnacle"), None)
    oracle_data = None
    if pinnacle_bm:
        oracle_data = extract_vig_free_h2h(pinnacle_bm)

    # Fallback to first bookmaker with valid h2h if Pinnacle not present
    if not oracle_data:
        for b in bookmakers:
            od = extract_vig_free_h2h(b)
            if od:
                oracle_data = od
                break

    if not oracle_data:
        return opportunities

    p_clean_h = oracle_data["p_clean_h"]
    p_clean_x = oracle_data["p_clean_x"]
    p_clean_a = oracle_data["p_clean_a"]

    # 2. Historical Forces & Lambdas (Dynamically resolved and attached to event)
    lambda_h, lambda_a, source_desc = resolve_event_lambdas(
        event=event,
        sport_meta=sport_meta,
        ai_adjustment=ai_adjustment,
    )

    # 3. 9x9 Poisson Matrix
    poisson_mat = calculate_poisson_matrix(lambda_h, lambda_a, max_goals=9)

    # Poisson outcome probabilities
    p_pois_h = float(np.sum(np.tril(poisson_mat, -1)))  # i > j (Home wins)
    p_pois_x = float(np.trace(poisson_mat))             # i == j (Draw)
    p_pois_a = float(np.sum(np.triu(poisson_mat, 1)))   # i < j (Away wins)

    # 4. Bayesian Blend: 60% Poisson + 40% Clean Oracle
    p_final_h = 0.60 * p_pois_h + 0.40 * p_clean_h
    p_final_x = 0.60 * p_pois_x + 0.40 * p_clean_x
    p_final_a = 0.60 * p_pois_a + 0.40 * p_clean_a

    # Normalize Bayesian probabilities
    norm_sum = p_final_h + p_final_x + p_final_a
    p_final_h /= norm_sum
    p_final_x /= norm_sum
    p_final_a /= norm_sum

    # ==========================================
    # MARKET A: Draw No Bet (Apuesta sin Empate - Local)
    # ==========================================
    p_eff_dnb = p_final_h / (p_final_h + p_final_a) if (p_final_h + p_final_a) > 0 else 0.0
    fair_odds_dnb = (1.0 / p_eff_dnb) if p_eff_dnb > 0 else 999.0

    # Evaluate each recreational bookmaker for DNB
    for bm in bookmakers:
        bm_key = bm.get("key", "")
        bm_title = bm.get("title", bm_key.capitalize())
        if bm_key == "pinnacle":
            continue  # Compare recreational against oracle
        # Filtrar exclusivamente casas autorizadas en Colombia
        if not any(h in bm_title for h in ("BetPlay", "Rushbet", "Wplay", "Bwin", "Codere", "Betsson", "Yajuego", "Rivalo", "Mozzart")):
            continue

        h2h_data = extract_vig_free_h2h(bm)
        if not h2h_data:
            continue

        odds_h = h2h_data["odds_h"]
        odds_x = h2h_data["odds_x"]

        # Synthetic DNB odds: Cuota_DNB = Cuota_1 * (1 - (1 / Cuota_X))
        synthetic_odds_dnb = odds_h * (1.0 - (1.0 / odds_x))
        if synthetic_odds_dnb < 1.05:
            continue

        # Expected Value (EV): EV = (P_1 * (Cuota_DNB - 1)) - P_2
        ev_dnb = (p_final_h * (synthetic_odds_dnb - 1.0)) - p_final_a
        ev_dnb_pct = ev_dnb * 100.0

        passes_filter, filter_msg = evaluate_opportunity_filters(
            odds=synthetic_odds_dnb,
            ev_percent=ev_dnb_pct,
            min_ev=min_ev,
            min_odds=min_odds,
            max_odds=max_odds,
        )

        if passes_filter:
            kelly_res = calculate_kelly_stake(
                odds=synthetic_odds_dnb,
                effective_prob=p_eff_dnb,
                bankroll=bankroll,
                kelly_fraction=kelly_k,
            )

            # Dutching breakdown for this recommended stake
            total_stake = kelly_res["recommended_stake"]
            dutching = calculate_dutching_saver(total_stake, odds_h, odds_x)

            instructions = (
                f"1. Ingresa a tu casa de apuestas ({bm_title}).\n"
                f"2. Busca el partido '{match_name}'.\n"
                f"3. Si disponen de mercado 'Apuesta sin Empate / Draw No Bet', apuesta directamente ${total_stake} "
                f"a favor de '{home_team}' a cuota mínima {round(fair_odds_dnb, 2)} (cuota sintética actual: {round(synthetic_odds_dnb, 2)}).\n"
                f"4. Si la casa no ofrece DNB, aplica el Seguro Sintético (Dutching Saver):\n"
                f"   • Apuesta ${dutching['stake_home']} a la Victoria de {home_team} (Cuota {odds_h}). Ganancia neta si gana: +${dutching['net_profit_if_home']}.\n"
                f"   • Apuesta ${dutching['stake_draw']} al Empate (Cuota {odds_x}). Si empatan, se reembolsa el 100% de los ${total_stake} apostados."
            )

            opportunities.append({
                "sport": "Fútbol",
                "tournament": sport_meta.get("name", "Fútbol"),
                "match": match_name,
                "commence_time": event.get("commence_time", ""),
                "market": "Draw No Bet (Local)",
                "selection": f"{home_team} (Apuesta sin Empate)",
                "bookmaker": bm_title,
                "odds": round(synthetic_odds_dnb, 2),
                "fair_odds": round(fair_odds_dnb, 2),
                "effective_prob_pct": round(p_eff_dnb * 100.0, 1),
                "ev_percent": round(ev_dnb_pct, 2),
                "recommended_stake": total_stake,
                "recommended_stake_pct": round(kelly_res["recommended_stake_pct"] * 100.0, 2),
                "is_capped": kelly_res["is_capped"],
                "instructions": instructions,
                "dutching": dutching,
                "lambdas": f"λH: {lambda_h} | λA: {lambda_a} ({source_desc})",
                "ai_adjustment": {
                    "factor_local": float(get_ai_field(ai_adjustment, "factor_ajuste_local", 1.0)),
                    "factor_visita": float(get_ai_field(ai_adjustment, "factor_ajuste_visita", 1.0)),
                    "confianza": float(get_ai_field(ai_adjustment, "confianza_ia", 0.8)),
                    "riesgo": str(get_ai_field(ai_adjustment, "nivel_riesgo", "Bajo")),
                    "justificacion": str(get_ai_field(ai_adjustment, "justificacion_tactica", "")),
                } if ai_adjustment is not None else None,
            })

    # ==========================================
    # MARKET B: Totales de Goles (Over 2.0 Asiático, Over/Under 2.5, Over/Under 1.5) & BTTS
    # ==========================================
    # Over 2.0 Asiático (reembolso en exactamente 2 goles)
    p_perder_over2 = float(poisson_mat[0, 0] + poisson_mat[1, 0] + poisson_mat[0, 1])
    p_reembolso_over2 = float(poisson_mat[2, 0] + poisson_mat[0, 2] + poisson_mat[1, 1])
    p_ganar_over2 = float(max(0.0, 1.0 - (p_perder_over2 + p_reembolso_over2)))
    denom_over2 = p_ganar_over2 + p_perder_over2
    p_eff_over2 = (p_ganar_over2 / denom_over2) if denom_over2 > 0 else 0.0
    fair_odds_over2 = (1.0 / p_eff_over2) if p_eff_over2 > 0 else 999.0

    # Over / Under 1.5, 2.5, 3.5 Goles (sin reembolso)
    p_under_15 = float(poisson_mat[0, 0] + poisson_mat[1, 0] + poisson_mat[0, 1])
    p_over_15 = float(max(0.0, 1.0 - p_under_15))

    p_under_25 = float(p_under_15 + poisson_mat[2, 0] + poisson_mat[0, 2] + poisson_mat[1, 1])
    p_over_25 = float(max(0.0, 1.0 - p_under_25))

    p_under_35 = float(p_under_25 + poisson_mat[3, 0] + poisson_mat[0, 3] + poisson_mat[2, 1] + poisson_mat[1, 2])
    p_over_35 = float(max(0.0, 1.0 - p_under_35))

    # Ambos Equipos Marcan (BTTS)
    p_btts_yes = float(np.sum(poisson_mat[1:, 1:]))
    p_btts_no = float(max(0.0, 1.0 - p_btts_yes))

    # Adjuntar métricas avanzadas de Poisson al evento para el Bet Builder de la interfaz
    try:
        from models.parlay_engine import calculate_poisson_advanced_markets
        event["advanced_poisson_props"] = calculate_poisson_advanced_markets(lambda_h, lambda_a)
        event["lambda_h"] = lambda_h
        event["lambda_a"] = lambda_a
    except Exception:
        pass

    ai_adj_dict = {
        "factor_local": float(get_ai_field(ai_adjustment, "factor_ajuste_local", 1.0)),
        "factor_visita": float(get_ai_field(ai_adjustment, "factor_ajuste_visita", 1.0)),
        "confianza": float(get_ai_field(ai_adjustment, "confianza_ia", 0.8)),
        "riesgo": str(get_ai_field(ai_adjustment, "nivel_riesgo", "Bajo")),
        "justificacion": str(get_ai_field(ai_adjustment, "justificacion_tactica", "")),
    } if ai_adjustment is not None else None

    # Evaluar casas recreacionales colombianas para mercados de Goles y BTTS
    for bm in bookmakers:
        bm_key = bm.get("key", "")
        bm_title = bm.get("title", bm_key.capitalize())
        if bm_key == "pinnacle":
            continue
        # Filtrar exclusivamente casas autorizadas en Colombia
        if not any(h in bm_title for h in ("BetPlay", "Rushbet", "Wplay", "Bwin", "Codere", "Betsson", "Yajuego", "Rivalo", "Mozzart")):
            continue

        for m in bm.get("markets", []):
            m_key = m.get("key", "").lower()
            if m_key == "totals":
                for out in m.get("outcomes", []):
                    name = out.get("name", "")
                    point = float(out.get("point", 0.0))
                    price = float(out.get("price", 0.0))
                    if price <= 1.0:
                        continue

                    # 1. Over 2.0 Asiático
                    if name.lower() == "over" and abs(point - 2.0) < 0.05:
                        ev_over2 = (p_ganar_over2 * (price - 1.0)) - p_perder_over2
                        ev_over2_pct = ev_over2 * 100.0

                        passes_filter, _ = evaluate_opportunity_filters(
                            odds=price,
                            ev_percent=ev_over2_pct,
                            min_ev=min_ev,
                            min_odds=min_odds,
                            max_odds=max_odds,
                        )

                        if passes_filter:
                            kelly_res = calculate_kelly_stake(
                                odds=price,
                                effective_prob=p_eff_over2,
                                bankroll=bankroll,
                                kelly_fraction=kelly_k,
                            )
                            total_stake = kelly_res["recommended_stake"]

                            instructions = (
                                f"1. Ingresa a {bm_title}.\n"
                                f"2. Localiza el encuentro '{match_name}'.\n"
                                f"3. Dirígete a la sección de 'Líneas Asiáticas / Totales de Goles'.\n"
                                f"4. Selecciona 'Más de 2.0 Goles Asiático' a cuota actual {price} (Mínima aceptable: {round(fair_odds_over2, 2)}).\n"
                                f"5. Coloca un stake de ${total_stake} ({round(kelly_res['recommended_stake_pct'] * 100, 1)}% del bankroll).\n"
                                f"   • Si hay 3 o más goles: Ganas íntegro (+${round(total_stake * (price - 1), 2)}).\n"
                                f"   • Si hay 2 goles exactos: REEMBOLSO TOTAL de tu apuesta ($0 ganancia / $0 pérdida).\n"
                                f"   • Si hay 0 o 1 gol: Apuesta perdida."
                            )

                            opportunities.append({
                                "sport": "Fútbol",
                                "tournament": sport_meta.get("name", "Fútbol"),
                                "match": match_name,
                                "commence_time": event.get("commence_time", ""),
                                "market": "Over 2.0 Asiático",
                                "selection": "Más de 2.0 Goles (Asiático)",
                                "bookmaker": bm_title,
                                "odds": round(price, 2),
                                "fair_odds": round(fair_odds_over2, 2),
                                "effective_prob_pct": round(p_eff_over2 * 100.0, 1),
                                "ev_percent": round(ev_over2_pct, 2),
                                "recommended_stake": total_stake,
                                "recommended_stake_pct": round(kelly_res["recommended_stake_pct"] * 100.0, 2),
                                "is_capped": kelly_res["is_capped"],
                                "instructions": instructions,
                                "dutching": None,
                                "lambdas": f"λH: {lambda_h} | λA: {lambda_a} (P_win: {round(p_ganar_over2*100, 1)}% | P_push: {round(p_reembolso_over2*100, 1)}%)",
                                "ai_adjustment": ai_adj_dict,
                            })

                    # 2. Over / Under 2.5 Goles (Mercado Estándar Internacional)
                    elif abs(point - 2.5) < 0.05:
                        is_over = name.lower() == "over"
                        p_sel = p_over_25 if is_over else p_under_25
                        sel_text = "Más de 2.5 Goles" if is_over else "Menos de 2.5 Goles"
                        mkt_name = "Over 2.5 Goles" if is_over else "Under 2.5 Goles"
                        fair_o = (1.0 / p_sel) if p_sel > 0 else 999.0
                        ev_pct = ((p_sel * price) - 1.0) * 100.0

                        passes_filter, _ = evaluate_opportunity_filters(
                            odds=price,
                            ev_percent=ev_pct,
                            min_ev=min_ev,
                            min_odds=min_odds,
                            max_odds=max_odds,
                        )

                        if passes_filter:
                            kelly_res = calculate_kelly_stake(
                                odds=price,
                                effective_prob=p_sel,
                                bankroll=bankroll,
                                kelly_fraction=kelly_k,
                            )
                            total_stake = kelly_res["recommended_stake"]

                            instructions = (
                                f"1. Ingresa a tu casa de apuestas ({bm_title}).\n"
                                f"2. Busca el partido '{match_name}'.\n"
                                f"3. En la sección 'Total de Goles', selecciona '{sel_text}' a cuota actual {price} (Mínima aceptable: {round(fair_o, 2)}).\n"
                                f"4. Coloca un stake de ${total_stake} ({round(kelly_res['recommended_stake_pct'] * 100, 1)}% del bankroll).\n"
                                f"5. Probabilidad real Poisson: {round(p_sel * 100, 1)}% | Ventaja matemática: +{round(ev_pct, 2)}%."
                            )

                            opportunities.append({
                                "sport": "Fútbol",
                                "tournament": sport_meta.get("name", "Fútbol"),
                                "match": match_name,
                                "commence_time": event.get("commence_time", ""),
                                "market": mkt_name,
                                "selection": sel_text,
                                "bookmaker": bm_title,
                                "odds": round(price, 2),
                                "fair_odds": round(fair_o, 2),
                                "effective_prob_pct": round(p_sel * 100.0, 1),
                                "ev_percent": round(ev_pct, 2),
                                "recommended_stake": total_stake,
                                "recommended_stake_pct": round(kelly_res["recommended_stake_pct"] * 100.0, 2),
                                "is_capped": kelly_res["is_capped"],
                                "instructions": instructions,
                                "dutching": None,
                                "lambdas": f"λH: {lambda_h} | λA: {lambda_a} (P: {round(p_sel*100, 1)}%)",
                                "ai_adjustment": ai_adj_dict,
                            })

                    # 3. Over / Under 1.5 Goles
                    elif abs(point - 1.5) < 0.05:
                        is_over = name.lower() == "over"
                        p_sel = p_over_15 if is_over else p_under_15
                        sel_text = "Más de 1.5 Goles" if is_over else "Menos de 1.5 Goles"
                        mkt_name = "Over 1.5 Goles" if is_over else "Under 1.5 Goles"
                        fair_o = (1.0 / p_sel) if p_sel > 0 else 999.0
                        ev_pct = ((p_sel * price) - 1.0) * 100.0

                        passes_filter, _ = evaluate_opportunity_filters(
                            odds=price,
                            ev_percent=ev_pct,
                            min_ev=min_ev,
                            min_odds=min_odds,
                            max_odds=max_odds,
                        )

                        if passes_filter:
                            kelly_res = calculate_kelly_stake(
                                odds=price,
                                effective_prob=p_sel,
                                bankroll=bankroll,
                                kelly_fraction=kelly_k,
                            )
                            total_stake = kelly_res["recommended_stake"]

                            instructions = (
                                f"1. Ingresa a {bm_title}.\n"
                                f"2. Busca el partido '{match_name}'.\n"
                                f"3. En 'Total de Goles', selecciona '{sel_text}' a cuota actual {price} (Mínima aceptable: {round(fair_o, 2)}).\n"
                                f"4. Coloca un stake de ${total_stake} ({round(kelly_res['recommended_stake_pct'] * 100, 1)}% del bankroll).\n"
                                f"5. Probabilidad real Poisson: {round(p_sel * 100, 1)}% | Ventaja matemática: +{round(ev_pct, 2)}%."
                            )

                            opportunities.append({
                                "sport": "Fútbol",
                                "tournament": sport_meta.get("name", "Fútbol"),
                                "match": match_name,
                                "commence_time": event.get("commence_time", ""),
                                "market": mkt_name,
                                "selection": sel_text,
                                "bookmaker": bm_title,
                                "odds": round(price, 2),
                                "fair_odds": round(fair_o, 2),
                                "effective_prob_pct": round(p_sel * 100.0, 1),
                                "ev_percent": round(ev_pct, 2),
                                "recommended_stake": total_stake,
                                "recommended_stake_pct": round(kelly_res["recommended_stake_pct"] * 100.0, 2),
                                "is_capped": kelly_res["is_capped"],
                                "instructions": instructions,
                                "dutching": None,
                                "lambdas": f"λH: {lambda_h} | λA: {lambda_a} (P: {round(p_sel*100, 1)}%)",
                                "ai_adjustment": ai_adj_dict,
                            })

            # 4. Ambos Equipos Marcan (BTTS)
            elif m_key in ("btts", "both_teams_to_score"):
                for out in m.get("outcomes", []):
                    name = out.get("name", "")
                    price = float(out.get("price", 0.0))
                    if price <= 1.0:
                        continue
                    is_yes = name.lower() in ("yes", "sí", "si")
                    p_sel = p_btts_yes if is_yes else p_btts_no
                    sel_text = "Ambos Marcan: Sí" if is_yes else "Ambos Marcan: No"
                    fair_o = (1.0 / p_sel) if p_sel > 0 else 999.0
                    ev_pct = ((p_sel * price) - 1.0) * 100.0

                    passes_filter, _ = evaluate_opportunity_filters(
                        odds=price,
                        ev_percent=ev_pct,
                        min_ev=min_ev,
                        min_odds=min_odds,
                        max_odds=max_odds,
                    )

                    if passes_filter:
                        kelly_res = calculate_kelly_stake(
                            odds=price,
                            effective_prob=p_sel,
                            bankroll=bankroll,
                            kelly_fraction=kelly_k,
                        )
                        total_stake = kelly_res["recommended_stake"]

                        instructions = (
                            f"1. Ingresa a {bm_title}.\n"
                            f"2. Busca el partido '{match_name}'.\n"
                            f"3. En 'Ambos Equipos Anotan', selecciona '{sel_text}' a cuota actual {price} (Mínima aceptable: {round(fair_o, 2)}).\n"
                            f"4. Coloca un stake de ${total_stake} ({round(kelly_res['recommended_stake_pct'] * 100, 1)}% del bankroll).\n"
                            f"5. Probabilidad real Poisson: {round(p_sel * 100, 1)}% | Ventaja matemática: +{round(ev_pct, 2)}%."
                        )

                        opportunities.append({
                            "sport": "Fútbol",
                            "tournament": sport_meta.get("name", "Fútbol"),
                            "match": match_name,
                            "commence_time": event.get("commence_time", ""),
                            "market": "Ambos Equipos Marcan",
                            "selection": sel_text,
                            "bookmaker": bm_title,
                            "odds": round(price, 2),
                            "fair_odds": round(fair_o, 2),
                            "effective_prob_pct": round(p_sel * 100.0, 1),
                            "ev_percent": round(ev_pct, 2),
                            "recommended_stake": total_stake,
                            "recommended_stake_pct": round(kelly_res["recommended_stake_pct"] * 100.0, 2),
                            "is_capped": kelly_res["is_capped"],
                            "instructions": instructions,
                            "dutching": None,
                            "lambdas": f"λH: {lambda_h} | λA: {lambda_a} (P: {round(p_sel*100, 1)}%)",
                            "ai_adjustment": ai_adj_dict,
                        })

    return opportunities
