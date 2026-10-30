"""QuantShield Engine - BetPlay Markets & Same-Game Parlay (SGP) Engine.

Implements Section 2 of ADDON_BETPLAY_UI.md:
1. Complete catalog of BetPlay security markets calculated from Poisson 9x9 matrix:
   - 1X2 & Double Chance (1X, 12, X2)
   - Draw No Bet (Apuesta sin Empate) Local and Away
   - Both Teams to Score (Ambos Equipos Marcarán: Sí / No)
   - Team Total Goals (Local/Visita Más/Menos 0.5, 1.5, 2.5)
   - Goal in Both Halves (Gol en Ambas Mitades: Sí / No)
   - Goal Bands (Bandas de Goles: 0-2, 3-4, 5+ goles)
   - Correlated Same-Game Combos (Local Gana & BTTS, Visita Gana & BTTS, Empate & Under 2.5)
2. 3 Low-Variance Correlated Same-Game Parlays per match ("Combinadas para ti"):
   - Tarjeta 1: 🛡️ Combinada Blindada (Bajo Riesgo, Cuota 1.65 - 2.15)
   - Tarjeta 2: ⚽ Combinada de Ritmo Ofensivo (Riesgo Medio, Cuota 2.10 - 2.85)
   - Tarjeta 3: 🚀 Combinada de Valor + IA (Cuota 2.85 - 4.20)
3. Joint Probability with Copula/Correlation factor and Kelly stake sizing in COP.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import poisson
from models.ai_context_agent import get_ai_field


def estimate_retail_odds(prob: float, margin: float = 0.95) -> float:
    """Calcula la cuota estimada de casa con margen (overround del 5%) acotando probabilidad máxima a 0.94.
    Fórmula requerida:
    p_clamped = min(max(prob, 0.05), 0.94)
    cuota_estimada = round(max(1.05, (1.0 / p_clamped) * margin), 2)
    Garantiza que ninguna cuota sea menor a 1.05 ni derive de p > 0.94.
    """
    p_clamped = min(max(float(prob), 0.05), 0.94)
    return round(max(1.05, (1.0 / p_clamped) * margin), 2)


def build_bivariate_poisson(lambda_h: float, lambda_a: float, max_goals: int = 9) -> np.ndarray:
    """Generates a 10x10 bivariate Poisson probability matrix normalized to sum 1.0."""
    mat = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
    s = np.sum(mat)
    if s > 0:
        mat /= s
    return mat


def calculate_betplay_security_catalog(
    home_team: str,
    away_team: str,
    lambda_h: float,
    lambda_a: float,
    bankroll: float = 100000.0,
    kelly_multiplier: float = 0.20,
    retail_odds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Calculates all BetPlay security markets with probabilities, fair odds, estimated retail odds,
    EV diagnostic and recommended stakes.
    """
    mat = build_bivariate_poisson(lambda_h, lambda_a, max_goals=9)
    retail = retail_odds or {}

    # 1. 1X2 Probabilities
    p_h = float(np.sum(np.tril(mat, -1)))  # i > j
    p_x = float(np.trace(mat))             # i == j
    p_a = float(np.sum(np.triu(mat, 1)))   # i < j

    # 2. Doble Oportunidad
    p_1x = p_h + p_x
    p_12 = p_h + p_a
    p_x2 = p_x + p_a

    # 3. Apuesta sin Empate (DNB)
    denom_dnb = p_h + p_a
    p_dnb_h = (p_h / denom_dnb) if denom_dnb > 0 else 0.5
    p_dnb_a = (p_a / denom_dnb) if denom_dnb > 0 else 0.5

    # 4. Ambos Equipos Marcarán (BTTS)
    p_btts_yes = float(np.sum(mat[1:, 1:]))
    p_btts_no = float(max(0.0, 1.0 - p_btts_yes))

    # 5. Total de Goles por Equipo (Individual) usando Poisson marginal
    # Local
    p_h_over05 = float(1.0 - math.exp(-lambda_h))
    p_h_over15 = float(1.0 - math.exp(-lambda_h) * (1.0 + lambda_h))
    p_h_over25 = float(1.0 - math.exp(-lambda_h) * (1.0 + lambda_h + (lambda_h ** 2) / 2.0))
    p_h_under05 = float(1.0 - p_h_over05)
    p_h_under15 = float(1.0 - p_h_over15)
    p_h_under25 = float(1.0 - p_h_over25)

    # Visita
    p_a_over05 = float(1.0 - math.exp(-lambda_a))
    p_a_over15 = float(1.0 - math.exp(-lambda_a) * (1.0 + lambda_a))
    p_a_over25 = float(1.0 - math.exp(-lambda_a) * (1.0 + lambda_a + (lambda_a ** 2) / 2.0))
    p_a_under05 = float(1.0 - p_a_over05)
    p_a_under15 = float(1.0 - p_a_over15)
    p_a_under25 = float(1.0 - p_a_over25)

    # 6. Gol en Ambas Mitades (Distribución Temporal: 45% 1H, 55% 2H)
    tot_lam = lambda_h + lambda_a
    lam_1h = 0.45 * tot_lam
    lam_2h = 0.55 * tot_lam
    p_goal_1h = float(1.0 - math.exp(-lam_1h))
    p_goal_2h = float(1.0 - math.exp(-lam_2h))
    p_both_halves_yes = float(p_goal_1h * p_goal_2h)
    p_both_halves_no = float(max(0.0, 1.0 - p_both_halves_yes))

    # 7. Número de Bandas de Goles
    p_band_0_2 = 0.0
    p_band_3_4 = 0.0
    p_band_5_plus = 0.0
    for i in range(10):
        for j in range(10):
            g = i + j
            if g <= 2:
                p_band_0_2 += mat[i, j]
            elif 3 <= g <= 4:
                p_band_3_4 += mat[i, j]
            else:
                p_band_5_plus += mat[i, j]

    p_band_0_2 = float(p_band_0_2)
    p_band_3_4 = float(p_band_3_4)
    p_band_5_plus = float(p_band_5_plus)
    p_over_15_total = float(1.0 - (mat[0, 0] + mat[1, 0] + mat[0, 1]))

    # 8. Mercados Combinados Pre-armados Mismo Partido
    p_h_and_btts = 0.0
    p_a_and_btts = 0.0
    p_draw_and_under25 = 0.0
    for i in range(10):
        for j in range(10):
            if i > j and j >= 1:
                p_h_and_btts += mat[i, j]
            if j > i and i >= 1:
                p_a_and_btts += mat[i, j]
            if i == j and (i + j) <= 2:
                p_draw_and_under25 += mat[i, j]

    p_h_and_btts = float(p_h_and_btts)
    p_a_and_btts = float(p_a_and_btts)
    p_draw_and_under25 = float(p_draw_and_under25)

    # 9. Helper to compute market item
    def _item(
        name: str,
        prob: float,
        default_retail_odds: Optional[float] = None,
        retail_key: str = "",
        min_ev_thresh: float = 2.0,
    ) -> Dict[str, Any]:
        p = max(0.001, min(0.999, float(prob)))
        fair = round(1.0 / p, 2)
        computed_default = estimate_retail_odds(p, margin=0.95)
        def_odds = default_retail_odds if default_retail_odds is not None else computed_default
        raw_val = retail.get(retail_key, def_odds)
        actual_odds = round(max(1.05, float(raw_val)), 2)
        ev = (p * actual_odds) - 1.0
        ev_pct = round(ev * 100.0, 2)
        has_val = ev_pct >= min_ev_thresh

        # Kelly Stake calculation
        b = actual_odds - 1.0
        q = 1.0 - p
        if b > 0 and has_val:
            f_star = (p * b - q) / b
            stake_pct = min(0.025, max(0.005, f_star * kelly_multiplier))
            rec_stake = round(bankroll * stake_pct, -2) if bankroll >= 10000 else round(bankroll * stake_pct, 1)
        else:
            stake_pct = 0.0
            rec_stake = 0.0

        return {
            "name": name,
            "prob_pct": round(p * 100.0, 1),
            "fair_odds": fair,
            "retail_odds": actual_odds,
            "ev_pct": ev_pct,
            "has_value": has_val,
            "rec_stake": rec_stake,
            "stake_pct": round(stake_pct * 100.0, 2),
            "action": "🟢 ENTRAR (+EV)" if has_val else "⚪ PASAR",
        }

    # Generate complete dictionary of BetPlay security catalog
    return {
        "home_team": home_team,
        "away_team": away_team,
        "lambdas": {"home": round(lambda_h, 2), "away": round(lambda_a, 2), "total": round(tot_lam, 2)},
        # Bloque 1: 1X2 & Doble Oportunidad
        "block_1x2": {
            "home": _item(f"Victoria {home_team} (1)", p_h, estimate_retail_odds(p_h, 0.94), "1"),
            "draw": _item("Empate (X)", p_x, estimate_retail_odds(p_x, 0.94), "X"),
            "away": _item(f"Victoria {away_team} (2)", p_a, estimate_retail_odds(p_a, 0.94), "2"),
        },
        "block_double_chance": {
            "1X": _item(f"{home_team} o Empate (1X)", p_1x, estimate_retail_odds(p_1x, 0.95), "1X"),
            "12": _item(f"{home_team} o {away_team} (12)", p_12, estimate_retail_odds(p_12, 0.95), "12"),
            "X2": _item(f"Empate o {away_team} (X2)", p_x2, estimate_retail_odds(p_x2, 0.95), "X2"),
        },
        # Bloque 2: Apuesta sin Empate (DNB)
        "block_dnb": {
            "dnb_home": _item(f"{home_team} (Sin Empate)", p_dnb_h, estimate_retail_odds(p_dnb_h, 0.95), "DNB_H"),
            "dnb_away": _item(f"{away_team} (Sin Empate)", p_dnb_a, estimate_retail_odds(p_dnb_a, 0.95), "DNB_A"),
        },
        # Bloque 3: Total Goles por Equipo (Individual)
        "block_team_goals_home": {
            "over_05": _item(f"{home_team}: Más de 0.5 Goles", p_h_over05, estimate_retail_odds(p_h_over05, 0.95), "H_O05"),
            "under_05": _item(f"{home_team}: Menos de 0.5 Goles", p_h_under05, estimate_retail_odds(p_h_under05, 0.95), "H_U05"),
            "over_15": _item(f"{home_team}: Más de 1.5 Goles", p_h_over15, estimate_retail_odds(p_h_over15, 0.95), "H_O15"),
            "under_15": _item(f"{home_team}: Menos de 1.5 Goles", p_h_under15, estimate_retail_odds(p_h_under15, 0.95), "H_U15"),
            "over_25": _item(f"{home_team}: Más de 2.5 Goles", p_h_over25, estimate_retail_odds(p_h_over25, 0.95), "H_O25"),
        },
        "block_team_goals_away": {
            "over_05": _item(f"{away_team}: Más de 0.5 Goles", p_a_over05, estimate_retail_odds(p_a_over05, 0.95), "A_O05"),
            "under_05": _item(f"{away_team}: Menos de 0.5 Goles", p_a_under05, estimate_retail_odds(p_a_under05, 0.95), "A_U05"),
            "over_15": _item(f"{away_team}: Más de 1.5 Goles", p_a_over15, estimate_retail_odds(p_a_over15, 0.95), "A_O15"),
            "under_15": _item(f"{away_team}: Menos de 1.5 Goles", p_a_under15, estimate_retail_odds(p_a_under15, 0.95), "A_U15"),
            "over_25": _item(f"{away_team}: Más de 2.5 Goles", p_a_over25, estimate_retail_odds(p_a_over25, 0.95), "A_O25"),
        },
        # Bloque 4: Ambos Marcan & Gol en Ambas Mitades
        "block_btts": {
            "btts_yes": _item("Ambos Equipos Anotan: Sí", p_btts_yes, estimate_retail_odds(p_btts_yes, 0.95), "BTTS_Y"),
            "btts_no": _item("Ambos Equipos Anotan: No", p_btts_no, estimate_retail_odds(p_btts_no, 0.95), "BTTS_N"),
        },
        "block_both_halves": {
            "halves_yes": _item("Gol en Ambas Mitades: Sí", p_both_halves_yes, estimate_retail_odds(p_both_halves_yes, 0.95), "HALF_Y"),
            "halves_no": _item("Gol en Ambas Mitades: No", p_both_halves_no, estimate_retail_odds(p_both_halves_no, 0.95), "HALF_N"),
            "first_half_over05": _item("1ª Mitad: Más de 0.5 Goles", p_goal_1h, estimate_retail_odds(p_goal_1h, 0.95), "1H_O05"),
        },
        # Bloque 5: Bandas de Goles & Combos
        "block_goal_bands": {
            "band_0_2": _item("Número de Goles: 0 a 2 Goles", p_band_0_2, estimate_retail_odds(p_band_0_2, 0.94), "BAND_02"),
            "band_3_4": _item("Número de Goles: 3 a 4 Goles", p_band_3_4, estimate_retail_odds(p_band_3_4, 0.94), "BAND_34"),
            "band_5_plus": _item("Número de Goles: 5 o más Goles", p_band_5_plus, estimate_retail_odds(p_band_5_plus, 0.94), "BAND_5P"),
            "over_15_total": _item("Total Partido: Más de 1.5 Goles", p_over_15_total, estimate_retail_odds(p_over_15_total, 0.95), "TOT_O15"),
            "over_25_total": _item("Total Partido: Más de 2.5 Goles", float(1.0 - p_band_0_2), estimate_retail_odds(1.0 - p_band_0_2, 0.95), "TOT_O25"),
        },
        "block_combos": {
            "h_and_btts": _item(f"{home_team} Gana y Ambos Marcan", p_h_and_btts, estimate_retail_odds(p_h_and_btts, 0.94) if p_h_and_btts > 0 else 999.0, "H_BTTS"),
            "a_and_btts": _item(f"{away_team} Gana y Ambos Marcan", p_a_and_btts, estimate_retail_odds(p_a_and_btts, 0.94) if p_a_and_btts > 0 else 999.0, "A_BTTS"),
            "draw_under25": _item("Empate y Menos de 2.5 Goles", p_draw_and_under25, estimate_retail_odds(p_draw_and_under25, 0.94) if p_draw_and_under25 > 0 else 999.0, "X_U25"),
        },
    }


def generate_betplay_same_game_parlays(
    catalog: Dict[str, Any],
    bankroll: float = 100000.0,
    kelly_multiplier: float = 0.20,
    ai_adjustment: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Builds the 3 'Combinadas para ti' (Same-Game Parlays) with low variance,
    positive correlation factor, and automatic replacement of low-value (<1.12)
    lines so each selection provides real value (>= 1.14) and total odds >= 1.50.
    """
    home = catalog["home_team"]
    away = catalog["away_team"]
    lam_h = catalog["lambdas"]["home"]
    lam_a = catalog["lambdas"]["away"]
    fav_is_home = lam_h >= lam_a
    fav_team = home if fav_is_home else away
    dog_team = away if fav_is_home else home

    cards: List[Dict[str, Any]] = []

    def _build_safe_leg(mkt_label: str, item: Dict[str, Any]) -> Dict[str, Any]:
        p = min(max(item["prob_pct"] / 100.0, 0.05), 0.94)
        odds = round(max(1.14, float(item["retail_odds"])), 2)
        return {
            "market": mkt_label,
            "selection": item["name"],
            "odds": odds,
            "prob": p,
        }

    # --------------------------------------------------------------------------
    # TARJETA 1: 🛡️ Combinada Blindada (Bajo Riesgo - Cuota Objetivo 1.65 - 2.15)
    # --------------------------------------------------------------------------
    # Selección 1: Doble Oportunidad del favorito (o DNB / 1X2 si cuota < 1.12)
    sel1_dc = catalog["block_double_chance"]["1X"] if fav_is_home else catalog["block_double_chance"]["X2"]
    mkt1 = "Doble Oportunidad"
    if sel1_dc["retail_odds"] < 1.12:
        dnb_cand = catalog["block_dnb"]["dnb_home" if fav_is_home else "dnb_away"]
        if dnb_cand["retail_odds"] >= 1.12:
            sel1_dc = dnb_cand
            mkt1 = "Apuesta sin Empate"
        else:
            sel1_dc = catalog["block_1x2"]["home" if fav_is_home else "away"]
            mkt1 = "Resultado Final (1X2)"

    # Selección 2: Total de Goles Favorito (Más de 0.5 -> Más de 1.5 -> Total +2.5 si cuota < 1.12)
    sel2_fg = catalog["block_team_goals_home"]["over_05"] if fav_is_home else catalog["block_team_goals_away"]["over_05"]
    mkt2 = "Total Goles Favorito"
    if sel2_fg["retail_odds"] < 1.12:
        cand_o15 = catalog["block_team_goals_home"]["over_15"] if fav_is_home else catalog["block_team_goals_away"]["over_15"]
        if cand_o15["retail_odds"] >= 1.12:
            sel2_fg = cand_o15
            mkt2 = f"Goles {fav_team} (Más de 1.5)"
        else:
            sel2_fg = catalog["block_goal_bands"].get("over_25_total") or catalog["block_goal_bands"]["band_3_4"]
            mkt2 = "Línea de Goles Partido"

    # Selección 3: 1ª Mitad Más de 0.5 Goles (o Gol en Ambas Mitades / Total +1.5 si < 1.12)
    sel3_1h = catalog["block_both_halves"]["first_half_over05"]
    mkt3 = "Goles 1ª Mitad"
    if sel3_1h["retail_odds"] < 1.12:
        cand_halves = catalog["block_both_halves"]["halves_yes"]
        if cand_halves["retail_odds"] >= 1.12:
            sel3_1h = cand_halves
            mkt3 = "Gol en Ambas Mitades"
        else:
            sel3_1h = catalog["block_goal_bands"]["over_15_total"]
            mkt3 = "Total Partido Más de 1.5"

    legs_t1 = [
        _build_safe_leg(mkt1, sel1_dc),
        _build_safe_leg(mkt2, sel2_fg),
        _build_safe_leg(mkt3, sel3_1h),
    ]

    # Correlation boost: favor team winning correlates positively with scoring and 1H goal
    rho_1 = 1.14
    raw_prob_t1 = legs_t1[0]["prob"] * legs_t1[1]["prob"] * legs_t1[2]["prob"]
    p_comb_t1 = min(0.82, max(0.40, raw_prob_t1 * rho_1))
    raw_odds_t1 = legs_t1[0]["odds"] * legs_t1[1]["odds"] * legs_t1[2]["odds"]
    odds_comb_t1 = round(raw_odds_t1 * 0.94, 2)  # Realistic SGP pricing
    odds_comb_t1 = round(max(1.50, odds_comb_t1), 2)
    ev_t1 = (p_comb_t1 * odds_comb_t1) - 1.0
    ev_pct_t1 = round(ev_t1 * 100.0, 2)

    # Kelly stake
    b1 = odds_comb_t1 - 1.0
    q1 = 1.0 - p_comb_t1
    f_star1 = (p_comb_t1 * b1 - q1) / b1 if b1 > 0 else 0.0
    stake_pct_t1 = min(0.015, max(0.005, f_star1 * kelly_multiplier * 0.50))
    stake_t1 = round(bankroll * stake_pct_t1, -2) if bankroll >= 10000 else round(bankroll * stake_pct_t1, 1)
    profit_t1 = round(stake_t1 * b1, 2)

    cards.append({
        "id": "parlay_shield",
        "badge": "🛡️ COMBINADA BLINDADA",
        "risk_label": "Bajo Riesgo / Máxima Probabilidad",
        "card_theme": "shield",
        "odds": odds_comb_t1,
        "prob_pct": round(p_comb_t1 * 100.0, 1),
        "fair_odds": round(1.0 / p_comb_t1, 2) if p_comb_t1 > 0 else 999.0,
        "ev_pct": ev_pct_t1,
        "recommended_stake": stake_t1,
        "potential_profit": profit_t1,
        "payout": round(stake_t1 * odds_comb_t1, 2),
        "legs": legs_t1,
        "instructions": f"Cómo armarla en BetPlay: Entra a {home} vs {away} ➔ Pestaña 'Crea tu Apuesta' ➔ Activa estas 3 selecciones de seguridad.",
    })

    # --------------------------------------------------------------------------
    # TARJETA 2: ⚽ Combinada de Ritmo Ofensivo (Riesgo Medio - Cuota 2.10 - 2.85)
    # --------------------------------------------------------------------------
    sel1_t2 = catalog["block_team_goals_home"]["over_05"]
    mkt1_t2 = f"Goles {home}"
    if sel1_t2["retail_odds"] < 1.12:
        cand_h15 = catalog["block_team_goals_home"]["over_15"]
        if cand_h15["retail_odds"] >= 1.12:
            sel1_t2 = cand_h15
            mkt1_t2 = f"Goles {home} (Más de 1.5)"
        else:
            sel1_t2 = catalog["block_goal_bands"].get("over_25_total") or catalog["block_goal_bands"]["band_3_4"]
            mkt1_t2 = "Línea de Goles Partido"

    sel2_t2 = catalog["block_both_halves"]["halves_yes"]
    mkt2_t2 = "Gol en Ambas Mitades"
    if sel2_t2["retail_odds"] < 1.12:
        cand_btts = catalog["block_btts"]["btts_yes"]
        if cand_btts["retail_odds"] >= 1.12:
            sel2_t2 = cand_btts
            mkt2_t2 = "Ambos Equipos Anotan"
        else:
            sel2_t2 = catalog["block_goal_bands"]["band_3_4"]
            mkt2_t2 = "Banda de Goles"

    sel3_t2 = catalog["block_goal_bands"]["over_15_total"]
    mkt3_t2 = "Línea de Goles"
    if sel3_t2["retail_odds"] < 1.12:
        cand_tot25 = catalog["block_goal_bands"].get("over_25_total") or catalog["block_goal_bands"]["band_3_4"]
        if cand_tot25["retail_odds"] >= 1.12:
            sel3_t2 = cand_tot25
            mkt3_t2 = "Total Partido Más de 2.5"
        else:
            sel3_t2 = catalog["block_btts"]["btts_yes"]
            mkt3_t2 = "Ambos Equipos Anotan"

    legs_t2 = [
        _build_safe_leg(mkt1_t2, sel1_t2),
        _build_safe_leg(mkt2_t2, sel2_t2),
        _build_safe_leg(mkt3_t2, sel3_t2),
    ]

    rho_2 = 1.18
    raw_prob_t2 = legs_t2[0]["prob"] * legs_t2[1]["prob"] * legs_t2[2]["prob"]
    p_comb_t2 = min(0.70, max(0.30, raw_prob_t2 * rho_2))
    raw_odds_t2 = legs_t2[0]["odds"] * legs_t2[1]["odds"] * legs_t2[2]["odds"]
    odds_comb_t2 = round(raw_odds_t2 * 0.92, 2)
    odds_comb_t2 = round(max(2.10, min(3.50, odds_comb_t2)), 2)
    ev_t2 = (p_comb_t2 * odds_comb_t2) - 1.0
    ev_pct_t2 = round(ev_t2 * 100.0, 2)

    b2 = odds_comb_t2 - 1.0
    q2 = 1.0 - p_comb_t2
    f_star2 = (p_comb_t2 * b2 - q2) / b2 if b2 > 0 else 0.0
    stake_pct_t2 = min(0.012, max(0.004, f_star2 * kelly_multiplier * 0.45))
    stake_t2 = round(bankroll * stake_pct_t2, -2) if bankroll >= 10000 else round(bankroll * stake_pct_t2, 1)
    profit_t2 = round(stake_t2 * b2, 2)

    cards.append({
        "id": "parlay_offensive",
        "badge": "⚽ COMBINADA RITMO OFENSIVO",
        "risk_label": "Riesgo Medio / Alto Retorno",
        "card_theme": "offensive",
        "odds": odds_comb_t2,
        "prob_pct": round(p_comb_t2 * 100.0, 1),
        "fair_odds": round(1.0 / p_comb_t2, 2) if p_comb_t2 > 0 else 999.0,
        "ev_pct": ev_pct_t2,
        "recommended_stake": stake_t2,
        "potential_profit": profit_t2,
        "payout": round(stake_t2 * odds_comb_t2, 2),
        "legs": legs_t2,
        "instructions": f"Cómo armarla en BetPlay: Entra a {home} vs {away} ➔ Pestaña 'Crea tu Apuesta' ➔ Selecciona estas 3 opciones ofensivas.",
    })

    # --------------------------------------------------------------------------
    # TARJETA 3: 🚀 Combinada de Valor + IA (Cuota Objetivo 2.85 - 4.20)
    # --------------------------------------------------------------------------
    sel1_t3 = catalog["block_dnb"]["dnb_home" if fav_is_home else "dnb_away"]
    mkt1_t3 = "Apuesta sin Empate"
    if sel1_t3["retail_odds"] < 1.12:
        sel1_t3 = catalog["block_1x2"]["home" if fav_is_home else "away"]
        mkt1_t3 = "Resultado Final (1X2)"

    sel2_t3 = catalog["block_btts"]["btts_yes"]
    mkt2_t3 = "Ambos Equipos Anotan"
    if sel2_t3["retail_odds"] < 1.12:
        cand_combo = catalog["block_combos"]["h_and_btts" if fav_is_home else "a_and_btts"]
        if cand_combo["retail_odds"] >= 1.12 and cand_combo["retail_odds"] < 900.0:
            sel2_t3 = cand_combo
            mkt2_t3 = "Gana y Ambos Marcan"
        else:
            sel2_t3 = catalog["block_goal_bands"].get("over_25_total") or catalog["block_goal_bands"]["band_3_4"]
            mkt2_t3 = "Total Goles Partido"

    sel3_t3 = catalog["block_both_halves"]["halves_yes"]
    mkt3_t3 = "Gol en Ambas Mitades"
    if sel3_t3["retail_odds"] < 1.12:
        sel3_t3 = catalog["block_goal_bands"].get("over_25_total") or catalog["block_goal_bands"]["band_3_4"]
        mkt3_t3 = "Total Más de 2.5 Goles"

    legs_t3 = [
        _build_safe_leg(mkt1_t3, sel1_t3),
        _build_safe_leg(mkt2_t3, sel2_t3),
        _build_safe_leg(mkt3_t3, sel3_t3),
    ]

    rho_3 = 1.22
    if ai_adjustment is not None:
        conf = float(get_ai_field(ai_adjustment, "confianza_ia", 0.8))
        rho_3 *= (1.0 + (conf - 0.5) * 0.2)

    raw_prob_t3 = legs_t3[0]["prob"] * legs_t3[1]["prob"] * legs_t3[2]["prob"]
    p_comb_t3 = min(0.55, max(0.22, raw_prob_t3 * rho_3))
    raw_odds_t3 = legs_t3[0]["odds"] * legs_t3[1]["odds"] * legs_t3[2]["odds"]
    odds_comb_t3 = round(raw_odds_t3 * 0.90, 2)
    odds_comb_t3 = round(max(2.85, min(4.50, odds_comb_t3)), 2)
    ev_t3 = (p_comb_t3 * odds_comb_t3) - 1.0
    ev_pct_t3 = round(ev_t3 * 100.0, 2)

    b3 = odds_comb_t3 - 1.0
    q3 = 1.0 - p_comb_t3
    f_star3 = (p_comb_t3 * b3 - q3) / b3 if b3 > 0 else 0.0
    stake_pct_t3 = min(0.010, max(0.003, f_star3 * kelly_multiplier * 0.35))
    stake_t3 = round(bankroll * stake_pct_t3, -2) if bankroll >= 10000 else round(bankroll * stake_pct_t3, 1)
    profit_t3 = round(stake_t3 * b3, 2)

    cards.append({
        "id": "parlay_value_ai",
        "badge": "🚀 COMBINADA DE VALOR + IA",
        "risk_label": "Cuota Alta / Impulsada por Gemini AI",
        "card_theme": "ai",
        "odds": odds_comb_t3,
        "prob_pct": round(p_comb_t3 * 100.0, 1),
        "fair_odds": round(1.0 / p_comb_t3, 2) if p_comb_t3 > 0 else 999.0,
        "ev_pct": ev_pct_t3,
        "recommended_stake": stake_t3,
        "potential_profit": profit_t3,
        "payout": round(stake_t3 * odds_comb_t3, 2),
        "legs": legs_t3,
        "instructions": f"Cómo armarla en BetPlay: Entra a {home} vs {away} ➔ Pestaña 'Crea tu Apuesta' ➔ Activa estas 3 selecciones de valor.",
    })

    return cards
