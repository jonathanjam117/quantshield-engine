"""QuantShield Engine - Parlay & Advanced Markets Engine.

Provides quantitative modeling for:
1. Multi-Match +EV Parlays (Combinadas de Valor con control de varianza).
2. Advanced Goal Markets (Over/Under 1.5/2.5/3.5, Ambos Marcan / BTTS).
3. Same-Game Parlay / Bet Builder Correlated Poisson Combos.
4. Quantitative Corners (Tiros de Esquina) Poisson Estimator.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import poisson


def calculate_poisson_advanced_markets(
    lambda_h: float,
    lambda_a: float,
    max_goals: int = 9,
) -> Dict[str, Any]:
    """Generates the full bivariate Poisson matrix and derives fair probabilities
    and fair decimal odds for all secondary/prop soccer markets.
    """
    poisson_mat = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            poisson_mat[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)

    # 1. Ambos Equipos Marcan (BTTS)
    # P(BTTS Sí) = sum_{i>=1, j>=1} poisson_mat[i, j]
    p_btts_yes = float(np.sum(poisson_mat[1:, 1:]))
    p_btts_no = float(max(0.0, 1.0 - p_btts_yes))

    # 2. Over / Under Markets (1.5, 2.5, 3.5)
    def _calc_totals(thresh: float) -> Tuple[float, float]:
        p_under = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if (i + j) < thresh:
                    p_under += poisson_mat[i, j]
        p_over = float(max(0.0, 1.0 - p_under))
        return p_over, float(p_under)

    p_over_15, p_under_15 = _calc_totals(1.5)
    p_over_25, p_under_25 = _calc_totals(2.5)
    p_over_35, p_under_35 = _calc_totals(3.5)

    # 3. Same-Game Combos (Correlated Bet Builder)
    # Combo 1: Local Gana Y Más de 1.5 Goles
    p_home_and_over15 = 0.0
    # Combo 2: Local Gana Y Ambos Marcan (Home Win & BTTS)
    p_home_and_btts = 0.0
    # Combo 3: Empate Y Menos de 2.5 Goles
    p_draw_and_under25 = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i > j and (i + j) >= 2:
                p_home_and_over15 += poisson_mat[i, j]
            if i > j and (i >= 1 and j >= 1):
                p_home_and_btts += poisson_mat[i, j]
            if i == j and (i + j) < 2.5:
                p_draw_and_under25 += poisson_mat[i, j]

    return {
        "lambda_h": round(lambda_h, 2),
        "lambda_a": round(lambda_a, 2),
        "expected_total_goals": round(lambda_h + lambda_a, 2),
        # BTTS
        "btts_yes_prob": round(p_btts_yes * 100.0, 1),
        "btts_yes_fair_odds": round(1.0 / p_btts_yes, 2) if p_btts_yes > 0 else 999.0,
        "btts_no_prob": round(p_btts_no * 100.0, 1),
        "btts_no_fair_odds": round(1.0 / p_btts_no, 2) if p_btts_no > 0 else 999.0,
        # Totales
        "over_15_prob": round(p_over_15 * 100.0, 1),
        "over_15_fair_odds": round(1.0 / p_over_15, 2) if p_over_15 > 0 else 999.0,
        "under_15_prob": round(p_under_15 * 100.0, 1),
        "under_15_fair_odds": round(1.0 / p_under_15, 2) if p_under_15 > 0 else 999.0,
        "over_25_prob": round(p_over_25 * 100.0, 1),
        "over_25_fair_odds": round(1.0 / p_over_25, 2) if p_over_25 > 0 else 999.0,
        "under_25_prob": round(p_under_25 * 100.0, 1),
        "under_25_fair_odds": round(1.0 / p_under_25, 2) if p_under_25 > 0 else 999.0,
        "over_35_prob": round(p_over_35 * 100.0, 1),
        "over_35_fair_odds": round(1.0 / p_over_35, 2) if p_over_35 > 0 else 999.0,
        # Bet Builder Combos
        "combo_home_over15_prob": round(p_home_and_over15 * 100.0, 1),
        "combo_home_over15_fair_odds": round(1.0 / p_home_and_over15, 2) if p_home_and_over15 > 0 else 999.0,
        "combo_home_btts_prob": round(p_home_and_btts * 100.0, 1),
        "combo_home_btts_fair_odds": round(1.0 / p_home_and_btts, 2) if p_home_and_btts > 0 else 999.0,
        "combo_draw_under25_prob": round(p_draw_and_under25 * 100.0, 1),
        "combo_draw_under25_fair_odds": round(1.0 / p_draw_and_under25, 2) if p_draw_and_under25 > 0 else 999.0,
    }


def calculate_parlay(
    selected_opportunities: List[Dict[str, Any]],
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.20,
    max_cap_pct: float = 0.015,  # 1.5% max stake cap for parlays (conservador por varianza)
) -> Optional[Dict[str, Any]]:
    """Calculates compound metrics for a multi-match accumulator / parlay:
    - Combined decimal odds
    - Compound win probability
    - Fair combined odds
    - Multiplied EV (%)
    - Variance-adjusted Kelly stake recommendation
    """
    if not selected_opportunities or len(selected_opportunities) < 2:
        return None

    # Verify legs are from different matches to respect independence assumption
    match_names = [opp.get("match", "") for opp in selected_opportunities]
    if len(match_names) != len(set(match_names)):
        # Same match selected multiple times
        has_duplicate_matches = True
    else:
        has_duplicate_matches = False

    combined_odds = 1.0
    combined_prob = 1.0

    legs_summary = []
    for opp in selected_opportunities:
        odds = float(opp.get("odds", 1.0))
        prob_pct = float(opp.get("effective_prob_pct", 50.0))
        prob = prob_pct / 100.0

        combined_odds *= odds
        combined_prob *= prob

        legs_summary.append({
            "match": opp.get("match", ""),
            "selection": opp.get("selection", ""),
            "market": opp.get("market", ""),
            "bookmaker": opp.get("bookmaker", ""),
            "odds": odds,
            "effective_prob_pct": prob_pct,
            "ev_percent": opp.get("ev_percent", 0.0),
        })

    combined_odds = round(combined_odds, 2)
    fair_odds = round(1.0 / combined_prob, 2) if combined_prob > 0 else 999.0
    combined_prob_pct = round(combined_prob * 100.0, 1)

    # Compound EV = (P_win * Combined_Odds) - 1.0
    compound_ev = (combined_prob * combined_odds) - 1.0
    compound_ev_pct = round(compound_ev * 100.0, 2)

    # Kelly Criterion adjusted for parlays:
    # f* = (p * b - q) / b
    b = combined_odds - 1.0
    q = 1.0 - combined_prob
    if b > 0:
        f_star = (combined_prob * b - q) / b
    else:
        f_star = 0.0

    # Parlays have higher variance, so we use fractional Kelly (e.g., 1/10th or 1/8th Kelly)
    conservative_multiplier = kelly_multiplier * 0.40  # Further reduce Kelly for accumulator variance
    raw_fraction = max(0.0, f_star * conservative_multiplier)

    is_capped = False
    if raw_fraction > max_cap_pct:
        final_fraction = max_cap_pct
        is_capped = True
    else:
        final_fraction = raw_fraction

    recommended_stake = round(bankroll * final_fraction, 2)
    potential_profit = round(recommended_stake * (combined_odds - 1.0), 2)
    potential_total_payout = round(recommended_stake * combined_odds, 2)

    return {
        "legs_count": len(selected_opportunities),
        "combined_odds": combined_odds,
        "fair_odds": fair_odds,
        "combined_prob_pct": combined_prob_pct,
        "compound_ev_pct": compound_ev_pct,
        "recommended_stake": recommended_stake,
        "recommended_stake_pct": round(final_fraction * 100.0, 2),
        "is_capped": is_capped,
        "potential_profit": potential_profit,
        "potential_total_payout": potential_total_payout,
        "has_duplicate_matches": has_duplicate_matches,
        "legs": legs_summary,
    }


def find_best_auto_parlay(
    opportunities: List[Dict[str, Any]],
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.20,
    num_legs: int = 2,
) -> Optional[Dict[str, Any]]:
    """Selects the top distinct +EV opportunities and generates the optimal
    'Super-Combinada de Valor' of 2 or 3 legs.
    """
    if not opportunities or len(opportunities) < num_legs:
        return None

    # Filter distinct matches with positive EV
    seen_matches = set()
    best_legs: List[Dict[str, Any]] = []

    # Sort by EV descending
    sorted_opps = sorted(opportunities, key=lambda x: x.get("ev_percent", 0.0), reverse=True)

    for opp in sorted_opps:
        m = opp.get("match", "")
        if m not in seen_matches and opp.get("ev_percent", 0.0) > 0:
            seen_matches.add(m)
            best_legs.append(opp)
            if len(best_legs) == num_legs:
                break

    if len(best_legs) < num_legs:
        return None

    return calculate_parlay(best_legs, bankroll=bankroll, kelly_multiplier=kelly_multiplier)


def evaluate_corners_poisson(
    expected_total_corners: float,
    line: float,
    odds_over: float,
    odds_under: float,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.20,
) -> Dict[str, Any]:
    """Evaluates corners markets using Poisson distribution.
    Line is typically 8.5, 9.5, or 10.5 corners.
    """
    lam = max(4.0, expected_total_corners)

    # P(Corners <= floor(line))
    k = int(math.floor(line))
    p_under = float(poisson.cdf(k, lam))
    p_over = float(max(0.0, 1.0 - p_under))

    fair_odds_over = round(1.0 / p_over, 2) if p_over > 0 else 999.0
    fair_odds_under = round(1.0 / p_under, 2) if p_under > 0 else 999.0

    ev_over = (p_over * odds_over) - 1.0
    ev_over_pct = round(ev_over * 100.0, 2)

    ev_under = (p_under * odds_under) - 1.0
    ev_under_pct = round(ev_under * 100.0, 2)

    # Best selection
    if ev_over_pct > ev_under_pct and ev_over_pct > 0:
        best_selection = f"Más de {line} Córners"
        best_odds = odds_over
        best_fair = fair_odds_over
        best_ev = ev_over_pct
        best_prob = round(p_over * 100.0, 1)
        has_value = True
    elif ev_under_pct > 0:
        best_selection = f"Menos de {line} Córners"
        best_odds = odds_under
        best_fair = fair_odds_under
        best_ev = ev_under_pct
        best_prob = round(p_under * 100.0, 1)
        has_value = True
    else:
        best_selection = "Sin Ventaja (Margen Casa)"
        best_odds = max(odds_over, odds_under)
        best_fair = min(fair_odds_over, fair_odds_under)
        best_ev = max(ev_over_pct, ev_under_pct)
        best_prob = round(max(p_over, p_under) * 100.0, 1)
        has_value = False

    return {
        "expected_corners": round(lam, 1),
        "line": line,
        "p_over_pct": round(p_over * 100.0, 1),
        "p_under_pct": round(p_under * 100.0, 1),
        "fair_odds_over": fair_odds_over,
        "fair_odds_under": fair_odds_under,
        "ev_over_pct": ev_over_pct,
        "ev_under_pct": ev_under_pct,
        "has_value": has_value,
        "best_selection": best_selection,
        "best_odds": best_odds,
        "best_fair": best_fair,
        "best_ev": best_ev,
        "best_prob": best_prob,
    }
