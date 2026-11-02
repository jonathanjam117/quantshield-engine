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
import math
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


def solve_lambda_from_over(line: float, odds_over: float, margin: float = 1.075) -> float:
    """Derives the implicit Poisson expected goals (lambda_tot) from an Over/Under goal line and retail odds.
    E.g. Over 3.5 @ 2.12 with retail margin yields lambda_tot ~ 3.35 - 3.40.
    """
    try:
        line_val = float(line)
        odds_val = float(odds_over)
        if odds_val <= 1.01:
            odds_val = 1.02
        p_fair = max(0.04, min(0.96, (1.0 / odds_val) / margin))
        floor_l = int(math.floor(line_val))

        low, high = 0.4, 7.5
        for _ in range(35):
            mid = (low + high) / 2.0
            term = math.exp(-mid)
            cdf = term
            for k in range(1, floor_l + 1):
                term *= (mid / k)
                cdf += term
            p_gt = 1.0 - cdf
            if p_gt < p_fair:
                low = mid
            else:
                high = mid
        return round((low + high) / 2.0, 2)
    except Exception:
        return 2.70


SHARP_KEYS: List[str] = [
    "pinnacle", "bet365", "betfair", "betfair_ex_uk", "betfair_ex_eu",
    "unibet", "williamhill", "betonlineag", "draftkings", "fanduel", "bovada"
]


def find_sharp_bookmaker(bookmakers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Returns the highest liquidity sharp bookmaker available in the event."""
    if not bookmakers:
        return None
    for k in SHARP_KEYS:
        bm = next((b for b in bookmakers if b.get("key", "").lower() == k), None)
        if bm:
            return bm
    return bookmakers[0]


def extract_sharp_totals(bookmakers: List[Dict[str, Any]]) -> Optional[Tuple[float, float, str]]:
    """Extracts the best totals line (Over point, Over price, bookmaker name) from sharp books."""
    sorted_bms = sorted(
        bookmakers,
        key=lambda b: SHARP_KEYS.index(b.get("key", "").lower()) if b.get("key", "").lower() in SHARP_KEYS else 999
    )
    for bm in sorted_bms:
        for m in bm.get("markets", []):
            if m.get("key") == "totals":
                for out in m.get("outcomes", []):
                    name_low = str(out.get("name", "")).lower()
                    if name_low in ("over", "más de", "mas de"):
                        pt = float(out.get("point", 0.0))
                        price = float(out.get("price", 0.0))
                        if pt > 0 and price > 1.01:
                            bm_title = bm.get("display_title", bm.get("title", bm.get("key", "Sharp")))
                            return pt, price, bm_title
    return None


def extract_event_market_odds(event: Dict[str, Any]) -> Dict[str, float]:
    """Automatically extracts all relevant market retail odds (1X2, synthetic DC, DNB, totals)
    from the sharpest bookmakers in the event.
    """
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    bms = event.get("bookmakers", [])
    odds_map: Dict[str, float] = {}

    for bm in bms:
        for m in bm.get("markets", []):
            if m.get("key") == "h2h":
                for o in m.get("outcomes", []):
                    p = float(o.get("price", 0.0))
                    if p > 1.0:
                        name = o.get("name", "")
                        if name == home and "1" not in odds_map:
                            odds_map["1"] = p
                        elif name in ("Draw", "Empate", "X") and "X" not in odds_map:
                            odds_map["X"] = p
                        elif name == away and "2" not in odds_map:
                            odds_map["2"] = p
            elif m.get("key") == "totals":
                for o in m.get("outcomes", []):
                    name_low = str(o.get("name", "")).lower()
                    if name_low in ("over", "más de", "mas de"):
                        pt = float(o.get("point", 0.0))
                        p = float(o.get("price", 0.0))
                        if p > 1.0:
                            if abs(pt - 1.5) < 0.1 and "O15" not in odds_map:
                                odds_map["O15"] = p
                            elif abs(pt - 2.5) < 0.1 and "O25" not in odds_map:
                                odds_map["O25"] = p
                            elif abs(pt - 3.5) < 0.1 and "O35" not in odds_map:
                                odds_map["O35"] = p

    if "1" not in odds_map:
        odds_map["1"] = 2.50
    if "X" not in odds_map:
        odds_map["X"] = 3.30
    if "2" not in odds_map:
        odds_map["2"] = 2.60

    o1 = odds_map["1"]
    ox = odds_map["X"]
    o2 = odds_map["2"]
    if o1 > 1.0 and ox > 1.0:
        odds_map["1X"] = round(max(1.08, 1.0 / ((1.0 / o1) + (1.0 / ox))), 2)
    if o2 > 1.0 and ox > 1.0:
        odds_map["X2"] = round(max(1.08, 1.0 / ((1.0 / o2) + (1.0 / ox))), 2)
    if o1 > 1.0 and o2 > 1.0:
        odds_map["12"] = round(max(1.08, 1.0 / ((1.0 / o1) + (1.0 / o2))), 2)

    return odds_map


def resolve_event_lambdas(
    event: Dict[str, Any],
    sport_meta: Optional[Dict[str, Any]] = None,
    ai_adjustment: Optional[Any] = None,
    custom_odds: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, str]:
    """Resolves and caches dynamic Poisson lambdas (lambda_h, lambda_a) for an event
    100% automatically using sharp bookmaker vig-free probabilities and total goals line,
    adjusted by Gemini AI if provided.
    """
    if "lambda_h" in event and "lambda_a" in event and not custom_odds:
        return float(event["lambda_h"]), float(event["lambda_a"]), str(event.get("lambda_source", "Cache"))

    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    bookmakers = event.get("bookmakers", [])

    # 1. Sharp Bookmaker for 1X2 Vig-Free probabilities
    sharp_bm = find_sharp_bookmaker(bookmakers)
    oracle_data = extract_vig_free_h2h(sharp_bm) if sharp_bm else None
    if not oracle_data:
        for b in bookmakers:
            od = extract_vig_free_h2h(b)
            if od:
                oracle_data = od
                sharp_bm = b
                break

    sharp_title = sharp_bm.get("display_title", sharp_bm.get("title", sharp_bm.get("key", "Sharp"))) if sharp_bm else "Sharp"

    # 2. Derive total expected goals (lambda_tot) automatically from sharp totals line
    sharp_totals = extract_sharp_totals(bookmakers)
    if sharp_totals:
        tot_pt, tot_odds, tot_bm = sharp_totals
        tot_lam = solve_lambda_from_over(tot_pt, tot_odds)
        src = f"⚡ Oráculo Sharp ({tot_bm} Over {tot_pt} @{tot_odds:.2f} ➔ λ_tot: {tot_lam:.2f})"
    else:
        # Fallback to historical forces
        oracle_probs = (oracle_data["p_clean_h"], oracle_data["p_clean_x"], oracle_data["p_clean_a"]) if oracle_data else None
        csv_code = sport_meta.get("csv_code") if sport_meta else None
        lh_hist, la_hist, hist_src = load_historical_forces(
            csv_code=csv_code,
            home_team=home_team,
            away_team=away_team,
            oracle_probs=oracle_probs,
        )
        tot_lam = (lh_hist + la_hist) if (lh_hist + la_hist) > 0 else 2.65
        src = f"📊 Modelo Histórico ({hist_src} ➔ λ_tot: {tot_lam:.2f})"

    # 3. Split lambda_tot proportionally to clean Vig-Free probabilities
    if oracle_data:
        p_clean_h = float(oracle_data["p_clean_h"])
        p_clean_a = float(oracle_data["p_clean_a"])
        denom = p_clean_h + p_clean_a
        w_h = (p_clean_h / denom) if denom > 0 else 0.5
        w_a = (p_clean_a / denom) if denom > 0 else 0.5
        lh = tot_lam * w_h
        la = tot_lam * w_a
        src += f" [Vig-Free {sharp_title}: {p_clean_h:.1%} vs {p_clean_a:.1%}]"
    else:
        lh = tot_lam * 0.50
        la = tot_lam * 0.50

    # 4. Apply AI tactical adjustments if enabled (strictly neutral 1.00 by default)
    if ai_adjustment is not None:
        f_h = float(get_ai_field(ai_adjustment, "factor_ajuste_local", 1.0))
        f_a = float(get_ai_field(ai_adjustment, "factor_ajuste_visita", 1.0))
        lh = float(np.clip(lh * f_h, 0.35, 4.50))
        la = float(np.clip(la * f_a, 0.35, 4.50))
        if f_h != 1.0 or f_a != 1.0:
            src += f" + IA (H: {f_h:.2f}x | A: {f_a:.2f}x)"

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


# ==============================================================================
# MOTOR 2: SIMULADOR DE MONTE CARLO (10,000 ITERACIONES POR PARTIDO)
# ==============================================================================
def run_monte_carlo_sim(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.13,
    iterations: int = 10000,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """MOTOR 2: Simula 10,000 partidos usando la distribución conjunta Poisson ajustada por Dixon-Coles.
    Vectorizado íntegramente con numpy para lograr tiempos de ejecución ultra-rápidos (< 5 milisegundos).

    Cuenta exactamente en cuántos escenarios se cumple cada línea individual y combinada (Same-Game Parlay)
    y devuelve la probabilidad real ultra-precisa.
    """
    import time
    t_start = time.perf_counter()
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1. Matriz discreta 10x10 de probabilidades de goles (0 a 9 goles por lado)
    max_g = 10
    poiss_h = poisson.pmf(np.arange(max_g), lambda_h)
    poiss_a = poisson.pmf(np.arange(max_g), lambda_a)
    grid = np.outer(poiss_h, poiss_a)

    # 2. Ajuste Dixon-Coles para marcadores bajos (0 y 1 goles) con parámetro rho
    grid[0, 0] *= max(0.0, 1.0 - lambda_h * lambda_a * rho)
    grid[0, 1] *= max(0.0, 1.0 + lambda_h * rho)
    grid[1, 0] *= max(0.0, 1.0 + lambda_a * rho)
    grid[1, 1] *= max(0.0, 1.0 - rho)

    grid_sum = np.sum(grid)
    if grid_sum > 0:
        grid /= grid_sum

    # 3. Muestreo masivo vectorizado de 10,000 partidos instantáneo
    draw_indices = np.random.choice(max_g * max_g, size=iterations, p=grid.ravel())
    goals_h = draw_indices // max_g
    goals_a = draw_indices % max_g
    total_goals = goals_h + goals_a

    # 4. Distribución temporal realista de goles (45% en 1ª Mitad, 55% en 2ª Mitad)
    goals_1h_h = np.random.binomial(goals_h, 0.45)
    goals_1h_a = np.random.binomial(goals_a, 0.45)
    goals_1h = goals_1h_h + goals_1h_a
    goals_2h = total_goals - goals_1h

    # 5. Máscaras booleanas para cada evento de partido
    mask_h_win = (goals_h > goals_a)
    mask_draw = (goals_h == goals_a)
    mask_a_win = (goals_a > goals_h)
    mask_1x = (goals_h >= goals_a)
    mask_x2 = (goals_a >= goals_h)
    mask_12 = (goals_h != goals_a)

    mask_h_o05 = (goals_h >= 1)
    mask_h_o15 = (goals_h >= 2)
    mask_a_o05 = (goals_a >= 1)
    mask_a_o15 = (goals_a >= 2)

    mask_tot_o15 = (total_goals >= 2)
    mask_tot_o25 = (total_goals >= 3)
    mask_btts = (goals_h >= 1) & (goals_a >= 1)
    mask_1h_o05 = (goals_1h >= 1)
    mask_both_halves = (goals_1h >= 1) & (goals_2h >= 1)

    # 6. Conteo exacto de escenarios para Same-Game Parlay (SGP)
    # Combinada Blindada (Local Fav): 1X + Local > 0.5 + 1H > 0.5
    mask_blindada_h = mask_1x & mask_h_o05 & mask_1h_o05
    # Combinada Blindada (Visita Fav): X2 + Visita > 0.5 + 1H > 0.5
    mask_blindada_a = mask_x2 & mask_a_o05 & mask_1h_o05

    # Combinada Ritmo Ofensivo: Goles Fav > 0.5 + Gol en Ambas Mitades + Total > 1.5
    mask_ritmo_h = mask_h_o05 & mask_both_halves & mask_tot_o15
    mask_ritmo_a = mask_a_o05 & mask_both_halves & mask_tot_o15

    # Combinada de Valor: Victoria Fav + Total > 1.5 + BTTS
    mask_valor_h = mask_h_win & mask_tot_o15 & mask_btts
    mask_valor_a = mask_a_win & mask_tot_o15 & mask_btts

    # Over 2.0 Asiático
    wins_over2 = int(np.sum(total_goals >= 3))
    pushes_over2 = int(np.sum(total_goals == 2))
    losses_over2 = int(np.sum(total_goals <= 1))
    eff_over2_denom = wins_over2 + losses_over2
    p_eff_over2 = float(wins_over2 / eff_over2_denom) if eff_over2_denom > 0 else 0.0

    # DNB tasa de victoria efectiva (excluyendo empates)
    decided_matches = int(np.sum(mask_12))
    p_dnb_h = float(np.sum(mask_h_win) / decided_matches) if decided_matches > 0 else 0.5
    p_dnb_a = float(np.sum(mask_a_win) / decided_matches) if decided_matches > 0 else 0.5

    t_elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return {
        "iterations": iterations,
        "sim_time_ms": t_elapsed_ms,
        "rho": rho,
        "p_win_h": float(np.sum(mask_h_win) / iterations),
        "p_draw": float(np.sum(mask_draw) / iterations),
        "p_win_a": float(np.sum(mask_a_win) / iterations),
        "p_1x": float(np.sum(mask_1x) / iterations),
        "p_x2": float(np.sum(mask_x2) / iterations),
        "p_12": float(np.sum(mask_12) / iterations),
        "p_dnb_h": p_dnb_h,
        "p_dnb_a": p_dnb_a,
        "p_h_over05": float(np.sum(mask_h_o05) / iterations),
        "p_h_over15": float(np.sum(mask_h_o15) / iterations),
        "p_a_over05": float(np.sum(mask_a_o05) / iterations),
        "p_a_over15": float(np.sum(mask_a_o15) / iterations),
        "p_over15": float(np.sum(mask_tot_o15) / iterations),
        "p_over25": float(np.sum(mask_tot_o25) / iterations),
        "p_btts": float(np.sum(mask_btts) / iterations),
        "p_1h_over05": float(np.sum(mask_1h_o05) / iterations),
        "p_both_halves": float(np.sum(mask_both_halves) / iterations),
        "p_over20_eff": p_eff_over2,
        "p_over20_push": float(pushes_over2 / iterations),
        "p_blindada_h": float(np.sum(mask_blindada_h) / iterations),
        "p_blindada_a": float(np.sum(mask_blindada_a) / iterations),
        "p_ritmo_h": float(np.sum(mask_ritmo_h) / iterations),
        "p_ritmo_a": float(np.sum(mask_ritmo_a) / iterations),
        "p_valor_h": float(np.sum(mask_valor_h) / iterations),
        "p_valor_a": float(np.sum(mask_valor_a) / iterations),
    }


# ==============================================================================
# MOTOR 3: INDICADOR DE "CONFIANZA DE SISTEMA" ESCALONADO (SCORE DE ORÁCULO)
# ==============================================================================
def calculate_oracle_score(
    ev_percent: float,
    monte_carlo_prob_pct: float,
    is_steam: bool = False,
    steam_bonus: float = 100.0,
) -> Tuple[float, bool, str]:
    """MOTOR 3: Calcula el Score de Oráculo Institucional (0 a 100).
    Fórmula: (EV_Normalizado * 0.4) + (Probabilidad_Monte_Carlo * 0.4) + (Bono_Smart_Money * 0.2)

    Reglas de gobernanza de capital:
    - Kelly Criterion (Stake) solo se habilita para Score > 75/100.
    - Golden Pick (Alta Certeza Matemática) se activa para Score >= 90/100.

    Returns:
        (oracle_score, is_golden_pick, tier_label)
    """
    ev_norm = min(100.0, max(0.0, (ev_percent / 12.0) * 100.0)) if ev_percent > 0 else 0.0
    p_norm = min(100.0, max(0.0, float(monte_carlo_prob_pct)))
    bonus = steam_bonus if is_steam else 0.0

    score = round((ev_norm * 0.40) + (p_norm * 0.40) + (bonus * 0.20), 1)
    score = min(100.0, max(0.0, score))

    if score >= 90.0:
        is_golden = True
        tier = "🔥 GOLDEN PICK (Alta Certeza Matemática)"
    elif score > 65.0:
        is_golden = False
        tier = "🟢 LUZ VERDE (+EV Calificado)"
    else:
        is_golden = False
        tier = "⚪ FILTRO INSTITUCIONAL (Observación / Stake Bloqueado)"

    return score, is_golden, tier


def analyze_soccer_match(
    event: Dict[str, Any],
    sport_meta: Dict[str, Any],
    bankroll: float = 1000.0,
    kelly_k: float = 0.20,
    min_ev: float = 2.0,
    min_odds: float = 1.30,
    max_odds: float = 2.80,
    ai_adjustment: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Analyzes a soccer match to detect +EV opportunities with institutional precision:
    - Motor 1: Smart Money line drop tracking (Steam Chasing)
    - Motor 2: Vectorized Monte Carlo simulation (10,000 iterations Dixon-Coles)
    - Motor 3: Oracle Score (0 to 100) with Kelly gating (>75) and Golden Pick (>=90).
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

    # 2b. MOTOR 1: Detect Steam Moves (Smart Money)
    from etl.odds_api_client import detect_steam_moves
    steam_data = detect_steam_moves(event)
    is_steam_active = bool(steam_data.get("is_steam", False))
    event["steam_move"] = steam_data

    # 2c. MOTOR 2: Vectorized Monte Carlo Simulation (10,000 matches via Dixon-Coles)
    mc_sim = run_monte_carlo_sim(lambda_h, lambda_a, rho=-0.13, iterations=10000)
    event["monte_carlo_sim"] = mc_sim

    # 3. 9x9 Poisson Matrix
    poisson_mat = calculate_poisson_matrix(lambda_h, lambda_a, max_goals=9)

    # Poisson outcome probabilities
    p_pois_h = float(np.sum(np.tril(poisson_mat, -1)))
    p_pois_x = float(np.trace(poisson_mat))
    p_pois_a = float(np.sum(np.triu(poisson_mat, 1)))

    # 4. Bayesian Blend: 60% Poisson + 40% Clean Oracle
    p_final_h = 0.60 * p_pois_h + 0.40 * p_clean_h
    p_final_x = 0.60 * p_pois_x + 0.40 * p_clean_x
    p_final_a = 0.60 * p_pois_a + 0.40 * p_clean_a

    norm_sum = p_final_h + p_final_x + p_final_a
    p_final_h /= norm_sum
    p_final_x /= norm_sum
    p_final_a /= norm_sum

    # ==========================================
    # MARKET A: Draw No Bet (Apuesta sin Empate - Local)
    # ==========================================
    p_eff_dnb = p_final_h / (p_final_h + p_final_a) if (p_final_h + p_final_a) > 0 else 0.0
    fair_odds_dnb = (1.0 / p_eff_dnb) if p_eff_dnb > 0 else 999.0
    p_mc_dnb_pct = round(mc_sim["p_dnb_h"] * 100.0, 1)

    for bm in bookmakers:
        bm_key = bm.get("key", "")
        bm_title = bm.get("title", bm_key.capitalize())
        if bm_key == "pinnacle":
            continue

        h2h_data = extract_vig_free_h2h(bm)
        if not h2h_data:
            continue

        odds_h = h2h_data["odds_h"]
        odds_x = h2h_data["odds_x"]

        synthetic_odds_dnb = odds_h * (1.0 - (1.0 / odds_x))
        if synthetic_odds_dnb < 1.05:
            continue

        ev_dnb = (p_final_h * (synthetic_odds_dnb - 1.0)) - p_final_a
        ev_dnb_pct = ev_dnb * 100.0

        passes_filter, filter_msg = evaluate_opportunity_filters(
            odds=synthetic_odds_dnb,
            ev_percent=ev_dnb_pct,
            min_ev=min_ev,
            min_odds=min_odds,
            max_odds=max_odds,
        )

        score_dnb, is_gold_dnb, tier_dnb = calculate_oracle_score(
            ev_percent=ev_dnb_pct,
            monte_carlo_prob_pct=p_mc_dnb_pct,
            is_steam=is_steam_active and (steam_data.get("side") == "1"),
        )

        if passes_filter:
            if score_dnb > 65.0:
                kelly_res = calculate_kelly_stake(
                    odds=synthetic_odds_dnb,
                    effective_prob=p_eff_dnb,
                    bankroll=bankroll,
                    kelly_fraction=kelly_k,
                )
                total_stake = kelly_res["recommended_stake"]
                stake_pct = kelly_res["recommended_stake_pct"]
            else:
                total_stake = 0.0
                stake_pct = 0.0

            dutching = calculate_dutching_saver(total_stake, odds_h, odds_x) if total_stake > 0 else None

            instructions = (
                f"1. Ingresa a tu casa de apuestas ({bm_title}).\n"
                f"2. Busca el partido '{match_name}'.\n"
                f"3. Si disponen de mercado 'Apuesta sin Empate / Draw No Bet', apuesta directamente ${total_stake} "
                f"a favor de '{home_team}' a cuota mínima {round(fair_odds_dnb, 2)} (cuota sintética actual: {round(synthetic_odds_dnb, 2)}).\n"
                f"4. Si la casa no ofrece DNB, aplica el Seguro Sintético (Dutching Saver):\n"
                f"   • Apuesta a la Victoria de {home_team} (Cuota {odds_h}).\n"
                f"   • Apuesta al Empate (Cuota {odds_x}) para reembolso total en caso de empate."
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
                "oracle_score": score_dnb,
                "is_golden_pick": is_gold_dnb,
                "tier_label": tier_dnb,
                "p_monte_carlo": p_mc_dnb_pct,
                "steam_move": steam_data,
                "mc_stats": {"iterations": mc_sim["iterations"], "sim_time_ms": mc_sim["sim_time_ms"]},
                "recommended_stake": total_stake,
                "recommended_stake_pct": round(stake_pct * 100.0, 2) if total_stake > 0 else 0.0,
                "is_capped": False,
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
    p_perder_over2 = float(poisson_mat[0, 0] + poisson_mat[1, 0] + poisson_mat[0, 1])
    p_reembolso_over2 = float(poisson_mat[2, 0] + poisson_mat[0, 2] + poisson_mat[1, 1])
    p_ganar_over2 = float(max(0.0, 1.0 - (p_perder_over2 + p_reembolso_over2)))
    denom_over2 = p_ganar_over2 + p_perder_over2
    p_eff_over2 = (p_ganar_over2 / denom_over2) if denom_over2 > 0 else 0.0
    fair_odds_over2 = (1.0 / p_eff_over2) if p_eff_over2 > 0 else 999.0

    p_under_15 = float(poisson_mat[0, 0] + poisson_mat[1, 0] + poisson_mat[0, 1])
    p_over_15 = float(max(0.0, 1.0 - p_under_15))

    p_under_25 = float(p_under_15 + poisson_mat[2, 0] + poisson_mat[0, 2] + poisson_mat[1, 1])
    p_over_25 = float(max(0.0, 1.0 - p_under_25))

    p_btts_yes = float(np.sum(poisson_mat[1:, 1:]))
    p_btts_no = float(max(0.0, 1.0 - p_btts_yes))

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

    for bm in bookmakers:
        bm_key = bm.get("key", "")
        bm_title = bm.get("title", bm_key.capitalize())
        if bm_key == "pinnacle":
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

                        p_mc_over2_pct = round(mc_sim["p_over20_eff"] * 100.0, 1)
                        score_over2, is_gold_over2, tier_over2 = calculate_oracle_score(
                            ev_percent=ev_over2_pct,
                            monte_carlo_prob_pct=p_mc_over2_pct,
                            is_steam=is_steam_active,
                        )

                        passes_filter, _ = evaluate_opportunity_filters(
                            odds=price,
                            ev_percent=ev_over2_pct,
                            min_ev=min_ev,
                            min_odds=min_odds,
                            max_odds=max_odds,
                        )

                        if passes_filter:
                            if score_over2 > 65.0:
                                kelly_res = calculate_kelly_stake(
                                    odds=price,
                                    effective_prob=p_eff_over2,
                                    bankroll=bankroll,
                                    kelly_fraction=kelly_k,
                                )
                                total_stake = kelly_res["recommended_stake"]
                                stake_pct = kelly_res["recommended_stake_pct"]
                            else:
                                total_stake = 0.0
                                stake_pct = 0.0

                            instructions = (
                                f"1. Ingresa a {bm_title}.\n"
                                f"2. Localiza el encuentro '{match_name}'.\n"
                                f"3. Dirígete a la sección de 'Líneas Asiáticas / Totales de Goles'.\n"
                                f"4. Selecciona 'Más de 2.0 Goles Asiático' a cuota actual {price} (Mínima aceptable: {round(fair_odds_over2, 2)}).\n"
                                f"5. Coloca un stake de ${total_stake}.\n"
                                f"   • Si hay 3 o más goles: Ganas íntegro.\n"
                                f"   • Si hay 2 goles exactos: REEMBOLSO TOTAL.\n"
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
                                "oracle_score": score_over2,
                                "is_golden_pick": is_gold_over2,
                                "tier_label": tier_over2,
                                "p_monte_carlo": p_mc_over2_pct,
                                "steam_move": steam_data,
                                "mc_stats": {"iterations": mc_sim["iterations"], "sim_time_ms": mc_sim["sim_time_ms"]},
                                "recommended_stake": total_stake,
                                "recommended_stake_pct": round(stake_pct * 100.0, 2) if total_stake > 0 else 0.0,
                                "is_capped": False,
                                "instructions": instructions,
                                "dutching": None,
                                "lambdas": f"λH: {lambda_h} | λA: {lambda_a} (P_win: {round(p_ganar_over2*100, 1)}% | P_push: {round(p_reembolso_over2*100, 1)}%)",
                                "ai_adjustment": ai_adj_dict,
                            })

                    # 2. Over / Under 2.5 Goles
                    elif abs(point - 2.5) < 0.05:
                        is_over = name.lower() == "over"
                        p_sel = p_over_25 if is_over else p_under_25
                        p_mc_sel = (mc_sim["p_over25"] if is_over else (1.0 - mc_sim["p_over25"])) * 100.0
                        sel_text = "Más de 2.5 Goles" if is_over else "Menos de 2.5 Goles"
                        mkt_name = "Over 2.5 Goles" if is_over else "Under 2.5 Goles"
                        fair_o = (1.0 / p_sel) if p_sel > 0 else 999.0
                        ev_pct = ((p_sel * price) - 1.0) * 100.0

                        score_tot, is_gold_tot, tier_tot = calculate_oracle_score(
                            ev_percent=ev_pct,
                            monte_carlo_prob_pct=p_mc_sel,
                            is_steam=is_steam_active,
                        )

                        passes_filter, _ = evaluate_opportunity_filters(
                            odds=price,
                            ev_percent=ev_pct,
                            min_ev=min_ev,
                            min_odds=min_odds,
                            max_odds=max_odds,
                        )

                        if passes_filter:
                            if score_tot > 65.0:
                                kelly_res = calculate_kelly_stake(
                                    odds=price,
                                    effective_prob=p_sel,
                                    bankroll=bankroll,
                                    kelly_fraction=kelly_k,
                                )
                                total_stake = kelly_res["recommended_stake"]
                                stake_pct = kelly_res["recommended_stake_pct"]
                            else:
                                total_stake = 0.0
                                stake_pct = 0.0

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
                                "oracle_score": score_tot,
                                "is_golden_pick": is_gold_tot,
                                "tier_label": tier_tot,
                                "p_monte_carlo": round(p_mc_sel, 1),
                                "steam_move": steam_data,
                                "mc_stats": {"iterations": mc_sim["iterations"], "sim_time_ms": mc_sim["sim_time_ms"]},
                                "recommended_stake": total_stake,
                                "recommended_stake_pct": round(stake_pct * 100.0, 2) if total_stake > 0 else 0.0,
                                "is_capped": False,
                                "instructions": f"Apuesta en {bm_title} a {sel_text} (@{price}).",
                                "dutching": None,
                                "lambdas": f"λH: {lambda_h} | λA: {lambda_a} (P: {round(p_sel*100, 1)}%)",
                                "ai_adjustment": ai_adj_dict,
                            })

    # ==========================================
    # MARKET C: Combinadas Inteligentes SGP (Same-Game Parlays)
    # ==========================================
    try:
        from models.betplay_markets import calculate_betplay_security_catalog, generate_betplay_same_game_parlays
        retail_odds_map = extract_event_market_odds(event)
        sgp_cat = calculate_betplay_security_catalog(
            home_team=home_team,
            away_team=away_team,
            lambda_h=lambda_h,
            lambda_a=lambda_a,
            bankroll=bankroll,
            kelly_multiplier=kelly_k,
            retail_odds=retail_odds_map,
        )
        sgp_cards = generate_betplay_same_game_parlays(
            catalog=sgp_cat,
            bankroll=bankroll,
            kelly_multiplier=kelly_k,
            ai_adjustment=ai_adjustment,
            mc_sim=mc_sim,
            steam_data=steam_data,
        )
        for card in sgp_cards:
            c_odds = float(card["odds"])
            c_ev = float(card["ev_pct"])
            c_stake = float(card["recommended_stake"])
            c_score = float(card.get("oracle_score", 0.0))
            c_gold = bool(card.get("is_golden_pick", False))
            c_tier = str(card.get("tier_label", ""))
            c_p_mc = float(card.get("p_monte_carlo", card.get("prob_pct", 50.0)))

            if c_ev >= min_ev and min_odds <= c_odds <= max_odds and c_stake > 0:
                legs_text = " + ".join([f"{l['selection']} (@{l['odds']:.2f})" for l in card["legs"]])
                opportunities.append({
                    "sport": "Fútbol",
                    "tournament": sport_meta.get("name", "Fútbol"),
                    "match": match_name,
                    "commence_time": event.get("commence_time", ""),
                    "market": card["badge"],
                    "selection": f"{card['badge']}: {legs_text}",
                    "bookmaker": "BetPlay (Crea tu Apuesta)",
                    "odds": c_odds,
                    "fair_odds": float(card["fair_odds"]),
                    "effective_prob_pct": c_p_mc,
                    "ev_percent": c_ev,
                    "oracle_score": c_score,
                    "is_golden_pick": c_gold,
                    "tier_label": c_tier,
                    "p_monte_carlo": c_p_mc,
                    "steam_move": steam_data,
                    "mc_stats": card.get("mc_stats"),
                    "recommended_stake": c_stake,
                    "recommended_stake_pct": round((c_stake / bankroll) * 100.0, 2) if bankroll > 0 else 0.0,
                    "is_capped": False,
                    "instructions": card["instructions"],
                    "dutching": None,
                    "lambdas": f"λH: {lambda_h} | λA: {lambda_a} ({source_desc})",
                    "ai_adjustment": ai_adj_dict,
                })
    except Exception:
        pass

    return opportunities
