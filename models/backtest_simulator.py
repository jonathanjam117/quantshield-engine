"""Historical Backtesting and Performance Simulator for QuantShield.

Simulates model execution over completed historical matches:
- Loads historical match results with Pinnacle & recreational bookmaker odds.
- Performs chronological walk-forward execution without lookahead bias.
- Applies fractional Kelly and fixed stake criteria with dynamic bankroll.
- Resolves Draw No Bet and Asian Over 2.0 (with Push refunds).
- Computes comprehensive portfolio metrics: ROI, Yield, Win Rate, and Drawdown curve.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import poisson

from etl.historical_loader import get_cache_dir

AVAILABLE_BACKTEST_LEAGUES = {
    "N1": "Países Bajos - Eredivisie (Línea de Goles)",
    "D2": "Alemania - 2. Bundesliga (Alta Anotación)",
    "I2": "Italia - Serie B (Fortaleza Local)",
}


def load_historical_matches(league_codes: List[str]) -> pd.DataFrame:
    """Loads and standardizes historical match results from cached CSV files."""
    cache_dir = get_cache_dir()
    dfs: List[pd.DataFrame] = []

    for code in league_codes:
        file_path = os.path.join(cache_dir, f"{code}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df["LeagueCode"] = code
                dfs.append(df)
            except Exception:
                continue

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)

    # Clean and parse columns
    required_cols = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for col in required_cols:
        if col not in full_df.columns:
            return pd.DataFrame()

    full_df["FTHG"] = pd.to_numeric(full_df["FTHG"], errors="coerce")
    full_df["FTAG"] = pd.to_numeric(full_df["FTAG"], errors="coerce")
    full_df = full_df.dropna(subset=["FTHG", "FTAG"]).copy()

    # Parse date for chronological sorting
    if "Date" in full_df.columns:
        full_df["ParsedDate"] = pd.to_datetime(full_df["Date"], dayfirst=True, errors="coerce")
        full_df = full_df.sort_values(by="ParsedDate").reset_index(drop=True)
    else:
        full_df["ParsedDate"] = datetime.now()

    return full_df


def run_historical_backtest(
    matches_df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    min_ev: float = 3.0,
    min_odds: float = 1.40,
    max_odds: float = 2.25,
    stake_mode: str = "kelly",  # 'kelly' or 'fixed'
    kelly_k: float = 0.20,
    fixed_stake_pct: float = 0.025,
    max_stake_cap_pct: float = 0.035,
    include_dnb: bool = True,
    include_over2: bool = True,
) -> Dict[str, Any]:
    """Executes a chronological walk-forward backtest simulation.

    Returns summary metrics, performance curve, and detailed bets log.
    """
    if matches_df.empty:
        return {
            "summary": {
                "initial_bankroll": initial_bankroll,
                "final_bankroll": initial_bankroll,
                "net_profit": 0.0,
                "roi_pct": 0.0,
                "yield_pct": 0.0,
                "win_rate_pct": 0.0,
                "total_bets": 0,
                "won_bets": 0,
                "lost_bets": 0,
                "push_bets": 0,
                "discarded_matches": 0,
                "max_drawdown_pct": 0.0,
            },
            "bets_df": pd.DataFrame(),
            "history_df": pd.DataFrame(),
        }

    current_bankroll = float(initial_bankroll)
    placed_bets: List[Dict[str, Any]] = []
    discarded_count = 0

    # Team running stats trackers to prevent lookahead bias
    team_home_goals_for: Dict[str, int] = {}
    team_home_goals_against: Dict[str, int] = {}
    team_home_games: Dict[str, int] = {}

    team_away_goals_for: Dict[str, int] = {}
    team_away_goals_against: Dict[str, int] = {}
    team_away_games: Dict[str, int] = {}

    total_matches_played = 0
    total_home_goals = 0
    total_away_goals = 0

    for idx, row in matches_df.iterrows():
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        league_name = AVAILABLE_BACKTEST_LEAGUES.get(row.get("LeagueCode", "N1"), row.get("LeagueCode", "Liga"))
        date_str = str(row.get("Date", "Fecha Desconocida"))
        match_name = f"{home} vs {away}"

        # Extract Pinnacle odds (Oracle)
        ps_h = float(row.get("PSH", row.get("B365H", 2.0)))
        ps_x = float(row.get("PSD", row.get("B365D", 3.3)))
        ps_a = float(row.get("PSA", row.get("B365A", 3.8)))

        # Extract Recreational odds
        rec_h = float(row.get("B365H", ps_h))
        rec_x = float(row.get("B365D", ps_x))
        rec_a = float(row.get("B365A", ps_a))
        rec_over25 = float(row.get("B365>2.5", 1.85))

        # Vig-Free Oracle Probabilities
        inv_h = 1.0 / max(1.05, ps_h)
        inv_x = 1.0 / max(1.05, ps_x)
        inv_a = 1.0 / max(1.05, ps_a)
        sum_inv = inv_h + inv_x + inv_a
        p_clean_h = inv_h / sum_inv
        p_clean_x = inv_x / sum_inv
        p_clean_a = inv_a / sum_inv

        # Calculate Poisson lambdas from historical running stats
        if total_matches_played >= 10:
            avg_league_h = max(0.8, total_home_goals / total_matches_played)
            avg_league_a = max(0.6, total_away_goals / total_matches_played)

            h_games = team_home_games.get(home, 0)
            h_att = (team_home_goals_for.get(home, 1) / max(1, h_games)) / avg_league_h if h_games > 0 else 1.0
            h_def = (team_home_goals_against.get(home, 1) / max(1, h_games)) / avg_league_a if h_games > 0 else 1.0

            a_games = team_away_games.get(away, 0)
            a_att = (team_away_goals_for.get(away, 1) / max(1, a_games)) / avg_league_a if a_games > 0 else 1.0
            a_def = (team_away_goals_against.get(away, 1) / max(1, a_games)) / avg_league_h if a_games > 0 else 1.0

            lambda_h = float(np.clip(h_att * a_def * avg_league_h, 0.45, 4.0))
            lambda_a = float(np.clip(a_att * h_def * avg_league_a, 0.35, 3.5))
        else:
            # Baseline from clean oracle
            ratio = max(0.3, min((p_clean_h / max(0.05, p_clean_a)) ** 0.5 * 1.15, 3.5))
            total_g = 2.65
            lambda_h = float(np.clip(total_g * (ratio / (1.0 + ratio)), 0.45, 4.0))
            lambda_a = float(np.clip(total_g * (1.0 / (1.0 + ratio)), 0.35, 3.5))

        # Generate 9x9 Poisson matrix
        home_probs = [poisson.pmf(i, lambda_h) for i in range(9)]
        away_probs = [poisson.pmf(j, lambda_a) for j in range(9)]
        poisson_mat = np.outer(home_probs, away_probs)
        poisson_mat = poisson_mat / np.sum(poisson_mat)

        # Poisson outcomes
        p_pois_h = float(np.sum(np.tril(poisson_mat, -1)))
        p_pois_x = float(np.trace(poisson_mat))
        p_pois_a = float(np.sum(np.triu(poisson_mat, 1)))

        # Bayesian blend: 60% Poisson + 40% Clean Oracle
        p_final_h = 0.60 * p_pois_h + 0.40 * p_clean_h
        p_final_x = 0.60 * p_pois_x + 0.40 * p_clean_x
        p_final_a = 0.60 * p_pois_a + 0.40 * p_clean_a
        n_sum = p_final_h + p_final_x + p_final_a
        p_final_h /= n_sum
        p_final_x /= n_sum
        p_final_a /= n_sum

        bet_placed_in_match = False

        # ==========================================
        # 1. EVALUATE MARKET: Draw No Bet (Local)
        # ==========================================
        if include_dnb and rec_x > 1.0 and rec_h > 1.0:
            synthetic_dnb = rec_h * (1.0 - (1.0 / rec_x))
            p_eff_dnb = p_final_h / (p_final_h + p_final_a) if (p_final_h + p_final_a) > 0 else 0.0

            if synthetic_dnb > 1.0 and p_eff_dnb > 0:
                ev_dnb = ((p_final_h * (synthetic_dnb - 1.0)) - p_final_a) * 100.0
                fair_odds_dnb = 1.0 / p_eff_dnb

                if ev_dnb >= min_ev and (min_odds <= synthetic_dnb <= max_odds):
                    # Sizing: Kelly or Fixed
                    if stake_mode == "kelly":
                        b = synthetic_dnb - 1.0
                        edge = (b * p_eff_dnb) - (1.0 - p_eff_dnb)
                        if edge > 0:
                            f_star = (edge / b) * kelly_k
                            stake_pct = min(f_star, max_stake_cap_pct)
                        else:
                            stake_pct = 0.0
                    else:
                        stake_pct = min(fixed_stake_pct, max_stake_cap_pct)

                    if stake_pct > 0 and current_bankroll > 5.0:
                        stake = round(stake_pct * current_bankroll, 2)
                        # Determine Result
                        if fthg > ftag:
                            result = "Ganada"
                            profit = round(stake * (synthetic_dnb - 1.0), 2)
                        elif fthg == ftag:
                            result = "Reembolso / Push"
                            profit = 0.0
                        else:
                            result = "Perdida"
                            profit = -stake

                        current_bankroll = round(current_bankroll + profit, 2)
                        bet_placed_in_match = True

                        placed_bets.append({
                            "id": len(placed_bets) + 1,
                            "fecha": date_str,
                            "liga": league_name,
                            "partido": match_name,
                            "mercado": "Draw No Bet (Local)",
                            "seleccion": f"{home} (DNB)",
                            "cuota": round(synthetic_dnb, 2),
                            "cuota_justa": round(fair_odds_dnb, 2),
                            "ev_pct": round(ev_dnb, 2),
                            "stake": stake,
                            "marcador": f"{fthg} - {ftag}",
                            "resultado": result,
                            "beneficio": profit,
                            "bankroll": current_bankroll,
                        })

        # ==========================================
        # 2. EVALUATE MARKET: Over 2.0 Asiático
        # ==========================================
        if include_over2 and rec_over25 > 1.0:
            # Over 2.0 has refund on 2 goals; odds are ~8-12% lower than Over 2.5
            odds_over2 = round(max(1.35, rec_over25 * 0.90), 2)

            # Poisson probabilities for Over 2.0
            p_loss = float(poisson_mat[0, 0] + poisson_mat[1, 0] + poisson_mat[0, 1])
            p_push = float(poisson_mat[2, 0] + poisson_mat[0, 2] + poisson_mat[1, 1])
            p_win = float(max(0.0, 1.0 - (p_loss + p_push)))
            denom = p_win + p_loss
            p_eff_over2 = (p_win / denom) if denom > 0 else 0.0

            if odds_over2 > 1.0 and p_eff_over2 > 0:
                ev_over2 = ((p_win * (odds_over2 - 1.0)) - p_loss) * 100.0
                fair_odds_over2 = 1.0 / p_eff_over2

                if ev_over2 >= min_ev and (min_odds <= odds_over2 <= max_odds):
                    if stake_mode == "kelly":
                        b = odds_over2 - 1.0
                        edge = (b * p_eff_over2) - (1.0 - p_eff_over2)
                        if edge > 0:
                            f_star = (edge / b) * kelly_k
                            stake_pct = min(f_star, max_stake_cap_pct)
                        else:
                            stake_pct = 0.0
                    else:
                        stake_pct = min(fixed_stake_pct, max_stake_cap_pct)

                    if stake_pct > 0 and current_bankroll > 5.0:
                        stake = round(stake_pct * current_bankroll, 2)
                        tot_goals = fthg + ftag

                        if tot_goals >= 3:
                            result = "Ganada"
                            profit = round(stake * (odds_over2 - 1.0), 2)
                        elif tot_goals == 2:
                            result = "Reembolso / Push"
                            profit = 0.0
                        else:
                            result = "Perdida"
                            profit = -stake

                        current_bankroll = round(current_bankroll + profit, 2)
                        bet_placed_in_match = True

                        placed_bets.append({
                            "id": len(placed_bets) + 1,
                            "fecha": date_str,
                            "liga": league_name,
                            "partido": match_name,
                            "mercado": "Over 2.0 Asiático",
                            "seleccion": "Más de 2.0 Goles",
                            "cuota": odds_over2,
                            "cuota_justa": round(fair_odds_over2, 2),
                            "ev_pct": round(ev_over2, 2),
                            "stake": stake,
                            "marcador": f"{fthg} - {ftag}",
                            "resultado": result,
                            "beneficio": profit,
                            "bankroll": current_bankroll,
                        })

        if not bet_placed_in_match:
            discarded_count += 1

        # Update historical trackers for subsequent matches
        team_home_goals_for[home] = team_home_goals_for.get(home, 0) + fthg
        team_home_goals_against[home] = team_home_goals_against.get(home, 0) + ftag
        team_home_games[home] = team_home_games.get(home, 0) + 1

        team_away_goals_for[away] = team_away_goals_for.get(away, 0) + ftag
        team_away_goals_against[away] = team_away_goals_against.get(away, 0) + fthg
        team_away_games[away] = team_away_games.get(away, 0) + 1

        total_matches_played += 1
        total_home_goals += fthg
        total_away_goals += ftag

    # Performance summary calculation
    bets_df = pd.DataFrame(placed_bets)
    if not bets_df.empty:
        total_bets = len(bets_df)
        won_bets = int((bets_df["resultado"] == "Ganada").sum())
        lost_bets = int((bets_df["resultado"] == "Perdida").sum())
        push_bets = int((bets_df["resultado"] == "Reembolso / Push").sum())

        decided = won_bets + lost_bets
        win_rate = (won_bets / decided * 100.0) if decided > 0 else 0.0

        total_staked = float(bets_df["stake"].sum())
        net_profit = round(current_bankroll - initial_bankroll, 2)
        yield_pct = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0
        roi_pct = (net_profit / initial_bankroll * 100.0) if initial_bankroll > 0 else 0.0

        # Drawdown calculation
        history_df = bets_df[["id", "fecha", "partido", "mercado", "resultado", "beneficio", "bankroll"]].copy()
        history_df["peak"] = history_df["bankroll"].cummax()
        history_df["drawdown_pct"] = ((history_df["bankroll"] - history_df["peak"]) / history_df["peak"]) * 100.0
        max_dd = float(history_df["drawdown_pct"].min())
    else:
        total_bets = won_bets = lost_bets = push_bets = 0
        win_rate = yield_pct = roi_pct = max_dd = 0.0
        net_profit = 0.0
        history_df = pd.DataFrame()

    return {
        "summary": {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": current_bankroll,
            "net_profit": net_profit,
            "roi_pct": round(roi_pct, 2),
            "yield_pct": round(yield_pct, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_bets": total_bets,
            "won_bets": won_bets,
            "lost_bets": lost_bets,
            "push_bets": push_bets,
            "discarded_matches": discarded_count,
            "max_drawdown_pct": round(max_dd, 2),
        },
        "bets_df": bets_df,
        "history_df": history_df,
    }
