"""Kelly Risk Engine for QuantShield.

Implements Fractional Kelly Criterion and strict risk management caps:
- f* = (((b * p) - q) / b) * k
- Efficient Odds Range: 1.40 - 2.25
- Max Stake Cap: 3.5% of total bankroll
- Minimum EV threshold filter (default >= 3.0%)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def calculate_kelly_stake(
    odds: float,
    effective_prob: float,
    bankroll: float,
    kelly_fraction: float = 0.20,
    max_stake_cap_pct: float = 0.035,
) -> Dict[str, Any]:
    """Calculates optimal fractional Kelly stake and applies hard safety limits.

    Parameters:
        odds: Decimal odds (e.g. 1.85)
        effective_prob: Model's effective winning probability (0.0 to 1.0)
        bankroll: Total bankroll amount
        kelly_fraction: Multiplier k (default 0.20 = 1/5th Kelly)
        max_stake_cap_pct: Maximum allowed bankroll % per single bet (default 0.035 = 3.5%)

    Returns:
        Dict with:
        - raw_kelly_fraction: Uncapped Kelly percentage
        - recommended_stake_pct: Final constrained fraction (0.0 to max_stake_cap_pct)
        - recommended_stake: Exact dollar/currency amount to wager
        - is_capped: Boolean indicating if the hard cap was triggered
    """
    if odds <= 1.0 or effective_prob <= 0.0 or effective_prob >= 1.0 or bankroll <= 0:
        return {
            "raw_kelly_fraction": 0.0,
            "recommended_stake_pct": 0.0,
            "recommended_stake": 0.0,
            "is_capped": False,
        }

    b = odds - 1.0
    p = effective_prob
    q = 1.0 - p

    # Standard Kelly formula: (b*p - q) / b
    edge = (b * p) - q
    if edge <= 0:
        return {
            "raw_kelly_fraction": 0.0,
            "recommended_stake_pct": 0.0,
            "recommended_stake": 0.0,
            "is_capped": False,
        }

    full_kelly = edge / b
    fractional_kelly = full_kelly * kelly_fraction

    # Apply 3.5% hard cap
    is_capped = fractional_kelly > max_stake_cap_pct
    final_pct = min(fractional_kelly, max_stake_cap_pct)
    final_stake = round(final_pct * bankroll, 2)

    return {
        "raw_kelly_fraction": round(fractional_kelly, 4),
        "recommended_stake_pct": round(final_pct, 4),
        "recommended_stake": final_stake,
        "is_capped": is_capped,
    }


def evaluate_opportunity_filters(
    odds: float,
    ev_percent: float,
    min_ev: float = 3.0,
    min_odds: float = 1.40,
    max_odds: float = 2.25,
) -> Tuple[bool, str]:
    """Checks whether an opportunity meets QuantShield's hard criteria:
    - EV >= min_ev (default 3.0%)
    - Efficient odds range between 1.40 and 2.25
    """
    if ev_percent < min_ev:
        return False, f"EV de {ev_percent:.2f}% es inferior al mínimo requerido ({min_ev:.1f}%)"

    if odds < min_odds or odds > max_odds:
        return False, f"Cuota {odds:.2f} fuera del rango eficiente ({min_odds:.2f} - {max_odds:.2f})"

    return True, "Cumple todos los criterios de valor y seguridad"
