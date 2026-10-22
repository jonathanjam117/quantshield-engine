"""Database Manager for QuantShield Engine.

Handles SQLite operations for bet tracking, CLV (Closing Line Value), and Bankroll audit.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

DEFAULT_DB_FILENAME = "quantshield.db"


def get_default_db_path() -> str:
    """Returns the absolute path to the default quantshield.db file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, DEFAULT_DB_FILENAME)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates a sqlite3 connection with row dictionary access."""
    path = db_path or get_default_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes the database schema if tables do not exist."""
    path = db_path or get_default_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sport TEXT NOT NULL,
                tournament TEXT NOT NULL,
                match TEXT NOT NULL,
                market TEXT NOT NULL,
                selection TEXT NOT NULL,
                bookmaker TEXT NOT NULL,
                odds REAL NOT NULL,
                fair_odds REAL,
                ev_percent REAL NOT NULL,
                stake REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pendiente',
                net_profit REAL DEFAULT 0.0,
                clv REAL DEFAULT 0.0,
                notes TEXT DEFAULT '',
                ai_analysis TEXT DEFAULT ''
            );
            """
        )
        # Migrate existing table if ai_analysis is missing
        cursor.execute("PRAGMA table_info(bets)")
        cols = [c[1] for c in cursor.fetchall()]
        if "ai_analysis" not in cols:
            try:
                cursor.execute("ALTER TABLE bets ADD COLUMN ai_analysis TEXT DEFAULT ''")
            except Exception:
                pass
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bankroll_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_bankroll REAL NOT NULL DEFAULT 1000.0,
                currency TEXT NOT NULL DEFAULT 'USD',
                updated_at TEXT NOT NULL
            );
            """
        )
        # Seed initial setting if not present
        cursor.execute("SELECT COUNT(*) FROM bankroll_settings WHERE id = 1")
        if cursor.fetchone()[0] == 0:
            now_iso = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO bankroll_settings (id, initial_bankroll, currency, updated_at) VALUES (1, 1000.0, 'USD', ?)",
                (now_iso,),
            )
        conn.commit()


def add_bet(
    sport: str,
    tournament: str,
    match: str,
    market: str,
    selection: str,
    bookmaker: str,
    odds: float,
    fair_odds: Optional[float],
    ev_percent: float,
    stake: float,
    clv: Optional[float] = 0.0,
    notes: str = "",
    status: str = "Pendiente",
    db_path: Optional[str] = None,
    ai_analysis: str = "",
) -> int:
    """Inserts a new bet into the database and returns its ID."""
    init_db(db_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    net_profit = 0.0
    if status == "Ganada":
        net_profit = stake * (odds - 1.0)
    elif status == "Perdida":
        net_profit = -stake
    elif status == "Reembolso / Push":
        net_profit = 0.0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO bets (
                timestamp, sport, tournament, match, market, selection,
                bookmaker, odds, fair_odds, ev_percent, stake, status, net_profit, clv, notes, ai_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                sport,
                tournament,
                match,
                market,
                selection,
                bookmaker,
                float(odds),
                float(fair_odds) if fair_odds is not None else None,
                float(ev_percent),
                float(stake),
                status,
                float(net_profit),
                float(clv or 0.0),
                notes,
                ai_analysis,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def update_bet_status(
    bet_id: int,
    status: str,
    net_profit: Optional[float] = None,
    clv: Optional[float] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Updates the status and profit of a bet based on QuantShield rules:
    - Ganada: stake * (odds - 1)
    - Perdida: -stake
    - Reembolso / Push: 0.0
    - Pendiente: 0.0
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT odds, stake FROM bets WHERE id = ?", (bet_id,))
        row = cursor.fetchone()
        if not row:
            return False

        odds = float(row["odds"])
        stake = float(row["stake"])

        if net_profit is None:
            if status == "Ganada":
                computed_profit = stake * (odds - 1.0)
            elif status == "Perdida":
                computed_profit = -stake
            elif status == "Reembolso / Push":
                computed_profit = 0.0
            else:
                computed_profit = 0.0
        else:
            computed_profit = net_profit

        if clv is not None:
            cursor.execute(
                "UPDATE bets SET status = ?, net_profit = ?, clv = ? WHERE id = ?",
                (status, computed_profit, float(clv), bet_id),
            )
        else:
            cursor.execute(
                "UPDATE bets SET status = ?, net_profit = ? WHERE id = ?",
                (status, computed_profit, bet_id),
            )
        conn.commit()
        return True


def delete_bet(bet_id: int, db_path: Optional[str] = None) -> bool:
    """Deletes a bet record by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_all_bets(db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieves all bets as a pandas DataFrame."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM bets ORDER BY id DESC", conn)
        return df


def set_initial_bankroll(amount: float, currency: str = "USD", db_path: Optional[str] = None) -> None:
    """Updates the configured initial bankroll."""
    init_db(db_path)
    now_iso = datetime.now().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE bankroll_settings SET initial_bankroll = ?, currency = ?, updated_at = ? WHERE id = 1",
            (float(amount), currency, now_iso),
        )
        conn.commit()


def get_bankroll_settings(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves the stored bankroll settings."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT initial_bankroll, currency FROM bankroll_settings WHERE id = 1")
        row = cursor.fetchone()
        if row:
            return {"initial_bankroll": float(row["initial_bankroll"]), "currency": row["currency"]}
        return {"initial_bankroll": 1000.0, "currency": "USD"}


def get_bankroll_metrics(
    initial_bankroll: Optional[float] = None, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Calculates bankroll KPIs, yield, win rate, and time-series for audit."""
    init_db(db_path)
    settings = get_bankroll_settings(db_path)
    init_bankroll = initial_bankroll if initial_bankroll is not None else settings["initial_bankroll"]

    df = get_all_bets(db_path)
    if df.empty:
        return {
            "initial_bankroll": init_bankroll,
            "current_bankroll": init_bankroll,
            "net_profit": 0.0,
            "yield_pct": 0.0,
            "roi_pct": 0.0,
            "win_rate_pct": 0.0,
            "total_bets": 0,
            "settled_bets": 0,
            "pending_bets": 0,
            "won_bets": 0,
            "lost_bets": 0,
            "push_bets": 0,
            "history_df": pd.DataFrame(),
        }

    # Count by status
    total_bets = len(df)
    won_mask = df["status"] == "Ganada"
    lost_mask = df["status"] == "Perdida"
    push_mask = df["status"] == "Reembolso / Push"
    pending_mask = df["status"] == "Pendiente"

    won_bets = int(won_mask.sum())
    lost_bets = int(lost_mask.sum())
    push_bets = int(push_mask.sum())
    pending_bets = int(pending_mask.sum())
    settled_bets = won_bets + lost_bets + push_bets

    # Profits & Stakes
    net_profit = float(df[~pending_mask]["net_profit"].sum())
    total_staked_settled = float(df[~pending_mask]["stake"].sum())
    current_bankroll = init_bankroll + net_profit

    # Yield & ROI
    yield_pct = (net_profit / total_staked_settled * 100.0) if total_staked_settled > 0 else 0.0
    roi_pct = (net_profit / init_bankroll * 100.0) if init_bankroll > 0 else 0.0

    # Win Rate: Won / (Won + Lost)
    decided_bets = won_bets + lost_bets
    win_rate_pct = (won_bets / decided_bets * 100.0) if decided_bets > 0 else 0.0

    # History curve sorted chronologically
    settled_df = df[~pending_mask].sort_values(by="id", ascending=True).copy()
    if not settled_df.empty:
        settled_df["cumulative_profit"] = settled_df["net_profit"].cumsum()
        settled_df["bankroll_curve"] = init_bankroll + settled_df["cumulative_profit"]
        # Drawdown calculation
        peak = settled_df["bankroll_curve"].cummax()
        settled_df["peak"] = peak
        settled_df["drawdown_val"] = settled_df["bankroll_curve"] - peak
        settled_df["drawdown_pct"] = (settled_df["drawdown_val"] / peak) * 100.0
    else:
        settled_df = pd.DataFrame()

    return {
        "initial_bankroll": init_bankroll,
        "current_bankroll": current_bankroll,
        "net_profit": net_profit,
        "yield_pct": yield_pct,
        "roi_pct": roi_pct,
        "win_rate_pct": win_rate_pct,
        "total_bets": total_bets,
        "settled_bets": settled_bets,
        "pending_bets": pending_bets,
        "won_bets": won_bets,
        "lost_bets": lost_bets,
        "push_bets": push_bets,
        "history_df": settled_df,
    }
