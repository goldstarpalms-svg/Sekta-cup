from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

BANKROLL_FILE = Path("data/bankroll_state.json")

DEFAULT_STATE = {
    "starting_bankroll": 20000,
    "current_bankroll": 20000,
    "currency": "NGN",
    "currency_symbol": "₦",
    "daily_loss_limit_pct": 10,
    "kelly_fraction": 0.25,
    "min_confidence": 0.65,
    "min_stake": 100,
    "max_stake_pct": 5,
    "created_at": None,
    "updated_at": None,
    "bets": [],
    "daily_pnl": {},
    "streak": {"current": 0, "type": "none", "best_win": 0, "worst_loss": 0},
}


def load_bankroll_state():
    if not BANKROLL_FILE.exists():
        state = DEFAULT_STATE.copy()
        state["created_at"] = datetime.utcnow().isoformat()
        state["updated_at"] = datetime.utcnow().isoformat()
        save_bankroll_state(state)
        return state
    try:
        with open(BANKROLL_FILE, "r") as f:
            state = json.load(f)
        for key, value in DEFAULT_STATE.items():
            if key not in state:
                state[key] = value
        return state
    except Exception:
        return DEFAULT_STATE.copy()


def save_bankroll_state(state):
    BANKROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.utcnow().isoformat()
    with open(BANKROLL_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def kelly_stake(bankroll, win_probability, decimal_odds, kelly_fraction=0.25, min_stake=100, max_stake_pct=5):
    if win_probability <= 0 or win_probability >= 1:
        return {"stake": 0, "kelly_pct": 0, "adjusted_kelly_pct": 0, "max_stake": 0, "reason": "Invalid probability"}
    if decimal_odds <= 1:
        return {"stake": 0, "kelly_pct": 0, "adjusted_kelly_pct": 0, "max_stake": 0, "reason": "Invalid odds"}
    b = decimal_odds - 1
    p = win_probability
    q = 1 - p
    kelly_pct = (b * p - q) / b
    if kelly_pct <= 0:
        return {"stake": 0, "kelly_pct": 0, "adjusted_kelly_pct": 0, "max_stake": 0, "reason": "No edge"}
    adjusted_kelly = kelly_pct * kelly_fraction
    stake = bankroll * adjusted_kelly
    max_stake = bankroll * (max_stake_pct / 100)
    stake = min(stake, max_stake)
    if stake < min_stake:
        stake = 0
    stake = round(stake / 50) * 50
    return {
        "stake": stake,
        "kelly_pct": round(kelly_pct * 100, 2),
        "adjusted_kelly_pct": round(adjusted_kelly * 100, 2),
        "max_stake": max_stake,
        "reason": "OK",
    }


def fair_odds_from_probability(probability):
    if probability <= 0:
        return 0
    return round(1 / probability, 2)


def add_bet(state, bet_data):
    bet_id = f"bet_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    bet = {
        "id": bet_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "pending",
        **bet_data,
    }
    state["bets"].append(bet)
    save_bankroll_state(state)
    return bet


def settle_bet(state, bet_id, won, actual_odds=None):
    for bet in state["bets"]:
        if bet["id"] == bet_id:
            bet["status"] = "settled"
            bet["won"] = won
            odds = actual_odds if actual_odds else bet.get("odds", 0)
            stake = bet.get("stake", 0)
            if won:
                bet["profit_loss"] = stake * (odds - 1)
                state["current_bankroll"] += bet["profit_loss"]
                if state["streak"]["type"] == "win":
                    state["streak"]["current"] += 1
                else:
                    state["streak"] = {
                        "current": 1,
                        "type": "win",
                        "best_win": max(state["streak"].get("best_win", 0), bet["profit_loss"]),
                        "worst_loss": state["streak"].get("worst_loss", 0)
                    }
            else:
                bet["profit_loss"] = -stake
                state["current_bankroll"] -= stake
                if state["streak"]["type"] == "loss":
                    state["streak"]["current"] += 1
                else:
                    state["streak"] = {
                        "current": 1,
                        "type": "loss",
                        "best_win": state["streak"].get("best_win", 0),
                        "worst_loss": min(state["streak"].get("worst_loss", 0), -stake)
                    }
            save_bankroll_state(state)
            return bet
    return None


def calculate_daily_pnl(state, date_str):
    pnl = 0
    for bet in state["bets"]:
        bet_date = bet["timestamp"][:10]
        if bet_date == date_str and bet["status"] == "settled":
            pnl += bet.get("profit_loss", 0)
    return pnl


def calculate_stats(state):
    settled = [b for b in state["bets"] if b["status"] == "settled"]
    pending = [b for b in state["bets"] if b["status"] == "pending"]
    won = [b for b in settled if b.get("won", False)]
    total_staked = sum(b.get("stake", 0) for b in settled)
    total_profit = sum(b.get("profit_loss", 0) for b in settled)
    return {
        "total_bets": len(state["bets"]),
        "settled_bets": len(settled),
        "pending_bets": len(pending),
        "won_bets": len(won),
        "lost_bets": len(settled) - len(won),
        "win_rate": (len(won) / len(settled) * 100) if settled else 0,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": (total_profit / total_staked * 100) if total_staked > 0 else 0,
    }


def format_currency(amount, symbol="₦"):
    return f"{symbol}{amount:,.0f}"


def check_loss_limit(state):
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_pnl = calculate_daily_pnl(state, today_str)
    daily_limit = state["current_bankroll"] * (state["daily_loss_limit_pct"] / 100)
    if today_pnl <= -daily_limit:
        return {"exceeded": True, "loss": today_pnl, "limit": -daily_limit}
    return {"exceeded": False, "loss": today_pnl, "limit": -daily_limit}
