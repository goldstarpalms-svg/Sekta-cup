from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any

PREDICTIONS_FILE = Path("data/ai_predictions.json")


def load_predictions():
    if not PREDICTIONS_FILE.exists():
        return []
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_predictions(predictions):
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, indent=2, default=str)


def add_prediction(prediction_data):
    """Auto-log every AI prediction shown to user."""
    predictions = load_predictions()
    
    # Check if this match already predicted (avoid duplicates within 1 hour)
    match_id = prediction_data.get("match_id", "")
    now = datetime.utcnow()
    
    for existing in predictions:
        if existing.get("match_id") == match_id and existing.get("status") == "pending":
            existing_time = datetime.fromisoformat(existing["logged_at"])
            hours_diff = (now - existing_time).total_seconds() / 3600
            if hours_diff < 1:
                return existing  # Already logged recently
    
    prediction = {
        "id": f"pred_{now.strftime('%Y%m%d_%H%M%S_%f')}",
        "logged_at": now.isoformat(),
        "status": "pending",
        **prediction_data,
    }
    predictions.append(prediction)
    save_predictions(predictions)
    return prediction


def update_prediction_result(match_id, actual_winner, actual_total_points=None, actual_first_set_total=None, actual_sets_played=None):
    """Update prediction with actual result."""
    predictions = load_predictions()
    updated = 0
    
    for pred in predictions:
        if pred.get("match_id") == match_id and pred.get("status") == "pending":
            pred["status"] = "settled"
            pred["actual_winner"] = actual_winner
            pred["winner_correct"] = pred.get("predicted_winner") == actual_winner
            
            if actual_total_points is not None:
                pred["actual_total_points"] = actual_total_points
                if "total_pick" in pred:
                    over_pick = "Over" in pred["total_pick"]
                    actual_over = actual_total_points > 75.5
                    pred["total_correct"] = over_pick == actual_over
            
            if actual_first_set_total is not None:
                pred["actual_first_set_total"] = actual_first_set_total
                if "first_pick" in pred:
                    over_pick = "Over" in pred["first_pick"]
                    actual_over = actual_first_set_total > 18.5
                    pred["first_correct"] = over_pick == actual_over
            
            if actual_sets_played is not None:
                pred["actual_sets_played"] = actual_sets_played
                if "sets_pick" in pred:
                    over_pick = "Over" in pred["sets_pick"]
                    actual_over = actual_sets_played > 3.5
                    pred["sets_correct"] = over_pick == actual_over
            
            pred["settled_at"] = datetime.utcnow().isoformat()
            updated += 1
    
    if updated > 0:
        save_predictions(predictions)
    return updated


def calculate_track_record():
    """Calculate overall AI prediction accuracy."""
    predictions = load_predictions()
    settled = [p for p in predictions if p.get("status") == "settled"]
    
    if not settled:
        return {
            "total": 0,
            "pending": len([p for p in predictions if p.get("status") == "pending"]),
            "settled": 0,
            "winner_accuracy": 0,
            "total_accuracy": 0,
            "first_set_accuracy": 0,
            "sets_accuracy": 0,
            "high_conf_accuracy": 0,
        }
    
    winner_correct = [p for p in settled if p.get("winner_correct")]
    total_settled = [p for p in settled if "total_correct" in p]
    total_correct = [p for p in total_settled if p.get("total_correct")]
    first_settled = [p for p in settled if "first_correct" in p]
    first_correct = [p for p in first_settled if p.get("first_correct")]
    sets_settled = [p for p in settled if "sets_correct" in p]
    sets_correct = [p for p in sets_settled if p.get("sets_correct")]
    
    high_conf = [p for p in settled if p.get("confidence", 0) >= 0.70]
    high_conf_correct = [p for p in high_conf if p.get("winner_correct")]
    
    return {
        "total": len(predictions),
        "pending": len([p for p in predictions if p.get("status") == "pending"]),
        "settled": len(settled),
        "winner_accuracy": (len(winner_correct) / len(settled) * 100) if settled else 0,
        "winner_wins": len(winner_correct),
        "winner_losses": len(settled) - len(winner_correct),
        "total_accuracy": (len(total_correct) / len(total_settled) * 100) if total_settled else 0,
        "total_settled": len(total_settled),
        "first_set_accuracy": (len(first_correct) / len(first_settled) * 100) if first_settled else 0,
        "first_settled": len(first_settled),
        "sets_accuracy": (len(sets_correct) / len(sets_settled) * 100) if sets_settled else 0,
        "sets_settled": len(sets_settled),
        "high_conf_accuracy": (len(high_conf_correct) / len(high_conf) * 100) if high_conf else 0,
        "high_conf_total": len(high_conf),
    }
