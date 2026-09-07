import json
from pathlib import Path
from typing import List, Optional, Dict, Any

HISTORY_FILE = Path("data/evaluation/metrics_history.json")

def load_metrics_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_metrics(scores: Dict[str, Any]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_metrics_history()
    history.append(scores)
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def get_latest_metrics() -> Optional[Dict[str, Any]]:
    history = load_metrics_history()
    return history[-1] if history else None