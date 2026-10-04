"""Tiny JSON-file history of analyses (no database needed)."""
import json
from datetime import datetime
from .config import HISTORY_FILE


def load_history() -> list:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def add_record(record: dict) -> None:
    hist = load_history()
    record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), **record}
    hist.append(record)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(hist[-500:], indent=1), encoding="utf-8")


def clear_history() -> None:
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
