"""Simple JSON-based long-term memory store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


MEMORY_FILENAME = "memory.json"


def _memory_path(data_dir: Path) -> Path:
    return data_dir / MEMORY_FILENAME


def load_memory_snapshot(data_dir: Path) -> Dict[str, Any]:
    """Load the current memory dictionary from disk.

    Args:
        data_dir: Directory where the memory file is stored.

    Returns:
        A dictionary representing long-term memory.
    """
    path = _memory_path(data_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_memory(data_dir: Path, key: str, value: Any) -> None:
    """Update a single key in the long-term memory store.

    Args:
        data_dir: Memory directory.
        key: Memory key, e.g. "user_preferences".
        value: Serializable value to store.
    """
    memory = load_memory_snapshot(data_dir)
    memory[key] = value
    path = _memory_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, indent=2), encoding="utf-8")
