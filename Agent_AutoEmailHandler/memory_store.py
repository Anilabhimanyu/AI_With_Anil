from typing import List, Dict, Any
from pathlib import Path # import Path for file operations
import json

MEMORY_FILE = Path("memory.json")
MAX_HISTORY = 10

def load_memory() -> List[Dict[str, Any]]:
    """Load memory from a JSON file."""
    if Path(MEMORY_FILE).exists():
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(history: List[Dict[str, Any]]) -> None:
    # keep only the last MAX_HISTORY entries
    trimmed = history[-MAX_HISTORY:]
    MEMORY_FILE.write_text(json.dumps(trimmed, indent=2))
    
def append_to_memory(item: Dict[str, Any]) -> None:
    history = load_memory()
    history.append(item)
    save_memory(history)
    
