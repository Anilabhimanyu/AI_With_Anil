from typing import Any, Dict
import json
from pathlib import Path

def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

def save_json(path: str, data: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2))