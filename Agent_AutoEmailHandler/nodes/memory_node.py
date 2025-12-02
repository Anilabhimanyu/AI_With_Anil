from typing import Dict, List
from memory_store import load_memory, append_to_memory

def merge_memory(email: Dict[str, str]) -> List[Dict[str, str]]:
    history = load_memory()
    # Optionally push the incoming email into memory for next steps
    append_to_memory({
    "from": email.get("from"),
    "to": email.get("to"),
    "subject": email.get("subject"),
    "body": email.get("body"),
    })
    return load_memory()