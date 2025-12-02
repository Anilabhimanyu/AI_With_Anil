from typing import Dict, Any

def decide_escalation(intent: str, confidence: float, reply: Dict[str, Any]) -> bool:
    # Example rule: escalate if complaint and confidence < 0.8
    if intent == "complaint" and confidence < 0.8:
        return True
    # else, if reply contains words asking to escalate explicitly
    if "escalate" in (reply.get("body") or "").lower():
        return True
    return False