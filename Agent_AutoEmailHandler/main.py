import sys
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from nodes.classify_intent_node import classify_intent
from nodes.summarize_node import summarize_email
from nodes.memory_node import merge_memory
from nodes.generate_reply import generate_reply
from nodes.decision_node import decide_escalation
from llm_client import LLMClient
from utils import load_json
import os
from typing import Dict, List

CONFIG = {
    "TONE_STYLE": os.getenv("TONE_STYLE", "friendly"),
    "DEFAULT_FROM": os.getenv("DEFAULT_FROM", "support@yourapp.com"),
}


def run_workflow(input_json: Dict[str, any]) -> Dict[str, any]:
    client = LLMClient()
    email = input_json

    # Parallelizable nodes: classification and summarization (run sequentially here)
    intent, confidence = classify_intent(email, client)
    summary = summarize_email(email, client)

    # Memory: merge current email with stored thread + provided history
    # If email contains 'history' key we append that too
    history = input_json.get("history", [])
    # seed memory from provided history
    for h in history:
        # note: append_to_memory is called inside merge_memory
        pass
    merged_memory = merge_memory(email)

    # Generate Reply
    reply = generate_reply(intent, summary, merged_memory, client, CONFIG)

    # Decision node for escalation
    escalate = decide_escalation(intent, confidence, reply)

    output = {
        "subject": reply.get("subject"),
        "body": reply.get("body"),
        "to": email.get("from"),
        "from": CONFIG["DEFAULT_FROM"],
        "intent": intent,
        "escalate": escalate,
        "confidence": confidence,
    }
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <input_json_file>")
        sys.exit(1)
    input_path = sys.argv[1]
    if not Path(input_path).exists():
        print("Input file not found:", input_path)
        sys.exit(1)
    payload = load_json(input_path)
    result = run_workflow(payload)
    print(json.dumps(result, indent=2))
