"""
LangGraph-based workflow for Smart Email Assistant
This file co-exists with main.py (sequential pipeline).
Here we implement the SAME workflow but using LangGraph nodes + edges.
"""

from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END

from nodes.classify_intent_node import classify_intent
from nodes.summarize_node import summarize_email
from nodes.memory_node import merge_memory
from nodes.generate_reply import generate_reply
from nodes.decision_node import decide_escalation
from llm_client import LLMClient
import os

# -----------------------------------------------------------------------------
# State Schema
# -----------------------------------------------------------------------------


class EmailState(TypedDict, total=False):
    email: Dict[str, Any]
    intent: str
    confidence: float
    summary: str
    memory: List[Dict[str, Any]]
    reply: Dict[str, Any]
    escalate: bool

# -----------------------------------------------------------------------------
# Node Wrappers
# -----------------------------------------------------------------------------

client = LLMClient()
CONFIG = {
    "TONE_STYLE": os.getenv("TONE_STYLE", "friendly"),
    "DEFAULT_FROM": os.getenv("DEFAULT_FROM", "support@yourapp.com"),
}


def node_classify(state: EmailState) -> EmailState:
    intent, confidence = classify_intent(state["email"], client)
    state["intent"] = intent
    state["confidence"] = confidence
    return state


def node_summarize(state: EmailState) -> EmailState:
    state["summary"] = summarize_email(state["email"], client)
    return state


def node_memory(state: EmailState) -> EmailState:
    state["memory"] = merge_memory(state["email"])
    return state


def node_reply(state: EmailState) -> EmailState:
    reply = generate_reply(
        state["intent"], state["summary"], state["memory"], client, CONFIG
    )
    state["reply"] = reply
    return state


def node_decide(state: EmailState) -> EmailState:
    state["escalate"] = decide_escalation(
        state["intent"], state["confidence"], state["reply"]
    )
    return state

# -----------------------------------------------------------------------------
# Build the LangGraph
# -----------------------------------------------------------------------------

graph = StateGraph(EmailState)

graph.add_node("classify", node_classify)
graph.add_node("summarize", node_summarize)
graph.add_node("memory", node_memory)
graph.add_node("reply", node_reply)
graph.add_node("decision", node_decide)

# Edges (flow): classify → summarize → memory → reply → decision → END
graph.add_edge("classify", "summarize")
graph.add_edge("summarize", "memory")
graph.add_edge("memory", "reply")
graph.add_edge("reply", "decision")
graph.add_edge("decision", END)

# Set entry point
graph.set_entry_point("classify")

# Compile application
app = graph.compile()

# -----------------------------------------------------------------------------
# Public run function
# -----------------------------------------------------------------------------

def run_graph_workflow(email_json: Dict[str, Any]):
    state = {"email": email_json}
    final_state = app.invoke(state)

    reply = final_state.get("reply", {})

    return {
        "subject": reply.get("subject"),
        "body": reply.get("body"),
        "to": email_json.get("from"),
        "from": CONFIG["DEFAULT_FROM"],
        "intent": final_state.get("intent"),
        "confidence": final_state.get("confidence"),
        "escalate": final_state.get("escalate"),
    }

# -----------------------------------------------------------------------------
# Example CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    from utils import load_json

    if len(sys.argv) < 2:
        print("Usage: python main_graph.py <input_json>")
        exit(1)

    payload = load_json(sys.argv[1])
    out = run_graph_workflow(payload)
    print(json.dumps(out, indent=2))
    print(app.get_graph().draw_mermaid())

