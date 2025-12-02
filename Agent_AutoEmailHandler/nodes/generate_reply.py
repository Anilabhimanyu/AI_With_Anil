from typing import Dict, List
from llm_client import LLMClient

REPLY_PROMPT = (
    "You are a support agent. Write a short, polite, and empathetic reply to the customer.\n"
    "Follow these strict rules:\n"
    "- Be brief (6–8 lines maximum).\n"
    "- First line: acknowledge the issue with empathy.\n"
    "- Second line: mention you checked the account (assume basic insights).\n"
    "- Third line: provide a clear next step.\n"
    "- Fourth line: offer escalation if issue continues.\n"
    "- Tone rules: Complaint → empathetic; Request → helpful; Inquiry → informative; Feedback → appreciative.\n"
    "- Do NOT ask the customer for additional information.\n"
    "- Do NOT include bullet points.\n"
    "- Do NOT add placeholders like [Your Name]. Use 'Support Team'.\n"
    "- Keep the reply focused and solution-oriented.\n\n"
    "Your output MUST be ONLY a JSON object in this exact format:\n"
    "{{\n"
    "  \"subject\": \"...\",\n"
    "  \"body\": \"...\"\n"
    "}}\n\n"
    "Intent: {intent}\n"
    "Summary: {summary}\n"
    "Conversation history: {history}\n"
)

def generate_reply(intent: str, summary: str, history: List[Dict[str, str]], client: LLMClient, config: Dict[str, str]) -> Dict[str, str]:
    history_text = "\n".join([f"From: {h.get('from')} - {h.get('body')}" for h in history])
    prompt = REPLY_PROMPT.format(intent=intent, summary=summary, history=history_text)
    # append config tone style
    if config.get("TONE_STYLE"):
        prompt += f"\nPreferred tone style: {config.get('TONE_STYLE')}\n"
    raw = client.run(prompt)
    # parse JSON safely
    try:
        import json
        parsed = json.loads(raw)
        return {"subject": parsed.get("subject"), "body": parsed.get("body")}
    except Exception:
        # fallback: simple templated reply
        subj = f"Re: {history[-1].get('subject') if history else 'your message'}"
        body = f"Hi {history[-1].get('from').split('@')[0].title() if history else ''},\n\nThanks for contacting support.\n\nBest,\nSupport Team"
        return {"subject": subj, "body": body}