from typing import Dict
from llm_client import LLMClient

SUMMARIZE_PROMPT = (
    "Summarize the email briefly in 2-3 lines, focusing on the sender's main point and tone."
    "\nEmail:\n\"\"\"\n{body}\n\"\"\""
)

def summarize_email(email: Dict[str, str], client: LLMClient) -> str:
    prompt = SUMMARIZE_PROMPT.format(body=email.get("body", ""))
    return client.run(prompt).strip()