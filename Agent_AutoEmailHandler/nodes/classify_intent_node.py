from typing import Dict, Tuple
from llm_client import LLMClient


CLASSIFY_PROMPT = (
    "Classify the intent of this email as one of: complaint, request, feedback, inquiry."
    "\nEmail:\n\"\"\"\n{body}\n\"\"\"\n\n"
    "Return a JSON object: {{\"intent\": <one of the labels>, \"confidence\": <0-1>}}"
)

def classify_intent(email: Dict[str, str], client: LLMClient) -> Tuple[str, float]:
    prompt = CLASSIFY_PROMPT.format(body=email.get("body", ""))
    intent, confidence = client.classify_with_confidence(prompt)
    return intent.lower(), confidence