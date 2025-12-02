import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key="Paste_Your_Gemini_Key_Here") #(api_key=os.getenv("GEMINI_API_KEY"))

class LLMClient:
    def __init__(self, model_name="gemini-2.5-flash-lite-preview-09-2025"):
        self.model = genai.GenerativeModel(model_name)

    def run(self, prompt: str) -> str:
        resp = self.model.generate_content(prompt)
        return resp.text

    def classify_with_confidence(self, prompt: str):
        formatted = prompt + "\nReturn JSON only: {\"intent\":..., \"confidence\":...}"
        resp = self.model.generate_content(formatted).text
        import json
        try:
            d = json.loads(resp)
            return d["intent"].lower(), float(d["confidence"])
        except:
            return "inquiry", 0.6
