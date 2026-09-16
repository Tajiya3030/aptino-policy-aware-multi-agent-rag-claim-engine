import os

class GeminiLLMClient:
    """Client for Gemini 2.5 Flash using google-genai SDK with deterministic fallback."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = "gemini-2.5-flash"
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[GeminiLLMClient] Could not initialize google.genai: {e}")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        if self.is_available:
            try:
                config = {}
                if system_instruction:
                    config["system_instruction"] = system_instruction
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                return response.text
            except Exception as e:
                print(f"[GeminiLLMClient] Gemini API call failed: {e}. Falling back to deterministic pipeline.")
                return ""
        return ""
