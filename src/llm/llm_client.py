import os
from groq import Groq
from dotenv import load_dotenv


load_dotenv()


class GroqLLMClient:
    def __init__(self, model_name: str | None = None):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in environment variables.")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 800) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()