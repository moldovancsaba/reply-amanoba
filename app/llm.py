import requests


class OllamaLLM:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def answer_with_context(self, question: str, context_blocks: list[str], model: str | None = None) -> str:
        context = "\n\n".join(context_blocks)

        prompt = f"""
You are an internal enterprise help assistant.
Rules:
1) Use ONLY facts explicitly present in CONTEXT.
2) If context is missing or ambiguous, respond exactly: INSUFFICIENT_CONTEXT
3) Do not invent policy, numbers, dates, names, or links.
4) Keep response concise and include citations like [doc:source#chunk].

QUESTION:
{question}

CONTEXT:
{context}
""".strip()

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model or self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
