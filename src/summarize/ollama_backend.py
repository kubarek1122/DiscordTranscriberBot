from __future__ import annotations

from ollama import AsyncClient

from src.prompts import POLISH_SUMMARY_SYSTEM, USER_TEMPLATE


class OllamaSummarizer:
    name = "ollama"

    def __init__(self, *, host: str, model: str) -> None:
        self._client = AsyncClient(host=host)
        self._model = model

    async def summarize(self, transcript: str) -> str:
        resp = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": POLISH_SUMMARY_SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript)},
            ],
            options={"temperature": 0.2},
        )
        return (resp["message"]["content"] or "").strip()
