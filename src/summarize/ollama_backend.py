from __future__ import annotations

from ollama import AsyncClient

from src.prompts import USER_TEMPLATE


class OllamaSummarizer:
    name = "ollama"

    def __init__(self, *, host: str, model: str) -> None:
        self._client = AsyncClient(host=host)
        self._model = model

    async def summarize(
        self,
        transcript: str,
        *,
        system_prompt: str,
        user_template: str = USER_TEMPLATE,
    ) -> str:
        resp = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(transcript=transcript)},
            ],
            options={"temperature": 0.2},
        )
        return (resp["message"]["content"] or "").strip()
