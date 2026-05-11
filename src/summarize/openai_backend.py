from __future__ import annotations

from openai import AsyncOpenAI

from src.prompts import POLISH_SUMMARY_SYSTEM, USER_TEMPLATE


class OpenAISummarizer:
    name = "openai"

    def __init__(self, *, api_key: str, model: str, max_tokens: int) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def summarize(self, transcript: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": POLISH_SUMMARY_SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript)},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
