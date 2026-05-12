from __future__ import annotations

from anthropic import AsyncAnthropic

from src.prompts import POLISH_SUMMARY_SYSTEM, USER_TEMPLATE


class ClaudeSummarizer:
    name = "claude"

    def __init__(self, *, api_key: str, model: str, max_tokens: int) -> None:
        # The SDK honours `Retry-After` on 429/529/5xx, which is what the
        # API actually wants us to do during overloads. Bump from the
        # default of 2 to give the outer tenacity loop slack on sustained
        # overloads (the two work multiplicatively).
        self._client = AsyncAnthropic(api_key=api_key, max_retries=4)
        self._model = model
        self._max_tokens = max_tokens

    async def summarize(self, transcript: str) -> str:
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": POLISH_SUMMARY_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(transcript=transcript),
                }
            ],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()
