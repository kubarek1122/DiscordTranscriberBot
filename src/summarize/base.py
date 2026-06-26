from __future__ import annotations

from typing import Protocol

from src.prompts import USER_TEMPLATE


class Summarizer(Protocol):
    name: str

    async def summarize(
        self,
        transcript: str,
        *,
        system_prompt: str,
        user_template: str = USER_TEMPLATE,
    ) -> str:
        """Run one completion: `system_prompt` as the system message, the
        transcript wrapped in `user_template` as the user message. Returns the
        model's text reply (Polish Markdown for a summary, or a single kind
        token when called with the classification prompt)."""
        ...
