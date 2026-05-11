from __future__ import annotations

from typing import Protocol


class Summarizer(Protocol):
    name: str

    async def summarize(self, transcript: str) -> str:
        """Return a Markdown summary in Polish, with the agreed sections."""
        ...
