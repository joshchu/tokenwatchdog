"""Shared ISO-8601 -> epoch-seconds parsing, used by both providers."""

from __future__ import annotations

from datetime import datetime


def parse_iso_to_epoch(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
