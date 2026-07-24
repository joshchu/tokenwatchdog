"""Provider protocol — the seam that makes both Codex and Claude first-class
citizens instead of one being a special case of the other.

Justified by having two real implementations (Codex + Claude) — exactly the
bar for introducing an interface.
"""

from __future__ import annotations

from typing import Protocol

from tokenwatchdog.config import Config
from tokenwatchdog.models import Window
from tokenwatchdog.store import Store


class QuotaProvider(Protocol):
    name: str

    def read(self, cfg: Config, store: Store) -> list[Window]:
        """Return whatever Window(s) this provider can see right now.

        `store` is passed to every provider uniformly even though only
        Claude's token-compute path uses it — it needs durable history
        (token_events) to avoid re-scanning hundreds of MB of transcripts
        every poll. A lopsided Protocol (some providers taking a store,
        some not) would be worse than one unused parameter on Codex.

        Return fewer than the full watched set if a window kind has no data
        (e.g. Codex's 5h window on a build that isn't emitting it) — never
        fabricate a reading.
        """
        ...
