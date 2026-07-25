"""Fixed-duration usage blocks, anchored deterministically from activity.

Claude's 5-hour window is not a rolling roll-off: it is a fixed block that
anchors at your first message and expires exactly `duration` later, all at
once. That makes the reset time *computable* from the token log alone —
no waiting to catch a percentage drop in the act, which is the only way
`predictor._derive_resets_at` can otherwise learn an anchor.

The rule, following ccusage's block algorithm: a new block starts at the
first activity that is either `duration` or more past the current block's
start, or `duration` or more after the previous activity. Shared by
`providers/claude.py` (which sums a block's tokens) and `predictor.py`
(which turns the anchor into a reset time), so the two can never disagree
about where a block begins.

One deliberate deviation: ccusage floors each anchor to the top of the hour.
This uses the exact first-activity timestamp instead, which matches the reset
times actually observed on this account. If real blocks turn out to be
hour-aligned, this runs up to 59 minutes early and flooring is the fix — a
question for observed data, not for a guess either way.
"""

from __future__ import annotations


def block_anchor(
    timestamps: list[float], duration_seconds: float, now: float
) -> float | None:
    """Start of the block that is still open at `now`, or None if none is.

    None means "no active block": either there was no activity at all, or
    the last block already expired, in which case the window is empty and
    the next block won't exist until the next request. That is a fact worth
    reporting as unknown rather than papering over with the expired anchor.
    """
    if not timestamps or duration_seconds <= 0:
        return None
    anchor = timestamps[0]
    previous = timestamps[0]
    for ts in timestamps[1:]:
        if ts - anchor >= duration_seconds or ts - previous >= duration_seconds:
            anchor = ts
        previous = ts
    if now - anchor >= duration_seconds:
        return None
    return anchor
