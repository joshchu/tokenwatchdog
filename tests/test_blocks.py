"""Fixed-block anchoring — the rule that makes a 5-hour reset computable.

The point of these is that `resets_at` no longer has to be caught in the
act. Before, an anchor existed only if a percentage drop happened to land
between two polls; now it follows from the activity log itself.
"""

from __future__ import annotations

from tokenwatchdog.blocks import block_anchor

HOUR = 3600.0
FIVE_HOURS = 5 * HOUR


def test_no_activity_has_no_anchor():
    assert block_anchor([], FIVE_HOURS, 1000.0) is None


def test_single_recent_request_anchors_the_block_at_itself():
    now = 100_000.0
    assert block_anchor([now - HOUR], FIVE_HOURS, now) == now - HOUR


def test_continuous_activity_keeps_the_original_anchor():
    """Requests every half hour inside one block must not re-anchor: the
    block expires on its own schedule, not on the latest request."""
    now = 100_000.0
    start = now - 4 * HOUR
    timestamps = [start + i * 1800.0 for i in range(8)]
    assert block_anchor(timestamps, FIVE_HOURS, now) == start


def test_a_gap_of_a_full_duration_starts_a_new_block():
    now = 100_000.0
    old = now - 9 * HOUR
    fresh = now - 2 * HOUR  # 7h after `old` -- more than a whole block later
    assert block_anchor([old, fresh], FIVE_HOURS, now) == fresh


def test_a_block_reaching_its_duration_re_anchors_even_without_a_gap():
    """Back-to-back hourly requests for six hours are two blocks, not one:
    the first expires five hours in regardless of activity."""
    now = 100_000.0
    start = now - 6 * HOUR
    timestamps = [start + i * HOUR for i in range(7)]
    anchor = block_anchor(timestamps, FIVE_HOURS, now)
    assert anchor == start + 5 * HOUR
    assert now - anchor < FIVE_HOURS


def test_an_expired_block_reports_no_anchor_rather_than_a_stale_one():
    """Nothing for six hours means the window is empty and the next block
    won't exist until the next request. Returning the expired anchor would
    put `resets_at` in the past."""
    now = 100_000.0
    assert block_anchor([now - 6 * HOUR], FIVE_HOURS, now) is None
