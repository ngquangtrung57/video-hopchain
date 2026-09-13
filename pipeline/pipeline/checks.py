"""Stage 6, the code checks: drop a candidate the code can prove is malformed.

A candidate is dropped when a mapping does not parse, a stored value is not one of the two
branch numbers, the 2^n totals collide, or every hop takes the same branch. The answer is
recomputed from the hop values whenever it disagrees with them.
"""

from __future__ import annotations

from .schema import branch_pair, hops_of
from .spec import branch_totals_unique


def check(candidate: dict) -> tuple:
    """Return (kept, reason). A kept candidate has its answer recomputed in place."""
    hops = hops_of(candidate)
    if not hops:
        return False, "no hops"

    branches, values = [], []
    for hop in hops:
        try:
            branches.append(list(branch_pair(hop)))
        except ValueError:
            return False, "mapping does not parse"
        text = str(hop.get("value", "")).strip()
        if not text.lstrip("-").isdigit():
            return False, "hop value is not an integer"
        values.append(int(text))

    for value, pair in zip(values, branches):
        if value not in pair:
            return False, "hop value is neither branch number"

    if not branch_totals_unique(branches):
        return False, "branch totals collide"

    directions = {value == max(pair) for value, pair in zip(values, branches)}
    if len(directions) == 1:
        return False, "every hop takes the same branch"

    candidate["hypothetical_answer"] = str(sum(values))
    return True, "ok"


def run(candidates: list) -> tuple:
    """Apply the checks to a list of candidates. Returns (kept, dropped_reasons)."""
    kept, dropped = [], []
    for candidate in candidates:
        ok, reason = check(candidate)
        if ok:
            kept.append(candidate)
        else:
            dropped.append((candidate.get("id"), reason))
    return kept, dropped


__all__ = ["check", "run"]
