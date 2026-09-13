"""Stage 4, the chain specification: the structure of every question, drawn by code.

The draw fixes the hop count, the family of every hop, the two numbers of every hop and the
way the hops link, before any model writes text. It is seeded by the video id, so the same
video always yields the same specification.
"""

from __future__ import annotations

import hashlib
import random

BRANCH_MAX = 80

BRANCH_TRIES = 20000

HOP_WEIGHTS = {4: 0.55, 5: 0.35, 6: 0.10}

DEPENDENCY_WEIGHTS = {"flat": 0.30, "selector": 0.40, "selector2": 0.30}

TEMPORAL_FAMILIES = ("order",)

DEEP_FAMILIES = ("order", "spatial_local", "action")

FILLER_FAMILIES = ("color",)

MAX_FILLER = {4: 1, 5: 1, 6: 2}


def sum_expression(n: int) -> str:
    """The arithmetic of a chain: a plain sum over the hop numbers."""
    return "+".join(f"H{i}" for i in range(1, n + 1))


def branch_totals_unique(branches: list) -> bool:
    """True when the 2^n branch sums of a spec are all distinct."""
    n = len(branches)
    seen = set()
    for combo in range(1 << n):
        total = sum(branches[i][(combo >> i) & 1] for i in range(n))
        if total in seen:
            return False
        seen.add(total)
    return True


def draw_branches(rng, n: int) -> list:
    """[then, else] pairs from 1..BRANCH_MAX whose 2^n sums are all distinct."""
    for _ in range(BRANCH_TRIES):
        pairs = [list(rng.sample(range(1, BRANCH_MAX + 1), 2)) for _ in range(n)]
        if branch_totals_unique(pairs):
            return pairs
    raise RuntimeError(f"no branch assignment gives 2^{n} distinct sums within {BRANCH_TRIES} "
                       f"draws over 1..{BRANCH_MAX}")


def draw_families(rng, n: int) -> list:
    """One family per hop, with at least one order hop and a capped number of colour hops."""
    families = [rng.choice(TEMPORAL_FAMILIES)]
    n_filler = rng.randint(0, MAX_FILLER.get(n, 2))
    families += [rng.choice(FILLER_FAMILIES) for _ in range(min(n_filler, n - 1))]
    while len(families) < n:
        families.append(rng.choice(DEEP_FAMILIES))
    rng.shuffle(families)
    return families


def _seed(tag: str) -> int:
    return int.from_bytes(hashlib.sha1(tag.encode("utf-8")).digest()[:8], "big")


def draw_selectors(rng, dependency: str, n: int) -> list:
    """Which hop an earlier hop routes, for a selector question."""
    if dependency == "selector":
        chosen = rng.randint(2, n)
        return [{"selected_hop": chosen, "selected_by_hop": rng.randint(1, chosen - 1)}]
    if dependency == "selector2":
        second = rng.randint(3, n)
        first = rng.randint(2, second - 1)
        return [{"selected_hop": first, "selected_by_hop": rng.randint(1, first - 1)},
                {"selected_hop": second, "selected_by_hop": first}]
    return []


def sample_specs(video_id: str, num_queries: int) -> list:
    """Draw the structure of every question of one video."""
    rng = random.Random(_seed(f"spec:{video_id}"))
    modes, mode_weights = list(DEPENDENCY_WEIGHTS), list(DEPENDENCY_WEIGHTS.values())
    hop_counts, hop_weights = list(HOP_WEIGHTS), list(HOP_WEIGHTS.values())
    specs = []
    for i in range(max(1, int(num_queries))):
        n = rng.choices(hop_counts, weights=hop_weights)[0]
        families = draw_families(rng, n)
        branch_rng = random.Random(_seed(f"branches:{video_id}:{i}"))
        branches = draw_branches(branch_rng, n)
        dependency = rng.choices(modes, weights=mode_weights)[0]
        selectors = draw_selectors(rng, dependency, n)
        specs.append({
            "id": i + 1,
            "hop_count": n,
            "arithmetic": sum_expression(n),
            "predicate_families": families,
            "branch_numbers": branches,
            "dependency": dependency,
            "selectors": selectors,
            "selector": selectors[0] if selectors else None,
        })
    return specs


__all__ = ["BRANCH_MAX", "DEPENDENCY_WEIGHTS", "HOP_WEIGHTS", "branch_totals_unique",
           "draw_branches", "draw_families", "draw_selectors", "sample_specs", "sum_expression"]
