"""The hop record, the question text rendered from it, and the released row format."""

from __future__ import annotations

import re

HOP_FAMILIES = ("order", "spatial_local", "action", "color")

BRANCH_RE = re.compile(r"then\s+(-?\d+)\s+else\s+(-?\d+)")

CONDITION_RE = re.compile(r"^\s*if\s+(.+?)\s+then\s+-?\d+\s+else\s+-?\d+\s*$", re.I | re.S)

# The exact system prompt carried by every released row. Keep it byte for byte,
# including the multiplication sign and the superscripts, or a run no longer
# matches the published dataset.
SYSTEM = (
    "You are a careful reasoning assistant. ALWAYS respond in this EXACT format:\n\n"
    "<think>step-by-step reasoning</think>\n"
    "<answer>\\boxed{final_answer}</answer>\n\n"
    "Examples:\n\n"
    "Q: 7 \u00d7 8?\n"
    "<think>7 \u00d7 8 = 56.</think>\n"
    "<answer>\\boxed{56}</answer>\n\n"
    "Q: A right triangle has legs of length 3 and 4. What is the hypotenuse?\n"
    "<think>By the Pythagorean theorem, c\u00b2 = 3\u00b2 + 4\u00b2 = 9 + 16 = 25, so c = 5.</think>\n"
    "<answer>\\boxed{5}</answer>\n\n"
    "For multiple-choice, put the letter, e.g. \\boxed{B}.\n"
    "Always wrap reasoning in <think>...</think> and answer in "
    "<answer>\\boxed{...}</answer>. No text outside these tags."
)

LETTERS = "ABCDEFGH"


def hops_of(question: dict) -> list:
    """The hop records of a question, in order."""
    return [h for h in (question.get("reasoning_hops") or []) if isinstance(h, dict)]


def branch_pair(hop: dict) -> tuple:
    """The (then, else) numbers of a hop, read from its mapping."""
    m = BRANCH_RE.search(str(hop.get("mapping") or ""))
    if not m:
        raise ValueError("hop mapping carries no 'then A else B'")
    return int(m.group(1)), int(m.group(2))


def condition_of(hop: dict) -> str:
    """The condition a hop tests, read from its mapping."""
    m = CONDITION_RE.match(str(hop.get("mapping") or ""))
    return (m.group(1) if m else str(hop.get("description") or "")).strip().rstrip(".")


def answer_of(question: dict) -> int:
    """The answer: the sum of the branch numbers the video selects."""
    return sum(int(str(h["value"]).strip()) for h in hops_of(question))


def selected_by(question: dict) -> dict:
    """Map a hop number to the earlier hop that decides where it looks."""
    out = {}
    for s in question.get("selectors") or []:
        try:
            out[int(s["selected_hop"])] = int(s["selected_by_hop"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def question_text(question: dict) -> str:
    """The question text of a released row.

    The generator returns the prose in `query` alongside the structured hops, and
    the released dataset carries that prose. We therefore ship it, and we fall
    back on the code renderer when a candidate carries no query, so a row always
    has text. The judge and the difficulty filter deliberately read
    `render_question` instead, because they must see exactly what the hop records
    say rather than what the generator wrote about them.
    """
    query = str(question.get("query") or "").strip()
    return query or render_question(question)


def render_question(question: dict) -> str:
    """Render the question text from the hop record, so no model writes the prose."""
    hops = hops_of(question)
    routed = selected_by(question)
    lines = []
    for i, hop in enumerate(hops, start=1):
        letter = LETTERS[i - 1]
        then, other = branch_pair(hop)
        scene = str(hop.get("scene_ref") or "").strip().rstrip(".")
        condition = condition_of(hop)
        opening = f"In {scene}" if scene else "In the video"
        if i in routed:
            opening = f"Using check {LETTERS[routed[i] - 1]}, {opening[0].lower()}{opening[1:]}"
        lines.append(f"{opening}, if {condition}, let {letter} be {then}; "
                     f"otherwise let {letter} be {other}.")
    total = " + ".join(LETTERS[i] for i in range(len(hops)))
    lines.append(f"Report {total} as a single integer.")
    return "\n".join(lines)


def build_row(question: dict, *, video_path: str, data_source: str,
              frames: int = 140, max_pixels: int = 50176) -> dict:
    """One released training row: the rendered question, the video, and the answer."""
    hops = hops_of(question)
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": "<video>\n" + question_text(question)}],
        "images": [],
        "videos": [{"video": f"file://{video_path}", "nframes": frames,
                    "max_pixels": max_pixels}],
        "data_source": data_source,
        "ability": "video_multihop_reasoning",
        "reward_model": {"ground_truth": str(answer_of(question)), "style": "rule"},
        "extra_info": {
            "reward_type": "numeric",
            "answer": str(answer_of(question)),
            "answer_type": "numeric",
            "num_hops": len(hops),
            "hop_types": [h.get("evidence_type") for h in hops],
            "question_id": str(question.get("id") or ""),
            "video_id": str(question.get("video_id") or ""),
            "dependency": question.get("dependency") or "flat",
            "tolerance": 0.0,
        },
    }


def subsample_timeline(timeline: list, max_segs: int) -> list:
    """Thin a caption timeline down to at most max_segs segments, spread evenly."""
    tl = list(timeline or [])
    n = len(tl)
    if max_segs <= 0 or n <= max_segs:
        return tl
    if max_segs == 1:
        return [tl[0]]
    idx = [round(i * (n - 1) / (max_segs - 1)) for i in range(max_segs)]
    out, seen = [], set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(tl[i])
    return out


__all__ = ["HOP_FAMILIES", "SYSTEM", "answer_of", "branch_pair", "build_row", "condition_of",
           "hops_of", "question_text", "render_question", "selected_by", "subsample_timeline"]
