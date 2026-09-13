"""Stage 9, the difficulty filter: drop a question the base model already solves.

The base model answers the question from the video proxy eight times. We drop a question that
it already solves in six of the eight rollouts, so the dataset keeps the questions that stay
difficult for the policy.
"""

from __future__ import annotations

import re

from .prompts import P_SOLVER
from .schema import answer_of, render_question

ROLLOUTS = 8

TOO_EASY_RATE = 0.75

FRAMES = 140

MAX_PIXELS = 50176

_INTEGER = re.compile(r"-?\d+")


def read_answer(text: str):
    """The integer a solver reported, or None."""
    try:
        from .model import extract_json
        value = extract_json(text)
        if isinstance(value, dict) and "answer" in value:
            found = _INTEGER.search(str(value["answer"]))
            return int(found.group()) if found else None
    except ValueError:
        pass
    found = _INTEGER.findall(text or "")
    return int(found[-1]) if found else None


def solve_rate(chat, question: dict, video_path: str, *, rollouts: int = ROLLOUTS) -> float:
    """The share of rollouts in which the base model reports the gold answer."""
    gold = answer_of(question)
    replies = chat.on_video(P_SOLVER, render_question(question), video_path,
                            fps=1.0, max_pixels=MAX_PIXELS, temperature=1.0, n=rollouts)
    correct = sum(1 for reply in replies if read_answer(reply) == gold)
    return correct / max(1, len(replies))


def too_easy(rate: float) -> bool:
    """True when the base model solves the question often enough to drop it."""
    return rate >= TOO_EASY_RATE


__all__ = ["FRAMES", "MAX_PIXELS", "ROLLOUTS", "TOO_EASY_RATE", "read_answer", "solve_rate",
           "too_easy"]
