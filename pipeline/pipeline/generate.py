"""Stage 5, generation: a text-only model fills every drawn specification from the captions.

The model never sees the video. It is handed the caption timeline and the specifications, and
it returns one question per specification, or nothing for a specification the captions cannot
ground.
"""

from __future__ import annotations

import json

from .prompts import P_HOPGEN
from .schema import subsample_timeline
from .spec import sample_specs

MAX_TIMELINE_SEGMENTS = 200

TOKENS_PER_CANDIDATE = 3072

MIN_TOKENS = 24576


def max_tokens_for(candidates: int) -> int:
    """Room for the reasoning plus one question per specification."""
    return max(MIN_TOKENS, MIN_TOKENS + TOKENS_PER_CANDIDATE * max(0, candidates - 4))


def render_timeline(timeline: list) -> str:
    """The caption timeline as the generator reads it, one line per segment."""
    lines = []
    for s in timeline:
        lines.append(f"[seg {int(s['seg'])} | {float(s.get('start', 0.0)):.1f}-"
                     f"{float(s.get('end', 0.0)):.1f} s] {str(s.get('text', '')).strip()}")
    return "\n".join(lines)


def build_payload(caption: dict, specs: list) -> str:
    """The generator user message: the caption timeline and the specifications to fill."""
    timeline = subsample_timeline(caption.get("timeline") or [], MAX_TIMELINE_SEGMENTS)
    return json.dumps({
        "num_queries": len(specs),
        "query_specs": specs,
        "target": ("build ONE query per spec in query_specs, filling each hop with a "
                   "caption-grounded robust observation; refuse (omit) any spec the caption "
                   "cannot support"),
        "n_segments": len(caption.get("timeline") or []),
        "windows": render_timeline(timeline),
    }, ensure_ascii=False)


def attach_specs(candidates: list, specs: list) -> list:
    """Give every returned candidate the specification it was drawn from."""
    used, out = set(), []
    for position, candidate in enumerate(candidates):
        spec = None
        try:
            wanted = int(candidate.get("id"))
            spec = next((s for s in specs if s["id"] == wanted and wanted not in used), None)
        except (TypeError, ValueError):
            spec = None
        if spec is None and position < len(specs) and specs[position]["id"] not in used:
            spec = specs[position]
        if spec is None:
            continue
        used.add(spec["id"])
        candidate["selectors"] = spec.get("selectors") or []
        candidate["selector"] = spec.get("selector")
        candidate["dependency"] = spec.get("dependency", "")
        candidate["spec_id"] = spec["id"]
        out.append(candidate)
    return out


def generate(chat, caption: dict, *, candidates: int = 2) -> list:
    """Generate the question candidates of one caption document."""
    video_id = str(caption.get("video_id") or "video")
    specs = sample_specs(video_id, candidates)
    result = chat.json_text(P_HOPGEN, build_payload(caption, specs),
                            max_tokens=max_tokens_for(candidates))
    if isinstance(result, list):
        result = {"sub_queries": result}
    if not isinstance(result, dict):
        return []
    returned = [c for c in (result.get("sub_queries") or []) if isinstance(c, dict)]
    out = attach_specs(returned[:candidates], specs)
    for candidate in out:
        candidate["video_id"] = video_id
        candidate["id"] = f"{video_id}_{candidate.get('spec_id')}"
    return out


def regenerate(chat, caption: dict, candidate: dict, *, candidates: int = 2):
    """Write one more question for the specification of a candidate the judge faulted."""
    video_id = str(caption.get("video_id") or "video")
    specs = sample_specs(video_id, candidates)
    spec = next((s for s in specs if s["id"] == candidate.get("spec_id")), None)
    if spec is None:
        return None
    result = chat.json_text(P_HOPGEN, build_payload(caption, [spec]),
                            max_tokens=max_tokens_for(1))
    if isinstance(result, list):
        result = {"sub_queries": result}
    if not isinstance(result, dict):
        return None
    returned = [c for c in (result.get("sub_queries") or []) if isinstance(c, dict)]
    out = attach_specs(returned[:1], [spec])
    if not out:
        return None
    question = out[0]
    question["video_id"] = video_id
    question["id"] = f"{video_id}_{question.get('spec_id')}"
    return question


__all__ = ["attach_specs", "build_payload", "generate", "max_tokens_for",
           "regenerate", "render_timeline"]
