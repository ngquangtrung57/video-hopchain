"""Stage 7, the judge: a text-only model re-examines every hop against the same captions.

The judge assigns each hop one fault or none. A question passes only when no hop is faulted.
"""

from __future__ import annotations

import json

from .prompts import P_JUDGE
from .schema import hops_of, render_question, subsample_timeline

FAULTS = ("VALUE_WRONG", "NOT_SETTLED", "AMBIGUOUS_ANSWER", "ROUTE_BROKEN", "CIRCULAR")

MAX_TIMELINE_SEGMENTS = 200

MAX_TOKENS = 16384


def build_payload(candidate: dict, caption: dict) -> str:
    """The judge user message: the caption windows, the question and its hop record."""
    timeline = subsample_timeline(caption.get("timeline") or [], MAX_TIMELINE_SEGMENTS)
    windows = "\n".join(
        f"[seg {int(s['seg'])} | {float(s.get('start', 0.0)):.1f}-"
        f"{float(s.get('end', 0.0)):.1f} s] {str(s.get('text', '')).strip()}"
        for s in timeline)
    return json.dumps({
        "n_segments": len(caption.get("timeline") or []),
        "windows": windows,
        "question": render_question(candidate),
        "hops": [{"hop_no": int(h.get("hop_no", i + 1)),
                  "scene_ref": str(h.get("scene_ref") or ""),
                  "description": str(h.get("description") or ""),
                  "mapping": str(h.get("mapping") or ""),
                  "value": str(h.get("value") or "")}
                 for i, h in enumerate(hops_of(candidate))],
        "selectors": candidate.get("selectors") or [],
        "dependency": candidate.get("dependency") or "flat",
    }, ensure_ascii=False)


def read_verdict(result) -> dict:
    """Map the judge output to {hop_no: fault}, keeping only the faults this stage knows."""
    faults = {}
    hops = (result or {}).get("hops") if isinstance(result, dict) else None
    for entry in hops or []:
        if not isinstance(entry, dict):
            continue
        fault = str(entry.get("fault") or "").strip().upper()
        if fault and fault not in ("NONE", "OK", "NULL"):
            try:
                faults[int(entry["hop_no"])] = fault
            except (KeyError, TypeError, ValueError):
                continue
    return faults


def judge(chat, candidate: dict, caption: dict) -> dict:
    """Judge one candidate. Returns {hop_no: fault}; empty means the question passes."""
    result = chat.json_text(P_JUDGE, build_payload(candidate, caption), max_tokens=MAX_TOKENS)
    return read_verdict(result)


__all__ = ["FAULTS", "build_payload", "judge", "read_verdict"]
