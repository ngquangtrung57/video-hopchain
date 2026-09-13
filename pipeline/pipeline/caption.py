"""Stage 3, captioning: one caption per shot, written by a vision-language model.

The captioner sees one shot at a time at 1 frame per second. The caption document is kept
only when at least half of its segments carry text.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from .prompts import P1_CAPTION

FPS = 1.0

MAX_PIXELS = 1000000

MAX_TOKENS = 1024

MIN_NON_EMPTY = 0.5


def cut(video_path: str, start: float, end: float, out_path: str) -> str:
    """Cut one shot out of a video with ffmpeg."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
                    "-to", f"{end:.3f}", "-i", video_path, "-an", out_path],
                   check=True)
    return out_path


def caption_segment(chat, video_path: str, segment: dict) -> str:
    """Caption one shot."""
    with tempfile.TemporaryDirectory() as work:
        clip = cut(video_path, float(segment["start"]), float(segment["end"]),
                   os.path.join(work, "shot.mp4"))
        replies = chat.on_video(P1_CAPTION, "Describe this segment.", clip,
                                fps=FPS, max_pixels=MAX_PIXELS, max_tokens=MAX_TOKENS)
    return (replies[0] or "").strip()


def caption_video(chat, video_path: str, skeleton: dict) -> dict:
    """Caption every shot of one video and return the caption document."""
    timeline = []
    for segment in skeleton.get("timeline") or []:
        text = ""
        try:
            text = caption_segment(chat, video_path, segment)
        except (subprocess.CalledProcessError, OSError, KeyError, ValueError):
            text = ""
        timeline.append({"seg": int(segment["seg"]), "start": float(segment["start"]),
                         "end": float(segment["end"]), "text": text})
    return {"video_id": skeleton.get("video_id"), "duration_s": skeleton.get("duration_s"),
            "timeline": timeline}


def keep(document: dict) -> tuple:
    """Keep the caption document only if at least half of its segments carry text."""
    timeline = document.get("timeline") or []
    if not timeline:
        return False, "no segments"
    non_empty = sum(1 for s in timeline if str(s.get("text", "")).strip())
    share = non_empty / len(timeline)
    if share < MIN_NON_EMPTY:
        return False, f"{non_empty}/{len(timeline)} segments captioned"
    return True, f"{non_empty}/{len(timeline)} segments captioned"


__all__ = ["FPS", "MAX_PIXELS", "caption_segment", "caption_video", "cut", "keep"]
