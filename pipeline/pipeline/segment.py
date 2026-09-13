"""Stage 2, segmentation: split the video into shots with PySceneDetect.

A shot shorter than the minimum folds into its neighbour, and a shot longer than the
maximum splits, so every segment lasts between 2 and 10 seconds.
"""

from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_THRESHOLD = 27.0
DEFAULT_MAX_LEN = 10.0
DEFAULT_MIN_LEN = 2.0


def get_duration(video_path: str) -> float:
    """Duration of a video file in seconds."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    finally:
        cap.release()


def segment_hybrid(video_path: str, threshold: float = DEFAULT_THRESHOLD,
                   max_len: float = DEFAULT_MAX_LEN,
                   min_len: float = DEFAULT_MIN_LEN) -> list[tuple[float, float]]:
    """Scene-cut segmentation, then split long scenes and merge short ones. Returns [(start,end)]."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()

    dur = get_duration(video_path)
    raw = ([(0.0, dur)] if not scenes
           else [(round(s[0].seconds, 2), round(s[1].seconds, 2)) for s in scenes])

    split: list[tuple[float, float]] = []
    for start, end in raw:
        length = end - start
        if length > max_len:
            n = int(length / max_len) + 1
            sub = length / n
            for i in range(n):
                s = round(start + i * sub, 2)
                e = round(start + (i + 1) * sub, 2)
                split.append((s, min(e, end)))
        else:
            split.append((start, end))

    merged: list[tuple[float, float]] = []
    i = 0
    while i < len(split):
        start, end = split[i]
        while (end - start) < min_len and i + 1 < len(split):
            i += 1
            _, end = split[i]
        merged.append((round(start, 2), round(end, 2)))
        i += 1
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_len:
        last = merged.pop()
        prev = merged.pop()
        merged.append((prev[0], last[1]))
    return merged


def video_id_of(video_path: str) -> str:
    """The video id is the file name without its extension."""
    return os.path.splitext(os.path.basename(video_path))[0]


def build_caption_skeleton(video_path: str, video_id: Optional[str] = None,
                           **seg_kwargs) -> dict:
    """The empty caption document of one video: its id, its duration and its segment timeline."""
    segs = segment_hybrid(video_path, **seg_kwargs)
    dur = get_duration(video_path)
    return {
        "video_id": video_id or video_id_of(video_path),
        "duration_s": round(dur, 2),
        "timeline": [{"seg": i, "start": s, "end": e, "text": ""}
                     for i, (s, e) in enumerate(segs)],
    }


def validate_caption_doc(doc: dict) -> list[str]:
    """Structural check of a filled caption document; returns the list of problems found."""
    errs: list[str] = []
    for k in ("video_id", "duration_s", "timeline"):
        if k not in doc:
            errs.append(f"missing top-level key: {k}")
    for i, seg in enumerate(doc.get("timeline", [])):
        for k in ("seg", "start", "end", "text"):
            if k not in seg:
                errs.append(f"timeline[{i}] missing key: {k}")
        if isinstance(seg.get("start"), (int, float)) and isinstance(seg.get("end"), (int, float)):
            if seg["end"] < seg["start"]:
                errs.append(f"timeline[{i}] end<start ({seg['end']}<{seg['start']})")
    return errs


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="print the caption skeleton for one video")
    ap.add_argument("video")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the one-line funnel to stderr and NOTHING to stdout")
    a = ap.parse_args()

    doc = build_caption_skeleton(a.video)
    print(f"# {doc['video_id']}: {len(doc['timeline'])} segments, {doc['duration_s']}s",
          file=sys.stderr)
    if not a.dry_run:
        print(json.dumps(doc, indent=2))
