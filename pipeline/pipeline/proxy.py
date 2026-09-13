"""Stage 1, admit and encode: keep a long enough video and write the proxy the policy sees.

The proxy holds 140 frames sampled evenly over the whole video, each resized to at most
50,176 pixels with sides that are multiples of 32.
"""

from __future__ import annotations

import os

import cv2

MIN_DURATION = 180.0

FRAMES = 140

MAX_PIXELS = 50176

MIN_PIXELS = 3136

SIDE_MULTIPLE = 32


def duration_of(video_path: str) -> float:
    """The duration of a video in seconds."""
    capture = cv2.VideoCapture(video_path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return float(count / fps) if fps > 0 else 0.0
    finally:
        capture.release()


def admit(video_path: str) -> tuple:
    """Reject a video too short to hold moments that lie far apart."""
    seconds = duration_of(video_path)
    if seconds < MIN_DURATION:
        return False, f"{seconds:.1f} s < {MIN_DURATION:.0f} s"
    return True, f"{seconds:.1f} s"


def target_size(width: int, height: int) -> tuple:
    """The proxy size: at most MAX_PIXELS, sides a multiple of 32, aspect ratio kept."""
    scale = min(1.0, (MAX_PIXELS / float(max(1, width * height))) ** 0.5)
    out_w = max(SIDE_MULTIPLE, int(width * scale) // SIDE_MULTIPLE * SIDE_MULTIPLE)
    out_h = max(SIDE_MULTIPLE, int(height * scale) // SIDE_MULTIPLE * SIDE_MULTIPLE)
    while out_w * out_h > MAX_PIXELS and (out_w > SIDE_MULTIPLE or out_h > SIDE_MULTIPLE):
        if out_w >= out_h:
            out_w = max(SIDE_MULTIPLE, out_w - SIDE_MULTIPLE)
        else:
            out_h = max(SIDE_MULTIPLE, out_h - SIDE_MULTIPLE)
    return out_w, out_h


def frame_positions(total: int, frames: int = FRAMES) -> list:
    """Frame indices spread evenly over the source, as linspace(0, total - 1, frames)."""
    if total <= 0:
        return []
    if frames == 1:
        return [0]
    return [round(i * (total - 1) / (frames - 1)) for i in range(frames)]


def write_proxy(video_path: str, out_path: str, *, frames: int = FRAMES) -> str:
    """Write the proxy of one video and return its path."""
    capture = cv2.VideoCapture(video_path)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"cannot read {video_path}")
        out_w, out_h = target_size(width, height)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (out_w, out_h))
        try:
            for index in frame_positions(total, frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    continue
                writer.write(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA))
        finally:
            writer.release()
    finally:
        capture.release()
    return out_path


__all__ = ["FRAMES", "MAX_PIXELS", "MIN_DURATION", "admit", "duration_of", "frame_positions",
           "target_size", "write_proxy"]
