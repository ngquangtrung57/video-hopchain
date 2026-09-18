#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

log = logging.getLogger("proxies")

IMAGE_FACTOR = 32
FRAME_FACTOR = 2
MIN_PIXELS = 3136


def _round_by_factor(n: float, f: int) -> int:
    return round(n / f) * f


def _floor_by_factor(n: float, f: int) -> int:
    return math.floor(n / f) * f


def _ceil_by_factor(n: float, f: int) -> int:
    return math.ceil(n / f) * f


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Byte-for-byte port of qwen_vl_utils.vision_process.smart_resize."""
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _decode_and_resize(path: str, frames: int, max_pixels: int):
    import numpy as np
    import torch
    from torchcodec.decoders import VideoDecoder
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    dec = VideoDecoder(path, num_ffmpeg_threads=int(os.environ.get("TORCHCODEC_NUM_THREADS", 4)))
    total = dec.metadata.num_frames

    max_frames = _floor_by_factor(frames, FRAME_FACTOR)
    nframes = min(min(max(total / dec.metadata.average_fps * 1.0, FRAME_FACTOR), max_frames), total)
    nframes = _floor_by_factor(nframes, FRAME_FACTOR)
    if nframes < FRAME_FACTOR:
        raise ValueError(f"nframes={nframes} < {FRAME_FACTOR}")

    idx = torch.linspace(0, total - 1, nframes).round().long().tolist()
    video = dec.get_frames_at(indices=idx).data

    _, _, h, w = video.shape
    th, tw = smart_resize(h, w, IMAGE_FACTOR, MIN_PIXELS, max_pixels)
    video = TF.resize(video, [th, tw], interpolation=InterpolationMode.BICUBIC, antialias=True)
    video = video.round().clamp(0, 255).to(torch.uint8)
    return video.permute(0, 2, 3, 1).contiguous().numpy(), th, tw


def _encode(dst: str, arr, h: int, w: int, crf: int) -> None:
    """Write frames as a 1-fps mp4 at native size. yuv444p avoids chroma subsampling loss."""
    tmp = dst + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "1", "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv444p",
        "-fps_mode", "passthrough", "-an", tmp,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(arr.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode()[:400]}")
    os.replace(tmp, dst)


def _proxy_frame_count(path: str) -> int:
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(path).metadata.num_frames


def build_one(args_tuple) -> dict:
    src, dst, frames, max_pixels, crf, overwrite = args_tuple
    vid = os.path.basename(src)[: -len(".mp4")]
    try:
        if not overwrite and os.path.exists(dst):
            try:
                if _proxy_frame_count(dst) == frames:
                    return {"video_id": vid, "status": "skip"}
            except Exception:
                pass
            os.remove(dst)

        arr, h, w = _decode_and_resize(src, frames, max_pixels)
        if arr.shape[0] != frames:
            return {"video_id": vid, "status": "fail",
                    "error": f"got {arr.shape[0]} frames, expected {frames}"}
        _encode(dst, arr, h, w, crf)

        got = _proxy_frame_count(dst)
        if got != frames:
            os.remove(dst)
            return {"video_id": vid, "status": "fail", "error": f"proxy reads back {got} frames"}
        return {"video_id": vid, "status": "ok", "h": h, "w": w,
                "bytes": os.path.getsize(dst)}
    except Exception as e:
        return {"video_id": vid, "status": "fail", "error": f"{type(e).__name__}: {e}"}


def load_ids(path: str) -> tuple[list[str], int]:
    ids: list[str] = []
    seen: set[str] = set()
    dup = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vid = line.split("\t")[0].strip()
            if vid.endswith(".mp4"):
                vid = vid[: -len(".mp4")]
            if not vid:
                continue
            if vid in seen:
                dup += 1
                continue
            seen.add(vid)
            ids.append(vid)
    return ids, dup


def select_names(src_dir: str, ids_file: str | None) -> tuple[list[str], dict, list[str]]:
    on_disk = {e.name for e in os.scandir(src_dir) if e.name.endswith(".mp4")}
    if not ids_file:
        return sorted(on_disk), {"src_dir_mp4s": len(on_disk), "ids_requested": 0,
                                 "ids_duplicate": 0, "matched": len(on_disk),
                                 "ids_no_source": 0}, []
    ids, dup = load_ids(ids_file)
    names, missing = [], []
    for vid in ids:
        fn = vid + ".mp4"
        if fn in on_disk:
            names.append(fn)
        else:
            missing.append(vid)
    names.sort()
    return names, {"src_dir_mp4s": len(on_disk), "ids_requested": len(ids) + dup,
                   "ids_duplicate": dup, "matched": len(names),
                   "ids_no_source": len(missing)}, missing


def verify(src_dir: str, out_dir: str, frames: int, max_pixels: int, n: int,
           seed: int = 0, ids_file: str | None = None) -> int:
    import random

    import numpy as np
    from qwen_vl_utils import process_vision_info

    os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"
    pool = sorted(f for f in os.listdir(out_dir) if f.endswith(".mp4"))
    if ids_file:
        want = {v + ".mp4" for v in load_ids(ids_file)[0]}
        pool = [f for f in pool if f in want]
    if not pool:
        print("VERIFY FAIL: no proxies found", file=sys.stderr)
        return 1
    proxies = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    print(f"sampling {len(proxies)} of {len(pool)} proxies at random (seed={seed}); "
          f"gate: shape identity AND mean|Δpixel| <= 3.0/255")

    bad = 0
    for f in proxies:
        vid = f[: -len(".mp4")]

        def tensor_for(path: str, mf: int):
            ele = {"type": "video", "video": "file://" + path, "fps": 1, "max_frames": mf,
                   "min_pixels": MIN_PIXELS, "max_pixels": max_pixels}
            conv = [{"role": "user", "content": [ele]}]
            _, vids = process_vision_info(conv, image_patch_size=16, return_video_metadata=True)
            return vids[0][0]

        ref = tensor_for(os.path.join(src_dir, f), frames)
        prx = tensor_for(os.path.join(out_dir, f), frames)

        if tuple(ref.shape) != tuple(prx.shape):
            print(f"  FAIL {vid}: shape {tuple(prx.shape)} != ref {tuple(ref.shape)}")
            bad += 1
            continue
        mad = float(np.abs(ref.float().numpy() - prx.float().numpy()).mean())
        flag = "FAIL" if mad > 3.0 else "ok"
        if flag == "FAIL":
            bad += 1
        print(f"  {flag} {vid}: shape {tuple(prx.shape)}  mean|Δpixel| {mad:.2f}/255")

    print(f"\nverified {len(proxies)} proxies, {bad} bad")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None,
                    help="resolve --src/--out from scripts/data/video_corpora.py (the ONLY "
                         "root table build_trace_inputs.py/build_hopchain_sft.py search). "
                         "Explicit --src/--out still win.")
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", default=None,
                    help="file of video ids to build (plain one-per-line, or TSV with the id "
                         "in column 0 -- data/worklist_v1/fresh_<corpus>.txt works as-is). "
                         "Without it the WHOLE --src dir is built, which crosses the SFT/RL "
                         "split.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the funnel and the missing-id list, transcode nothing")
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-n", type=int, default=20)
    ap.add_argument("--verify-seed", type=int, default=0,
                    help="seed for the RANDOM verify sample (was the alphabetical head, which "
                         "is one shard's early work on a sharded build)")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.corpus:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from video_corpora import CORPORA, PROXY_ROOTS

        if a.corpus not in CORPORA:
            log.error("--corpus %s unknown; known: %s", a.corpus, ", ".join(CORPORA))
            return 2
        a.src = a.src or CORPORA[a.corpus]
        a.out = a.out or PROXY_ROOTS[a.corpus]
        log.info("--corpus %s -> src=%s out=%s", a.corpus, a.src, a.out)
    if not a.src or not a.out:
        log.error("need --src and --out (or --corpus)")
        return 2

    if a.verify:
        return verify(a.src, a.out, a.frames, a.max_pixels, a.verify_n, a.verify_seed, a.ids)

    all_vids, funnel, no_source = select_names(a.src, a.ids)
    log.info("id funnel | src_dir_mp4s=%(src_dir_mp4s)d ids_requested=%(ids_requested)d "
             "ids_duplicate=%(ids_duplicate)d matched=%(matched)d ids_no_source=%(ids_no_source)d",
             funnel)
    shard = all_vids[a.shard :: a.num_shards]
    if not shard:
        log.error("shard %d/%d matched 0 videos of %d -- refusing to run",
                  a.shard, a.num_shards, len(all_vids))
        return 3

    log.info("shard %d/%d: %d of %d videos | frames=%d max_pixels=%d crf=%d workers=%d",
             a.shard, a.num_shards, len(shard), len(all_vids), a.frames, a.max_pixels,
             a.crf, a.workers)

    if a.dry_run:
        log.info("[dry-run] would build %d videos into %s; nothing written", len(shard), a.out)
        if no_source:
            log.info("[dry-run] %d ids have no source file; first few: %s",
                     len(no_source), no_source[:5])
        return 0

    os.makedirs(a.out, exist_ok=True)
    if no_source:
        p = os.path.join(a.out, f"IDS_NO_SOURCE_shard{a.shard}.txt")
        with open(p, "w") as fh:
            fh.write("\n".join(no_source) + "\n")
        log.warning("%d of %d requested ids have no file under %s -- listed in %s",
                    len(no_source), funnel["ids_requested"], a.src, p)

    tasks = [(os.path.join(a.src, f), os.path.join(a.out, f), a.frames, a.max_pixels,
              a.crf, a.overwrite) for f in shard]

    ok = skip = fail = 0
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["status"] == "ok":
                ok += 1
            elif r["status"] == "skip":
                skip += 1
            else:
                fail += 1
                failures.append(r)
                log.warning("FAIL %s: %s", r["video_id"], r.get("error"))
            if i % 50 == 0:
                log.info("shard %d: %d/%d (ok=%d skip=%d fail=%d)",
                         a.shard, i, len(tasks), ok, skip, fail)

    log.info("shard %d DONE: ok=%d skip=%d fail=%d (of %d tasks; ids_no_source=%d)",
             a.shard, ok, skip, fail, len(tasks), len(no_source))
    assert ok + skip + fail == len(tasks), "funnel does not close -- a video was dropped silently"
    if failures:
        dst = os.path.join(a.out, f"FAILED_shard{a.shard}.json")
        tmp = dst + ".partial"
        with open(tmp, "w") as fh:
            json.dump(failures, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    else:
        open(os.path.join(a.out, f"DONE_shard{a.shard}"), "w").close()
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
