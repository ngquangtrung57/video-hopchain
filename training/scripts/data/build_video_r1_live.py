#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import os
import random
import re
import sys

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

CURATED = "__SCRATCH__/rl/data/curated_v1/train_video"
SOURCES = ("llava", "star", "nextqa", "clevrer", "perceptiontest")
PLAIN_BASE_RUNS = (
    "video_mix_8b_base",
    "video_mix_8b_s04",
    "video_mix_22tv4_s04",
)
WS = re.compile(r"\s+")

VIDEOS_TYPE = pa.list_(pa.struct([
    ("fps", pa.int64()), ("max_frames", pa.int64()), ("max_pixels", pa.int64()),
    ("min_pixels", pa.int64()), ("type", pa.string()), ("video", pa.string()),
]))
SCHEMA = pa.schema([
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("images", pa.list_(pa.null())),
    ("videos", VIDEOS_TYPE),
    ("data_source", pa.large_string()),
    ("ability", pa.large_string()),
    ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
    ("extra_info", pa.struct([
        ("reward_type", pa.string()), ("answer", pa.string()), ("answer_type", pa.string()),
        ("num_hops", pa.int64()), ("hop_types", pa.list_(pa.string())),
        ("question_id", pa.string()), ("video_id", pa.string()),
        ("solve_rate", pa.float64()), ("difficulty_band", pa.string()),
    ])),
])


def qkey(text: str) -> str:
    return hashlib.sha1(WS.sub(" ", text).strip().encode()).hexdigest()[:16]


def band(sr: float) -> str:
    """Difficulty label. Cut points chosen so `hard` is where the model mostly fails but not always."""
    if sr <= 0.25:
        return "hard"
    if sr <= 0.75:
        return "medium"
    return "easy"


def load_corpus():
    """question key -> the full parquet row, keeping only keys that are unique corpus-wide."""
    by_key = collections.defaultdict(list)
    for name in SOURCES:
        table = pq.read_table(f"{CURATED}/video_r1_{name}_mc_v1.parquet")
        for row in table.to_pylist():
            user = [m for m in row["prompt"] if m["role"] == "user"][0]["content"]
            body = user.split("<video>\n", 1)[1] if "<video>\n" in user else user
            by_key[qkey(body)].append(row)
    dropped = sum(len(v) for v in by_key.values() if len(v) > 1)
    return {k: v[0] for k, v in by_key.items() if len(v) == 1}, dropped


def load_stats(stats_dir, keys, runs=None):
    pooled = collections.defaultdict(lambda: [0, 0])
    used = 0
    for path in sorted(glob.glob(f"{stats_dir}/per_run/*.json")):
        run = os.path.basename(path)[:-5]
        if runs is not None and run not in runs:
            continue
        used += 1
        for k, v in orjson.loads(open(path, "rb").read()).items():
            if k in keys:
                pooled[k][0] += v[0]
                pooled[k][1] += v[1]
    return pooled, used


VR1_ROOT = "video-r1-extracted/"


def video_id_of(path: str) -> str:
    p = path[len("file://"):] if path.startswith("file://") else path
    i = p.find(VR1_ROOT)
    if i >= 0:
        p = p[i + len(VR1_ROOT):]
    return os.path.splitext(p)[0]


def to_verl(row: dict, key: str, sr: float, frames: int, min_px: int, max_px: int) -> dict:
    src = row["data_source"]
    video = row["videos"][0]["video"]
    return {
        "prompt": [{"role": m["role"], "content": m["content"]} for m in row["prompt"]],
        "images": [],
        "videos": [{"fps": 1, "max_frames": frames, "max_pixels": max_px,
                    "min_pixels": min_px, "type": "video", "video": video}],
        "data_source": src,
        "ability": row["ability"],
        "reward_model": {"ground_truth": str(row["reward_model"]["ground_truth"]),
                         "style": "rule"},
        "extra_info": {
            "reward_type": row["extra_info"]["reward_type"],
            "answer": str(row["extra_info"]["answer"]),
            "answer_type": "multiple_choice",
            "num_hops": 0, "hop_types": [],
            "question_id": f"{src}__{key}",
            "video_id": video_id_of(video),
            "solve_rate": float(sr), "difficulty_band": band(sr),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir",
                    default="__SCRATCH__/rl/data/rollout_stats")
    ap.add_argument("--target", type=int, default=30000, help="TRAIN rows")
    ap.add_argument("--val-rows", type=int, default=500)
    ap.add_argument("--min-rollouts", type=int, default=16)
    ap.add_argument("--max-solve-rate", type=float, default=1.0,
                    help="upper bound; 1.0 means 'take the hardest --target of the live band'")
    ap.add_argument("--max-q-per-video", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="video_r1_live")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--max-pixels", type=int, default=100000)
    ap.add_argument("--min-pixels", type=int, default=3136)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    corpus, collided = load_corpus()
    print(f"corpus: {len(corpus)} uniquely-keyed questions ({collided} rows dropped for a "
          f"colliding question text)")

    pooled, nruns = load_stats(a.stats_dir, corpus)
    plain, nplain = load_stats(a.stats_dir, corpus, set(PLAIN_BASE_RUNS))
    print(f"rollout stats: {sum(v[0] for v in pooled.values()):,} rollouts over {nruns} runs "
          f"({sum(v[0] for v in plain.values()):,} over {nplain} plain base-init runs)")

    stats = collections.Counter()
    live = {}
    for k, (n, c) in pooled.items():
        if n < a.min_rollouts:
            stats["drop_too_few_rollouts"] += 1
            continue
        if c == 0:
            stats["drop_never_solved"] += 1
            continue
        if c == n:
            stats["drop_always_solved"] += 1
            continue
        sr = c / n
        if sr > a.max_solve_rate:
            stats["drop_above_max_solve_rate"] += 1
            continue
        live[k] = sr
    stats["live"] = len(live)
    for k in sorted(stats):
        print(f"  {k:28s} {stats[k]:7d}")

    want = a.target + a.val_rows
    if len(live) < want:
        print(f"ERROR: only {len(live)} live questions, need {want}. Lower --target or "
              f"--min-rollouts.", file=sys.stderr)
        return 1

    ordered = sorted(live.items(), key=lambda kv: (kv[1], kv[0]))
    if a.max_q_per_video:
        per_video, capped = collections.Counter(), []
        for k, sr in ordered:
            vid = corpus[k]["videos"][0]["video"]
            if per_video[vid] >= a.max_q_per_video:
                continue
            per_video[vid] += 1
            capped.append((k, sr))
        print(f"per-video cap {a.max_q_per_video}: {len(ordered)} -> {len(capped)} questions")
        ordered = capped
        if len(ordered) < want:
            print(f"ERROR: cap leaves {len(ordered)} < {want}", file=sys.stderr)
            return 1
    chosen = ordered[:want]
    thr = chosen[-1][1]
    print(f"selected {len(chosen)} questions, solve_rate <= {thr:.4f}, "
          f"mean {sum(s for _, s in chosen)/len(chosen):.4f}")

    rng = random.Random(a.seed)
    by_video = collections.defaultdict(list)
    for k, sr in chosen:
        by_video[corpus[k]["videos"][0]["video"]].append((k, sr))
    vids = sorted(by_video)
    rng.shuffle(vids)
    val_rows, val_vids = [], set()
    for v in vids:
        if len(val_rows) >= a.val_rows:
            break
        val_vids.add(v)
        val_rows.extend(by_video[v])
    val_rows = val_rows[:a.val_rows]
    val_keys = {k for k, _ in val_rows}
    train_rows = [(k, sr) for k, sr in chosen
                  if corpus[k]["videos"][0]["video"] not in val_vids]
    dropped_edge = len(chosen) - len(train_rows) - len(val_rows)
    train_rows = train_rows[:a.target]

    out = {}
    for split, rr in (("train", train_rows), ("val", val_rows)):
        out[split] = [to_verl(corpus[k], k, sr, a.frames, a.min_pixels, a.max_pixels)
                      for k, sr in rr]

    tr_v = {r["extra_info"]["video_id"] for r in out["train"]}
    va_v = {r["extra_info"]["video_id"] for r in out["val"]}
    assert not (tr_v & va_v), "VIDEO LEAK between train and val"
    for split, rr in out.items():
        qids = [r["extra_info"]["question_id"] for r in rr]
        assert len(qids) == len(set(qids)), f"{split}: duplicate question_id"
        assert all(0.0 < r["extra_info"]["solve_rate"] < 1.0 for r in rr), \
            f"{split}: a dead row survived"
        assert all(r["extra_info"]["reward_type"] == "multiple_choice" for r in rr)
        assert all(str(r["reward_model"]["ground_truth"]).strip() for r in rr)
        assert all(r["videos"][0]["max_frames"] == a.frames
                   and r["videos"][0]["max_pixels"] == a.max_pixels for r in rr), \
            "decode contract does not match the contract the rollout stats were measured at"

    print(f"\ntrain {len(out['train'])} rows / {len(tr_v)} videos | "
          f"val {len(out['val'])} rows / {len(va_v)} videos "
          f"({dropped_edge} rows dropped at the val-video boundary)")
    for split, rr in out.items():
        srs = [r["extra_info"]["solve_rate"] for r in rr]
        print(f"  {split}: mean solve_rate {sum(srs)/len(srs):.4f} | "
              f"bands {dict(collections.Counter(r['extra_info']['difficulty_band'] for r in rr))}")
    print("  train by data_source:",
          dict(collections.Counter(r["data_source"] for r in out["train"]).most_common()))

    if a.dry_run:
        print("[dry-run] nothing written")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    for split, rr in out.items():
        path = os.path.join(a.out_dir, f"{a.prefix}_{split}.parquet")
        pq.write_table(pa.Table.from_pylist(rr, schema=SCHEMA), path)
        print(f"wrote {path}  ({len(rr)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
