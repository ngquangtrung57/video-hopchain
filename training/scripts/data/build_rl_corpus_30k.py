#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The prompt that every training row carries, inlined so this file stands alone.
SYSTEM_PROMPT = (
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
    "Always wrap reasoning in <think>...</think> and answer in <answer>\\boxed{...}</answer>. "
    "No text outside these tags."
)

ABILITY = "video_multihop_reasoning"

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


def _s(x):
    """Normalise a staging field: the JSONL stringifies None as the literal 'None'."""
    return None if x in (None, "None", "") else str(x).strip()


def _i(x, default=0):
    v = _s(x)
    return int(v) if v is not None and v.lstrip("-").isdigit() else default


def _is_int(x):
    return x is not None and x.lstrip("-").isdigit()


def gate_row(r: dict) -> str:
    if _s(r.get("judge_status")) == "dropped":
        return "judge_dropped"
    gold = _s(r.get("gold_recommended"))
    if not _is_int(gold):
        return "gold_not_int"
    if not _s(r.get("query_canonical")):
        return "no_canonical_query"
    if _s(r.get("repair_status")) == "irreparable":
        return "irreparable"
    n_hops = _s(r.get("n_hops"))
    if not (n_hops and n_hops.isdigit() and 3 <= int(n_hops) <= 7):
        return "hops_out_of_range"
    if _s(r.get("gold_from_condition_verdicts")) != gold:
        return "cv_gold_mismatch"
    judge_gold = _s(r.get("gold_from_judge"))
    if judge_gold is not None and judge_gold != gold:
        return "judge_gold_mismatch"
    if _i(r.get("n_hop_value_vs_verdict_conflicts")) > 0:
        return "hop_verdict_conflict"
    if _i(r.get("n_hops_quote_unanchored")) > 0:
        return "quote_unanchored"
    if _i(r.get("n_hops_scene_ref_restates_condition")) > 0:
        return "scene_ref_restates"
    return ""


def staging_to_verl(r: dict, proxy_dir: str, data_source: str, frames: int,
                    min_pixels: int, max_pixels: int) -> dict:
    """One gated staging row -> one verl row, in the live corpus's exact shape."""
    vid = r["video_id"]
    gold = str(int(_s(r["gold_recommended"])))
    hops = r.get("reasoning_hops") or []
    hop_types = [str(h.get("evidence_type") or "") for h in hops
                 if (h.get("evidence_type") or "") != "arithmetic"]
    vpath = os.path.join(proxy_dir, f"{vid}.mp4")
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": "<video>\n" + _s(r["query_canonical"])}],
        "images": [],
        "videos": [{"fps": 1, "max_frames": frames, "max_pixels": max_pixels,
                    "min_pixels": min_pixels, "type": "video", "video": f"file://{vpath}"}],
        "data_source": data_source,
        "ability": ABILITY,
        "reward_model": {"ground_truth": gold, "style": "rule"},
        "extra_info": {
            "reward_type": "numeric", "answer": gold, "answer_type": "numeric",
            "num_hops": _i(r.get("n_hops"), len(hop_types)), "hop_types": hop_types,
            "question_id": str(r.get("question_id")), "video_id": vid,
            "solve_rate": -1.0, "difficulty_band": "",
        },
    }


def rank_key(row: dict, judged: bool):
    return (0 if judged else 1, row["extra_info"]["question_id"])


def load_staging(name: str, path: str, proxy_dir: str, exclude: set, args):
    have = {f[:-4] for f in os.listdir(proxy_dir) if f.endswith(".mp4")}
    stats = collections.Counter()
    by_video = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                stats["drop_unreadable"] += 1
                continue
            stats["seen"] += 1
            vid = r.get("video_id")
            if not vid:
                stats["drop_no_video_id"] += 1
                continue
            if vid in exclude:
                stats["drop_excluded"] += 1
                continue
            why = gate_row(r)
            if why:
                stats["drop_" + why] += 1
                continue
            if vid not in have:
                stats["drop_no_proxy"] += 1
                continue
            row = staging_to_verl(r, proxy_dir, f"hopchain_{name}_numeric",
                                  args.frames, args.min_pixels, args.max_pixels)
            by_video[vid].append((rank_key(row, _s(r.get("judge_status")) == "kept"), row))
            stats["kept"] += 1
    return by_video, stats


def load_carry(name: str, path: str, exclude: set, proxy_roots: list[str]):
    stats = collections.Counter()
    by_video = collections.defaultdict(list)
    index = {}
    for root in proxy_roots:
        if not os.path.isdir(root):
            continue
        for fn in os.listdir(root):
            if fn.endswith(".mp4"):
                index.setdefault(fn[:-4], os.path.join(root, fn))
    table = pq.read_table(path)
    for row in table.to_pylist():
        stats["seen"] += 1
        vid = row["extra_info"]["video_id"]
        if vid in exclude:
            stats["drop_excluded"] += 1
            continue
        vpath = row["videos"][0]["video"][len("file://"):]
        if not any(os.path.dirname(vpath) == r.rstrip("/") for r in proxy_roots):
            proxy = index.get(vid)
            if proxy is None:
                stats["drop_no_proxy"] += 1
                continue
            row["videos"][0]["video"] = f"file://{proxy}"
            stats["repointed_to_proxy"] += 1
        elif not os.path.exists(vpath):
            stats["drop_no_proxy"] += 1
            continue
        by_video[vid].append((rank_key(row, False), row))
        stats["kept"] += 1
    return by_video, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", action="append", default=[],
                    help="NAME:JSONL:PROXY_DIR — repeatable")
    ap.add_argument("--carry", action="append", default=[],
                    help="NAME:PARQUET — an already-built verl parquet. Repeatable")
    ap.add_argument("--proxy-root", action="append", default=[],
                    help="extra proxy dir used to repoint raw-source paths in --carry parquets. "
                         "The --staging proxy dirs are searched automatically. Repeatable")
    ap.add_argument("--exclude-ids", action="append", default=[],
                    help="file of video ids to EXCLUDE. Repeatable")
    ap.add_argument("--max-q-per-video", type=int, default=2)
    ap.add_argument("--target-videos", type=int, default=15000, help="TRAIN videos")
    ap.add_argument("--val-videos", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="rl_30k_2qpv")
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--min-pixels", type=int, default=3136)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cap = a.max_q_per_video
    if cap < 1:
        print("ERROR: --max-q-per-video must be >= 1", file=sys.stderr)
        return 1

    exclude = set()
    for p in a.exclude_ids:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                vid = line.split("\t")[0].strip()
                if vid:
                    exclude.add(vid)
    if exclude:
        print(f"excluding {len(exclude)} video ids")

    pools: dict[str, dict] = {}
    for spec in a.staging:
        name, path, proxy_dir = spec.split(":", 2)
        by_video, stats = load_staging(name, path, proxy_dir, exclude, a)
        pools[name] = by_video
        print(f"\n[staging {name}]")
        for k in sorted(stats):
            print(f"  {k:28s} {stats[k]:7d}")
    proxy_roots = [spec.split(":", 2)[2] for spec in a.staging] + a.proxy_root
    for spec in a.carry:
        name, path = spec.split(":", 1)
        by_video, stats = load_carry(name, path, exclude, proxy_roots)
        pools[name] = by_video
        print(f"\n[carry {name}]")
        for k in sorted(stats):
            print(f"  {k:28s} {stats[k]:7d}")
    if not pools:
        print("ERROR: no --staging and no --carry given", file=sys.stderr)
        return 1

    qualified = {n: sorted(v for v, rows in by.items() if len(rows) >= cap)
                 for n, by in pools.items()}
    total_q = sum(len(v) for v in qualified.values())
    want_videos = a.target_videos + a.val_videos
    print(f"\nvideos with >= {cap} gated questions:")
    for n in sorted(qualified):
        print(f"  {n:12s} {len(qualified[n]):6d}")
    print(f"  {'TOTAL':12s} {total_q:6d}   -> max rows at cap {cap} = {total_q * cap}")
    print(f"want {want_videos} videos ({a.target_videos} train + {a.val_videos} val) "
          f"= {want_videos * cap} rows")
    if total_q < want_videos:
        print(f"ERROR: short by {want_videos - total_q} videos. Lower --target-videos, "
              f"widen the pools, or lower --max-q-per-video.", file=sys.stderr)
        return 1

    rng = random.Random(a.seed)
    names = sorted(qualified)
    exact = {n: len(qualified[n]) / total_q * want_videos for n in names}
    alloc = {n: int(exact[n]) for n in names}
    for n in sorted(names, key=lambda k: (-(exact[k] - alloc[k]), k))[:want_videos - sum(alloc.values())]:
        alloc[n] += 1
    assert sum(alloc.values()) == want_videos, "allocation does not sum to the target"

    chosen: dict[str, list] = {}
    for n in names:
        vids = qualified[n][:]
        rng.shuffle(vids)
        chosen[n] = vids[:alloc[n]]
        print(f"  draw {n:12s} {len(chosen[n]):6d} of {len(qualified[n]):6d}")

    val_alloc = {n: int(len(chosen[n]) / want_videos * a.val_videos) for n in names}
    for n in sorted(names, key=lambda k: (-(len(chosen[k]) / want_videos * a.val_videos
                                            - val_alloc[k]), k))[:a.val_videos - sum(val_alloc.values())]:
        val_alloc[n] += 1
    assert sum(val_alloc.values()) == a.val_videos, "val allocation does not sum"

    rows_tr, rows_va = [], []
    for n in names:
        val_vids = set(chosen[n][:val_alloc[n]])
        for vid in chosen[n]:
            picked = [row for _, row in sorted(pools[n][vid], key=lambda kr: kr[0])[:cap]]
            (rows_va if vid in val_vids else rows_tr).extend(picked)

    tr_v = {r["extra_info"]["video_id"] for r in rows_tr}
    va_v = {r["extra_info"]["video_id"] for r in rows_va}

    assert not (tr_v & va_v), "VIDEO LEAK between train and val"
    assert not (tr_v & exclude) and not (va_v & exclude), "an excluded video reached the corpus"
    assert len(tr_v) == a.target_videos, f"train videos {len(tr_v)} != {a.target_videos}"
    assert len(va_v) == a.val_videos, f"val videos {len(va_v)} != {a.val_videos}"
    for split, rr in (("train", rows_tr), ("val", rows_va)):
        per_vid = collections.Counter(r["extra_info"]["video_id"] for r in rr)
        assert set(per_vid.values()) == {cap}, f"{split}: a video does not carry exactly {cap} rows"
        qids = [r["extra_info"]["question_id"] for r in rr]
        assert len(qids) == len(set(qids)), f"{split}: duplicate question_id"
        assert all(r["extra_info"]["reward_type"] == "numeric" for r in rr)
        assert all(str(r["reward_model"]["ground_truth"]).lstrip("-").isdigit() for r in rr)
        assert all(r["prompt"][0]["content"] == SYSTEM_PROMPT for r in rr)
        assert all(r["prompt"][1]["content"].startswith("<video>\n") for r in rr)
        assert all(r["videos"][0]["video"].startswith("file://") for r in rr)
        stray = [r["extra_info"]["question_id"] for r in rr
                 if not any(os.path.dirname(r["videos"][0]["video"][len("file://"):])
                            == root.rstrip("/") for root in proxy_roots)]
        assert not stray, (f"{split}: {len(stray)} rows point outside every proxy root "
                           f"(raw-source decode is ~34x slower); first: {stray[:3]}")
    all_qids = {r["extra_info"]["question_id"] for r in rows_tr} | \
               {r["extra_info"]["question_id"] for r in rows_va}
    assert len(all_qids) == len(rows_tr) + len(rows_va), "question_id collides across the splits"

    print(f"\ntrain {len(rows_tr)} rows / {len(tr_v)} videos"
          f" | val {len(rows_va)} rows / {len(va_v)} videos")
    src = collections.Counter(r["data_source"] for r in rows_tr)
    print("train by data_source:", dict(src.most_common()))
    known = [r["extra_info"]["solve_rate"] for r in rows_tr if r["extra_info"]["solve_rate"] >= 0]
    print(f"train rows with a REAL solve_rate: {len(known)} / {len(rows_tr)}"
          + (f"  (mean {sum(known)/len(known):.4f})" if known else ""))
    hops = collections.Counter(r["extra_info"]["num_hops"] for r in rows_tr)
    print("train num_hops:", dict(sorted(hops.items())))

    if a.dry_run:
        print("[dry-run] nothing written")
        return 0

    os.makedirs(a.out_dir, exist_ok=True)
    for split, rr in (("train", rows_tr), ("val", rows_va)):
        out = os.path.join(a.out_dir, f"{a.prefix}_{split}.parquet")
        pq.write_table(pa.Table.from_pylist(rr, schema=SCHEMA), out)
        print(f"wrote {out}  ({len(rr)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
