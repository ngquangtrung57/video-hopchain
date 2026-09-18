#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curate_prelim_data import SYSTEM_PROMPT
from build_rl_corpus_30k import SCHEMA as SCHEMA_30K

_EXTRA = SCHEMA_30K.field("extra_info").type
SCHEMA = SCHEMA_30K.set(
    SCHEMA_30K.get_field_index("extra_info"),
    pa.field("extra_info", pa.struct(list(_EXTRA) + [pa.field("tolerance", pa.float64())])),
)

sys.path.insert(0, "__HOME__/pipeline")
from pipeline.hopgen import branch_totals_unique
from pipeline.query_render import split_mapping

ABILITY = "video_multihop_reasoning"
PROXY_BASE = "__SCRATCH__/videoreason"
PROXY_DIRS = [f"{PROXY_BASE}/proxy_longvila_140f50k", f"{PROXY_BASE}/proxy_finevideo_140f50k",
              f"{PROXY_BASE}/proxy_vript_140f50k"]
_ORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7}
_CHECK_REF = re.compile(r"\bcheck\s+(one|two|three|four|five|six|seven|\d)\b"
                        r"|\b(first|second|third|fourth|fifth|sixth|seventh)\s+check\b", re.I)
_HOP_WORD = re.compile(r"\bhop\s*\d+", re.I)


def dangling_check_ref(query: str, n: int) -> bool:
    """True if the query names a check number the question does not have."""
    for m in _CHECK_REF.finditer(query):
        tok = (m.group(1) or m.group(2) or "").lower()
        i = int(tok) if tok.isdigit() else _ORD.get(tok, 0)
        if i < 1 or i > n:
            return True
    return False


_LET = re.compile(r"\blet ([A-H]) be\b", re.I)


def letter_hop_mismatch(query: str, n: int) -> bool:
    got = {m.group(1).upper() for m in _LET.finditer(query or "")}
    return bool(got) and len(got) != n


_DUP_THRESHOLD = 0.35
_STOP = set("the a an of in on at to is are and or moment where scene with his her its their".split())


def _toks(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 2}


def scene_refs(c) -> list:
    return [str(h.get("scene_ref") or "")
            for h in (c.get("reasoning_hops") or [])[:n_hops_of(c)] if h.get("scene_ref")]


def shared_scenes(a, b) -> int:
    out = 0
    for x in a:
        tx = _toks(x)
        if tx and any(len(tx & _toks(y)) / max(1, len(tx | _toks(y))) >= _DUP_THRESHOLD for y in b):
            out += 1
    return out


_PRED_STOP = set("the a an of in on at to is are and or if let be otherwise then it its his her "
                 "their side frame moment scene where with while shown".split())


def _pred_toks(s: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", (s or "").lower()) if w not in _PRED_STOP}


def predicate_collisions(a, b) -> int:
    n = 0
    for ta, pa, sa in _hop_signature(a):
        for tb, pb, sb in _hop_signature(b):
            if ta != tb or sa != sb or not pa or not pb:
                continue
            if len(pa & pb) / max(1, len(pa | pb)) >= 0.5:
                n += 1
    return n


def _hop_signature(c) -> list:
    """(family, predicate tokens, taken-branch side) per real hop."""
    out = []
    for h in (c.get("reasoning_hops") or [])[:n_hops_of(c)]:
        p = split_mapping(h.get("mapping"))
        side = None
        if p:
            try:
                v, t, e = int(str(h.get("value")).strip()), int(p[1]), int(p[2])
                side = "T" if v == t else ("E" if v == e else None)
            except (TypeError, ValueError):
                pass
        out.append((h.get("evidence_type") or "?", _pred_toks(h.get("description")), side))
    return out


def pick_diverse(rows: list, cands: list, cap: int) -> list:
    import itertools
    if len(rows) <= cap:
        return rows
    best, best_key = None, None
    for combo in itertools.combinations(range(len(rows)), cap):
        collide = sum(predicate_collisions(cands[i], cands[j])
                      for i, j in itertools.combinations(combo, 2))
        overlap = sum(shared_scenes(scene_refs(cands[i]), scene_refs(cands[j]))
                      for i, j in itertools.combinations(combo, 2))
        overlap = (collide, overlap)
        key = (overlap, tuple(rows[i]["extra_info"]["question_id"] for i in combo))
        if best_key is None or key < best_key:
            best, best_key = combo, key
    return [rows[i] for i in best]


def n_hops_of(c):
    """Hop count from the expression, NOT len(reasoning_hops) -- there is a TRAILING TOTAL ROW."""
    return len(set(re.findall(r"H(\d+)", c.get("arithmetic_expression") or "")))


def check(r: dict, min_hops: int) -> str:
    """"" when the row is usable, else the reason it is not."""
    n = n_hops_of(r)
    if n < min_hops:
        return "hops_below_min"
    if n > 6:
        return "hops_above_max"
    hops = list(r.get("reasoning_hops") or [])[:n]
    if len(hops) != n:
        return "hop_rows_missing"
    q = (r.get("query") or "").strip()
    if not q:
        return "no_query"
    if _HOP_WORD.search(q):
        return "hop_word_in_query"
    if letter_hop_mismatch(q, n):
        return "letter_hop_mismatch"
    if dangling_check_ref(q, n):
        return "dangling_check_reference"
    if r.get("selectors"):
        for sel in r["selectors"]:
            if sel.get("selected_by_hop") is None or sel.get("selected_hop") is None:
                return "selector_incomplete"
    vals = []
    for h in hops:
        v = str(h.get("value") or "").strip()
        if not v.lstrip("-").isdigit():
            return "nonnumeric_hop_value"
        vals.append(int(v))
    gold = str(r.get("hypothetical_answer") or "").strip()
    if not gold.lstrip("-").isdigit():
        return "gold_not_int"
    if sum(vals) != int(gold):
        return "sum_mismatch"
    br = []
    for h in hops:
        p = split_mapping(h.get("mapping"))
        if p is None:
            return "unparseable_mapping"
        try:
            br.append([int(p[1]), int(p[2])])
        except (TypeError, ValueError):
            return "nonnumeric_branch"
    if not branch_totals_unique(br):
        return "branch_totals_collide"
    return ""


def to_verl(r: dict, vpath: str, data_source: str, frames: int,
            min_pixels: int, max_pixels: int) -> dict:
    n = n_hops_of(r)
    hops = list(r.get("reasoning_hops") or [])[:n]
    gold = str(int(str(r["hypothetical_answer"]).strip()))
    hop_types = [str(h.get("evidence_type") or "") for h in hops
                 if (h.get("evidence_type") or "") != "arithmetic"]
    qid = str(r.get("question_id") or f"{r['video_id']}::{r.get('id')}")
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": "<video>\n" + r["query"].strip()}],
        "images": [],
        "videos": [{"fps": 1, "max_frames": frames, "max_pixels": max_pixels,
                    "min_pixels": min_pixels, "type": "video", "video": f"file://{vpath}"}],
        "data_source": data_source,
        "ability": ABILITY,
        "reward_model": {"ground_truth": gold, "style": "rule"},
        "extra_info": {
            "reward_type": "numeric", "answer": gold, "answer_type": "numeric",
            "num_hops": n, "hop_types": hop_types,
            "question_id": qid, "video_id": r["video_id"],
            "solve_rate": -1.0, "difficulty_band": "",
            "tolerance": 0.0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="hopchain")
    ap.add_argument("--min-hops", type=int, default=3)
    ap.add_argument("--max-q-per-video", type=int, default=2)
    ap.add_argument("--val-videos", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--min-pixels", type=int, default=3136)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    proxy = {}
    for d in PROXY_DIRS:
        src = os.path.basename(d).replace("proxy_", "").replace("_140f50k", "")
        for fn in os.listdir(d):
            if fn.endswith(".mp4"):
                proxy.setdefault(fn[:-4], (src, os.path.join(d, fn)))

    stats = collections.Counter()
    by_video = collections.defaultdict(list)
    with open(a.final, encoding="utf-8") as fh:
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
            why = check(r, a.min_hops)
            if why:
                stats["drop_" + why] += 1
                continue
            hit = proxy.get(vid)
            if hit is None:
                stats["drop_no_proxy"] += 1
                continue
            src, vpath = hit
            row = to_verl(r, vpath, f"hopchain_{src}", a.frames, a.min_pixels, a.max_pixels)
            by_video[vid].append((row, r))
            stats["kept"] += 1

    dropped_dup = 0
    for vid in by_video:
        by_video[vid].sort(key=lambda x: x[0]["extra_info"]["question_id"])
        rows = [x[0] for x in by_video[vid]]
        cands = [x[1] for x in by_video[vid]]
        n_before = len(rows)
        kept = pick_diverse(rows, cands, a.max_q_per_video)
        dropped_dup += n_before - len(kept)
        by_video[vid] = kept
    stats["capped_per_video"] = dropped_dup

    vids = sorted(by_video)
    random.Random(a.seed).shuffle(vids)
    val_vids = set(vids[:a.val_videos])
    train_vids = [v for v in vids if v not in val_vids]

    train = [r for v in sorted(train_vids) for r in by_video[v]]
    val = [r for v in sorted(val_vids) for r in by_video[v]]

    for k in sorted(stats):
        print(f"{k:32s} {stats[k]}")
    print(f"\ntrain {len(train)} rows / {len(train_vids)} videos "
          f"({len(train) / max(1, len(train_vids)):.2f} q/video)")
    print(f"val   {len(val)} rows / {len(val_vids)} videos")
    assert not (set(train_vids) & val_vids), "train/val video overlap"

    if a.dry_run:
        print("\n-- dry run, nothing written --")
        return 0

    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        dst = os.path.join(a.out_dir, f"{a.prefix}_{name}.parquet")
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), dst)
        print(f"wrote {len(rows):>6} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
