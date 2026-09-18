#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

RLVR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = ("__SCRATCH__/cache/huggingface/"
         "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")


def load_reward():
    path = os.path.join(RLVR_ROOT, "rewards", "boxed_reward_wrapper.py")
    spec = importlib.util.spec_from_file_location("boxed_reward_wrapper", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["boxed_reward_wrapper"] = mod
    spec.loader.exec_module(mod)
    return mod.compute_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--max-prompt-length", type=int, default=5120)
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--n-reward", type=int, default=25)
    ap.add_argument("--n-vision", type=int, default=3)
    a = ap.parse_args()

    tr = pd.read_parquet(a.train)
    va = pd.read_parquet(a.val)
    fails = 0

    tv = {r["video_id"] for r in tr.extra_info}
    vv = {r["video_id"] for r in va.extra_info}
    overlap = tv & vv
    print(f"[split] train {len(tr)} rows / {len(tv)} videos | val {len(va)} rows / {len(vv)} videos")
    if overlap:
        print(f"  FAIL: {len(overlap)} videos in BOTH splits: {sorted(overlap)[:5]}")
        fails += 1
    else:
        print("  ok: zero video overlap")

    missing = 0
    for df in (tr, va):
        for v in df.videos:
            p = v[0]["video"].replace("file://", "")
            if not os.path.exists(p):
                missing += 1
    print(f"[files] missing video files: {missing}")
    if missing:
        fails += 1

    compute_score = load_reward()
    print(f"\n[reward] round-trip on {a.n_reward} rows (gold -> must score 1.0; wrong -> must be < 1.0)")
    think = ("I trace each hop through the video in turn, resolve every conditional branch "
             "against what is actually on screen, carry the per-hop numbers forward, and then "
             "apply the stated arithmetic to reach the final integer answer. " * 2)
    good = bad = 0
    for i in np.linspace(0, len(tr) - 1, a.n_reward, dtype=int):
        r = tr.iloc[int(i)]
        gt = r.reward_model["ground_truth"]
        ok_resp = f"<think>{think}</think>\n<answer>\\boxed{{{gt}}}</answer>"
        wrong = f"<think>{think}</think>\n<answer>\\boxed{{{int(gt) + 7}}}</answer>"
        s_ok = compute_score(data_source=r.data_source, solution_str=ok_resp,
                             ground_truth=gt, extra_info=dict(r.extra_info))
        s_bad = compute_score(data_source=r.data_source, solution_str=wrong,
                              ground_truth=gt, extra_info=dict(r.extra_info))
        s_ok = s_ok["score"] if isinstance(s_ok, dict) else s_ok
        s_bad = s_bad["score"] if isinstance(s_bad, dict) else s_bad
        if abs(s_ok - 1.0) < 1e-6:
            good += 1
        else:
            print(f"  FAIL gold gt={gt!r} scored {s_ok}")
        if s_bad < 1.0:
            bad += 1
        else:
            print(f"  FAIL wrong answer for gt={gt!r} also scored {s_bad}")
    print(f"  gold scored 1.0: {good}/{a.n_reward} | wrong scored <1.0: {bad}/{a.n_reward}")
    if good != a.n_reward or bad != a.n_reward:
        fails += 1

    os.environ.setdefault("HF_HOME", "__SCRATCH__/cache/huggingface")
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    print(f"\n[vision] {a.n_vision} rows through the real qwen_vl_utils + processor path")
    worst = 0
    for i in np.linspace(0, len(tr) - 1, a.n_vision, dtype=int):
        r = tr.iloc[int(i)]
        user = next(m for m in r.prompt if m["role"] == "user")["content"]
        conv = [
            {"role": "system", "content": [{"type": "text", "content": None,
                                            "text": r.prompt[0]["content"]}]},
            {"role": "user", "content": [dict(r.videos[0]),
                                         {"type": "text", "text": user.replace("<video>\n", "")}]},
        ]
        _, vids = process_vision_info(conv, image_patch_size=16, return_video_metadata=True)
        vt = [v[0] for v in vids]
        vm = [v[1] for v in vids]
        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        out = proc(text=[text], images=None, videos=vt, video_metadata=vm,
                   do_sample_frames=False, return_tensors="pt")
        n = out["input_ids"].shape[1]
        worst = max(worst, n)
        shape_ok = vt[0].shape[0] == a.frames
        print(f"  {'ok  ' if shape_ok else 'FAIL'} {r.extra_info['video_id']}: "
              f"tensor {tuple(vt[0].shape)}  prompt {n} tokens")
        if not shape_ok:
            fails += 1
    print(f"  worst prompt {worst} tokens vs max_prompt_length {a.max_prompt_length} "
          f"({'ok' if worst <= a.max_prompt_length else 'FAIL — WOULD TRUNCATE'})")
    if worst > a.max_prompt_length:
        fails += 1

    print(f"\n{'ALL GATES PASS' if fails == 0 else f'{fails} GATE(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
