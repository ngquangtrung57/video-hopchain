#!/usr/bin/env python3
"""Run the pipeline over a folder of videos, one stage after another.

Every stage writes its artefact under --out, so a stage can be rerun on the output of the
one before it. The model stages call an OpenAI-compatible endpoint; see --base-url.

    python run.py --videos VIDEOS --out WORK \
        --caption-model CAPTIONER --text-model GENERATOR --solver-model BASE_MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pipeline import assemble, caption, checks, difficulty, generate, judge, proxy
from pipeline.model import Chat
from pipeline.segment import build_caption_skeleton

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")


def _load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save(path: str, value) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=1)


def caption_one(args, chat, video_path: str, video_id: str) -> dict | None:
    """Stages 1 to 3 for one video: admit it, write the proxy, segment it and caption it."""
    admitted, why = proxy.admit(video_path)
    if not admitted:
        print(f"{video_id}: not admitted, {why}")
        return None

    proxy_path = os.path.join(args.out, "proxies", video_id + ".mp4")
    if not os.path.exists(proxy_path):
        proxy.write_proxy(video_path, proxy_path)

    caption_path = os.path.join(args.out, "captions", video_id + ".json")
    if os.path.exists(caption_path):
        document = _load(caption_path)
    else:
        skeleton = build_caption_skeleton(video_path, video_id)
        document = caption.caption_video(chat, video_path, skeleton)
        _save(caption_path, document)

    kept, why = caption.keep(document)
    if not kept:
        print(f"{video_id}: caption dropped, {why}")
        return None

    return document


def clean_question(args, chat, document: dict, candidate: dict):
    """Stages 7 and 8: judge one candidate, and write it again while the judge faults a hop."""
    video_id = str(document.get("video_id"))
    for attempt in range(1, args.attempts + 1):
        faults = judge.judge(chat, candidate, document)
        if not faults:
            return candidate
        print(f"{video_id}: {candidate.get('id')} faulted at hop(s) {sorted(faults)} "
              f"on attempt {attempt}")
        if attempt == args.attempts:
            break
        candidate = generate.regenerate(chat, document, candidate,
                                        candidates=args.candidates)
        if candidate is None:
            print(f"{video_id}: the generator returned nothing, question dropped")
            return None
        kept, dropped = checks.run([candidate])
        if not kept:
            print(f"{video_id}: the new question fails the code checks, {dropped[0][1]}")
            return None
        candidate = kept[0]
    print(f"{video_id}: still faulted after {args.attempts} attempts, question dropped")
    return None


def questions_of(args, text_chat, solver_chat, document: dict, proxy_path: str) -> list:
    """Stages 4 to 9 for one video: specify, generate, check, judge, regenerate, filter."""
    video_id = str(document.get("video_id"))
    candidates = generate.generate(text_chat, document, candidates=args.candidates)
    candidates, dropped = checks.run(candidates)
    for question_id, reason in dropped:
        print(f"{video_id}: {question_id} dropped by the code checks, {reason}")

    survivors = []
    for candidate in candidates:
        question = clean_question(args, text_chat, document, candidate)
        if question is None:
            continue
        if solver_chat is not None:
            rate = difficulty.solve_rate(solver_chat, question, proxy_path)
            if difficulty.too_easy(rate):
                print(f"{video_id}: {question.get('id')} solved in {rate:.0%} of rollouts, dropped")
                continue
        survivors.append(question)
    return survivors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder of source videos")
    parser.add_argument("--out", required=True, help="folder for the artefacts of every stage")
    parser.add_argument("--caption-model", required=True, help="the captioner, stage 3")
    parser.add_argument("--text-model", required=True,
                        help="the generator and the judge, stages 5, 7 and 8")
    parser.add_argument("--solver-model", default="",
                        help="the base model of the difficulty filter, stage 9; "
                             "leave unset to skip that stage")
    parser.add_argument("--base-url", default=None, help="the OpenAI-compatible endpoint")
    parser.add_argument("--candidates", type=int, default=2,
                        help="questions drawn per video, stage 4")
    parser.add_argument("--attempts", type=int, default=3,
                        help="judge and regenerate attempts per question, stages 7 and 8")
    parser.add_argument("--held-out", type=int, default=0,
                        help="videos to hold out for the validation split, stage 10")
    parser.add_argument("--source", default="hopchain", help="data_source of the released rows")
    args = parser.parse_args()

    client = {"base_url": args.base_url} if args.base_url else {}
    caption_chat = Chat(args.caption_model, **client)
    text_chat = Chat(args.text_model, **client)
    solver_chat = Chat(args.solver_model, **client) if args.solver_model else None

    videos = sorted(name for name in os.listdir(args.videos)
                    if name.lower().endswith(VIDEO_SUFFIXES))
    if not videos:
        print(f"no videos under {args.videos}")
        return 1

    questions, video_paths, sources = [], {}, {}
    for name in videos:
        video_id = os.path.splitext(name)[0]
        video_path = os.path.join(args.videos, name)
        document = caption_one(args, caption_chat, video_path, video_id)
        if document is None:
            continue
        proxy_path = os.path.join(args.out, "proxies", video_id + ".mp4")
        found = questions_of(args, text_chat, solver_chat, document, proxy_path)
        if not found:
            continue
        _save(os.path.join(args.out, "questions", video_id + ".json"), found)
        questions += found
        video_paths[video_id] = proxy_path
        sources[video_id] = args.source
        print(f"{video_id}: {len(found)} question(s) kept")

    if not questions:
        print("no questions survived")
        return 1

    train_rows, val_rows = assemble.assemble(questions, video_paths, sources,
                                             held_out=args.held_out)
    failures = assemble.release_gates(train_rows, val_rows, video_paths)
    for failure in failures:
        print(f"release gate: {failure}")
    if failures:
        return 1

    for rows, split in ((train_rows, "train"), (val_rows, "val")):
        if rows:
            path = os.path.join(args.out, f"videohopchain_{split}.parquet")
            assemble.write_parquet(rows, path)
            print(f"{split}: {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
