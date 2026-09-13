"""Stage 10, assemble and verify: render the rows, split by video, and run the release gates.

The question text is rendered by code from the hop record. At most two questions per video
survive, chosen so that the two reuse as few of the same moments and hops as possible. The
split is by video, so no video appears on both sides.
"""

from __future__ import annotations

import random

from .schema import answer_of, build_row, hops_of, render_question

SPLIT_SEED = 42

MAX_PER_VIDEO = 2

MAX_PROMPT_TOKENS = 5376

CHARS_PER_TOKEN = 4


def _fingerprint(question: dict) -> tuple:
    """The moments and hop families a question uses."""
    hops = hops_of(question)
    return ({str(h.get("scene_ref") or "").strip().lower() for h in hops},
            {str(h.get("evidence_type") or "").strip().lower() for h in hops})


def choose_pair(questions: list, limit: int = MAX_PER_VIDEO) -> list:
    """Keep at most limit questions of one video, reusing as few moments and hops as possible."""
    if len(questions) <= limit:
        return list(questions)
    if limit != 2:
        return list(questions)[:limit]
    best, best_score = None, None
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            scenes_i, families_i = _fingerprint(questions[i])
            scenes_j, families_j = _fingerprint(questions[j])
            score = len(scenes_i & scenes_j) + len(families_i & families_j)
            if best_score is None or score < best_score:
                best, best_score = [questions[i], questions[j]], score
    return best or list(questions)[:limit]


def split_videos(video_ids: list, held_out: int, seed: int = SPLIT_SEED) -> tuple:
    """Draw the held-out videos. The split is by video, so no video is on both sides."""
    ordered = sorted(set(video_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    held = set(ordered[:max(0, held_out)])
    return [v for v in sorted(set(video_ids)) if v not in held], sorted(held)


def assemble(questions: list, video_paths: dict, sources: dict, *, held_out: int = 0) -> tuple:
    """Build the released rows. Returns (train_rows, val_rows)."""
    by_video = {}
    for question in questions:
        by_video.setdefault(str(question.get("video_id")), []).append(question)

    train_ids, val_ids = split_videos(list(by_video), held_out)
    train_rows, val_rows = [], []
    for video_id, group in by_video.items():
        path = video_paths.get(video_id)
        if not path:
            continue
        source = sources.get(video_id, "hopchain")
        if video_id in set(val_ids):
            chosen = choose_pair(group, 1)
            val_rows += [build_row(q, video_path=path, data_source=source) for q in chosen]
        else:
            chosen = choose_pair(group, MAX_PER_VIDEO)
            train_rows += [build_row(q, video_path=path, data_source=source) for q in chosen]
    return train_rows, val_rows


def release_gates(train_rows: list, val_rows: list, video_paths: dict) -> list:
    """The release gates. Returns the list of failures; empty means the corpus may ship."""
    import os

    failures = []

    train_videos = {r["extra_info"]["video_id"] for r in train_rows}
    val_videos = {r["extra_info"]["video_id"] for r in val_rows}
    shared = train_videos & val_videos
    if shared:
        failures.append(f"{len(shared)} video(s) appear in both splits")

    missing = [v for v in train_videos | val_videos
               if not os.path.exists(video_paths.get(v, ""))]
    if missing:
        failures.append(f"{len(missing)} video file(s) do not resolve")

    for row in train_rows + val_rows:
        gold = str(row["reward_model"]["ground_truth"])
        if not gold.lstrip("-").isdigit():
            failures.append(f"{row['extra_info']['question_id']}: answer is not an integer")
            break
        # An off-by-one answer must score zero. It does under an exact match, but
        # extra_info carries a numeric tolerance, and a tolerance of one or more
        # would let a wrong answer score one.
        tolerance = float(row["extra_info"].get("tolerance", 0.0) or 0.0)
        if tolerance >= 1.0:
            failures.append(f"{row['extra_info']['question_id']}: tolerance {tolerance} "
                            "lets an off-by-one answer score 1")
            break
        if str(row["extra_info"].get("answer")) != gold:
            failures.append(f"{row['extra_info']['question_id']}: extra_info answer "
                            "disagrees with the ground truth")
            break

    for row in train_rows + val_rows:
        video = (row.get("videos") or [{}])[0]
        if int(video.get("nframes", 0)) != 140:
            failures.append(f"{row['extra_info']['question_id']}: proxy is not 140 frames")
            break

    longest = max((len(r["prompt"][0]["content"]) + len(r["prompt"][1]["content"])
                   for r in train_rows + val_rows), default=0)
    if longest / CHARS_PER_TOKEN > MAX_PROMPT_TOKENS:
        failures.append(f"the longest prompt needs about {longest // CHARS_PER_TOKEN} tokens, "
                        f"over the {MAX_PROMPT_TOKENS} of the training context")

    return failures


def write_parquet(rows: list, out_path: str) -> str:
    """Write rows to a parquet file, with the schema the trainer expects.

    Every structured column stays structured. An earlier version stored prompt,
    videos, reward_model and extra_info as JSON strings, which loads without
    error and then fails inside the trainer, because verl indexes prompt as a
    list of {role, content} dicts and a string indexes characters instead.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    return out_path


__all__ = ["MAX_PER_VIDEO", "SPLIT_SEED", "answer_of", "assemble", "choose_pair",
           "release_gates", "render_question", "split_videos", "write_parquet"]
