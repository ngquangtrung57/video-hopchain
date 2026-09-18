#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROMPT_TYPE = pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))
IMAGES_TYPE = pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
VIDEOS_TYPE = pa.list_(
    pa.struct(
        [
            ("type", pa.string()),
            ("video", pa.string()),
            ("fps", pa.int64()),
            ("max_frames", pa.int64()),
            ("min_pixels", pa.int64()),
            ("max_pixels", pa.int64()),
        ]
    )
)
REWARD_MODEL_TYPE = pa.struct([("ground_truth", pa.string()), ("style", pa.string())])
EXTRA_INFO_TYPE = pa.struct([("reward_type", pa.string()), ("answer", pa.string())])

EXPECTED_COLUMNS = ["prompt", "images", "videos", "data_source", "ability", "reward_model", "extra_info"]
EXPECTED_TYPES = {
    "prompt": PROMPT_TYPE,
    "images": IMAGES_TYPE,
    "videos": VIDEOS_TYPE,
    "data_source": pa.string(),
    "ability": pa.string(),
    "reward_model": REWARD_MODEL_TYPE,
    "extra_info": EXTRA_INFO_TYPE,
}

KNOWN_REWARD_TYPES = {
    "multiple_choice",
    "numeric",
    "string_match",
    "list_string_match",
    "counting",
    "number_list",
    "grounding",
    "clicking",
    "web_action",
    "search",
    "instruction_following",
    "judge_required",
}

V2_REQUIRED_EXTRA_FIELDS = ("reward_type", "answer")


_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def bold(t: str) -> str:
    return _c("1", t)


def cyan(t: str) -> str:
    return _c("36", t)




def check_schema(table: pa.Table) -> list[str]:
    """Return list of schema error strings (empty = OK)."""
    errors: list[str] = []
    actual_cols = table.schema.names

    for col in EXPECTED_COLUMNS:
        if col not in actual_cols:
            errors.append(f"missing column '{col}'")

    extra = [c for c in actual_cols if c not in EXPECTED_COLUMNS]
    if extra:
        errors.append(f"unexpected extra columns: {extra} (warn only)")

    for col, expected_type in EXPECTED_TYPES.items():
        if col not in actual_cols:
            continue
        actual_type = table.schema.field(col).type
        if actual_type == expected_type:
            continue
        if col == "extra_info" and pa.types.is_struct(actual_type):
            fields = [
                (actual_type.field(i).name, actual_type.field(i).type)
                for i in range(actual_type.num_fields)
            ]
            names = [n for n, _ in fields]
            if (
                len(fields) >= 2
                and names[:2] == list(V2_REQUIRED_EXTRA_FIELDS)
                and fields[0][1] == pa.string()
                and fields[1][1] == pa.string()
            ):
                continue
        errors.append(f"column '{col}' type mismatch: expected {expected_type}, got {actual_type}")

    return errors




def _count_tokens(text: str, token: str) -> int:
    return text.count(token)


def _user_content(prompt: list[dict]) -> str:
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def check_rows(table: pa.Table, fs_check: bool) -> tuple[list[str], list[str]]:
    """Sample up to 100 rows and return (errors, warnings)."""
    n = table.num_rows
    sample_size = min(100, n)

    import random
    rng = random.Random(0)
    if n <= 100:
        indices = list(range(n))
    else:
        indices = sorted(rng.sample(range(n), sample_size))

    prompts: list = []
    images_col: list = []
    videos_col: list = []
    reward_models: list = []
    extra_infos: list = []
    for i in indices:
        row = table.slice(i, 1).to_pylist()[0]
        prompts.append(row["prompt"])
        images_col.append(row["images"])
        videos_col.append(row["videos"])
        reward_models.append(row["reward_model"])
        extra_infos.append(row["extra_info"])

    errors: list[str] = []
    warnings: list[str] = []

    for local_i, global_i in enumerate(indices):
        prefix = f"row[{global_i}]"

        prompt = prompts[local_i]
        if not isinstance(prompt, list) or len(prompt) == 0:
            errors.append(f"{prefix}: prompt is empty or not a list")
            continue

        roles = [m.get("role") for m in prompt if isinstance(m, dict)]
        if "system" not in roles:
            errors.append(f"{prefix}: prompt missing system message")
        if "user" not in roles:
            errors.append(f"{prefix}: prompt missing user message")

        user_text = _user_content(prompt)

        imgs = images_col[local_i] or []
        vids = videos_col[local_i] or []
        n_img_tokens = _count_tokens(user_text, "<image>")
        n_vid_tokens = _count_tokens(user_text, "<video>")

        is_image_row = len(imgs) > 0
        is_video_row = len(vids) > 0

        if is_image_row and is_video_row:
            errors.append(f"{prefix}: both images and videos are non-empty")
        elif is_image_row:
            if len(imgs) != n_img_tokens:
                errors.append(
                    f"{prefix}: images length {len(imgs)} != <image> token count {n_img_tokens}"
                )
            if len(vids) != 0:
                errors.append(f"{prefix}: image row has non-empty videos")
        elif is_video_row:
            if len(vids) != n_vid_tokens:
                errors.append(
                    f"{prefix}: videos length {len(vids)} != <video> token count {n_vid_tokens}"
                )
            if len(imgs) != 0:
                errors.append(f"{prefix}: video row has non-empty images")

        rm = reward_models[local_i]
        if not isinstance(rm, dict):
            errors.append(f"{prefix}: reward_model is not a dict, got {type(rm)}")
        else:
            gt = rm.get("ground_truth", "")
            if not gt:
                errors.append(f"{prefix}: reward_model.ground_truth is empty")

        ei = extra_infos[local_i]
        if not isinstance(ei, dict):
            errors.append(f"{prefix}: extra_info is not a dict, got {type(ei)}")
        else:
            rt = ei.get("reward_type", "")
            if rt not in KNOWN_REWARD_TYPES:
                warnings.append(f"{prefix}: unknown reward_type '{rt}'")

        if is_video_row and fs_check:
            for vid in vids:
                if not isinstance(vid, dict):
                    continue
                vpath = vid.get("video", "")
                if isinstance(vpath, str) and vpath.startswith("file://"):
                    fpath = vpath[len("file://"):]
                    if not os.path.exists(fpath):
                        errors.append(f"{prefix}: video file not found: {fpath}")

    return errors, warnings




def print_aggregate(table: pa.Table, filename: str) -> None:
    n = table.num_rows
    ds_counts = Counter(table.column("data_source").to_pylist())
    import pyarrow.compute as pc
    img_lens = pc.list_value_length(table.column("images")).to_pylist()
    vid_lens = pc.list_value_length(table.column("videos")).to_pylist()
    n_image = sum(1 for x in img_lens if x and x > 0)
    n_video = sum(1 for x in vid_lens if x and x > 0)
    n_neither = n - n_image - n_video

    try:
        rt_arr = pc.struct_field(table.column("extra_info"), "reward_type").to_pylist()
    except Exception:
        rt_arr = []
    rt_counts: Counter = Counter(x for x in rt_arr if x is not None)

    print(f"\n  {bold('Aggregate metrics')} ({filename}, {n} rows):")
    print(f"    {'image rows':<20} {n_image}")
    print(f"    {'video rows':<20} {n_video}")
    if n_neither:
        print(f"    {'text-only rows':<20} {n_neither}")

    print(f"    {'data_source':<20}")
    for src, cnt in sorted(ds_counts.items(), key=lambda x: -x[1]):
        print(f"      {src:<30} {cnt}")

    print(f"    {'reward_type':<20}")
    for rt, cnt in sorted(rt_counts.items(), key=lambda x: -x[1]):
        marker = "" if rt in KNOWN_REWARD_TYPES else yellow(" (unknown)")
        print(f"      {rt:<30} {cnt}{marker}")




def validate_file(path: str, strict: bool, fs_check: bool) -> bool:
    """Returns True if file passes (or only has warnings in non-strict mode)."""
    print(f"\n{bold(cyan(path))}")

    if not os.path.exists(path):
        print(f"  {red('ERROR')}: file not found")
        return False

    try:
        table = pq.read_table(path)
    except Exception as exc:
        print(f"  {red('ERROR')}: failed to read parquet: {exc}")
        return False

    print(f"  rows={table.num_rows}  size={os.path.getsize(path) / 1e6:.1f} MB")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    schema_errors = check_schema(table)
    for e in schema_errors:
        if "warn only" in e:
            all_warnings.append(e.replace(" (warn only)", ""))
        else:
            all_errors.append(e)

    row_errors, row_warnings = check_rows(table, fs_check)
    all_errors.extend(row_errors)
    all_warnings.extend(row_warnings)

    print_aggregate(table, os.path.basename(path))

    if all_warnings:
        print(f"  {yellow('WARNINGS')} ({len(all_warnings)}):")
        for w in all_warnings[:20]:
            print(f"    {yellow('!')} {w}")
        if len(all_warnings) > 20:
            print(f"    ... and {len(all_warnings) - 20} more")

    if all_errors:
        print(f"  {red('ERRORS')} ({len(all_errors)}):")
        for e in all_errors[:30]:
            print(f"    {red('x')} {e}")
        if len(all_errors) > 30:
            print(f"    ... and {len(all_errors) - 30} more")
        print(f"  {red('FAIL')}: {os.path.basename(path)}")
        return False

    print(f"  {green('PASS')}: {os.path.basename(path)}")
    return True




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate verl-ready v1 parquets (schema + content).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Parquet file or directory to validate")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any file fails (warnings are always printed)",
    )
    parser.add_argument(
        "--no-fs-check",
        action="store_true",
        help="Skip video file existence checks on disk",
    )
    args = parser.parse_args()

    fs_check = not args.no_fs_check

    target = Path(args.path)
    if target.is_dir():
        parquet_files = sorted(target.rglob("*.parquet"))
        if not parquet_files:
            print(red(f"No .parquet files found under {args.path}"))
            sys.exit(1)
    elif target.is_file():
        parquet_files = [target]
    else:
        print(red(f"Path not found: {args.path}"))
        sys.exit(1)

    print(bold(f"Validating {len(parquet_files)} parquet file(s)..."))

    results: list[tuple[str, bool]] = []
    for pf in parquet_files:
        ok = validate_file(str(pf), strict=args.strict, fs_check=fs_check)
        results.append((str(pf), ok))

    n_pass = sum(1 for _, ok in results if ok)
    n_fail = len(results) - n_pass
    print(f"\n{bold('Summary')}: {green(str(n_pass))} passed, {red(str(n_fail)) if n_fail else '0'} failed")
    for path, ok in results:
        mark = green("PASS") if ok else red("FAIL")
        print(f"  [{mark}] {path}")

    if n_fail > 0 and args.strict:
        sys.exit(1)
    elif n_fail > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
