# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional video-decode controls for the AgentLoop rollout path.

Three independent knobs, all OFF by default. With no env var set,
`decode_multi_modal_info` calls the original synchronous function exactly as
`RLHFDataset` always has.

  RVRL_DECODE_THREADS   int, 0 = leave torch threading alone. Otherwise the
                        torch intra-op thread count used during decode, which
                        is otherwise sized for the whole node.
  RVRL_DECODE_CACHE     int, 0 = disabled. Entries in the decoded-video cache,
                        which lets the n rollouts of one prompt share a decode.
  RVRL_DECODE_EXECUTOR  "1" to run decode in the default thread pool instead of
                        inline on the event loop.

Set RVRL_DECODE_THREADS before RVRL_DECODE_EXECUTOR: offloading decodes that
each claim every core multiplies the over-subscription instead of fixing it.
"""

import asyncio
import hashlib
import json
import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _int_env(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DECODE_THREADS = _int_env("RVRL_DECODE_THREADS", 0)
DECODE_CACHE_SIZE = _int_env("RVRL_DECODE_CACHE", 0)
DECODE_USE_EXECUTOR = os.getenv("RVRL_DECODE_EXECUTOR", "0") == "1"

_thread_cap_applied = False


def _apply_thread_cap() -> None:
    """Cap torch intra-op threads once per process.

    Set at runtime rather than via OMP_NUM_THREADS because the env var is read
    when the OpenMP runtime initialises, which has already happened by the time a
    Ray actor starts. torch.set_num_threads takes effect immediately and covers
    the resize/float conversion that dominates decode CPU.
    """
    global _thread_cap_applied
    if _thread_cap_applied or DECODE_THREADS <= 0:
        return
    try:
        import torch

        torch.set_num_threads(DECODE_THREADS)
        logger.warning("[rvrl-decode] torch.set_num_threads(%d)", DECODE_THREADS)
    except Exception as e:  # pragma: no cover - never let a perf knob kill a run
        logger.warning("[rvrl-decode] could not cap threads: %s", e)
    _thread_cap_applied = True


# key -> decoded result;  key -> in-flight future (single-flight)
_cache: "dict[str, tuple]" = {}
_cache_order: "list[str]" = []
_inflight: "dict[str, asyncio.Future]" = {}
_lock: "asyncio.Lock | None" = None

_stats = {"hit": 0, "miss": 0}


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def _cache_key(messages) -> "str | None":
    """Stable key over the media items, or None when the row must not be cached.

    Returns None if any IMAGE item is present. `_process_multi_modal_info`
    MUTATES image dicts (`item.setdefault("max_pixels", ...)`) and those same
    message objects are later fed to apply_chat_template, so serving a cached
    decode would skip a mutation that changes the prompt. Video rows have no such
    mutation, and this corpus is video-only, so the cache still applies to it.
    """
    parts = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "image":
                return None
            if kind == "video":
                payload = {k: v for k, v in item.items() if k != "type"}
                parts.append(json.dumps(payload, sort_keys=True, default=str))
    if not parts:
        return None
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


async def _run_raw(raw_fn, messages, image_patch_size, config):
    if DECODE_USE_EXECUTOR:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: raw_fn(messages, image_patch_size=image_patch_size, config=config)
        )
    return raw_fn(messages, image_patch_size=image_patch_size, config=config)


async def decode_multi_modal_info(messages, image_patch_size, config, raw_fn):
    """Drop-in for RLHFDataset.process_multi_modal_info.

    With every env var unset this is exactly `raw_fn(...)`, inline, as before.
    """
    _apply_thread_cap()

    if DECODE_CACHE_SIZE <= 0:
        return await _run_raw(raw_fn, messages, image_patch_size, config)

    key = _cache_key(messages)
    if key is None:
        return await _run_raw(raw_fn, messages, image_patch_size, config)

    lock = _get_lock()
    async with lock:
        if key in _cache:
            _stats["hit"] += 1
            return _cache[key]
        future = _inflight.get(key)
        owner = future is None
        if owner:
            future = asyncio.get_running_loop().create_future()
            _inflight[key] = future
            _stats["miss"] += 1

    # A non-owner waits on the owner's decode instead of starting its own. The
    # n rollouts of a prompt are dispatched concurrently, so a plain
    # check-then-decode cache would miss n times before the first result lands.
    if not owner:
        return await future

    try:
        result = await _run_raw(raw_fn, messages, image_patch_size, config)
    except BaseException as e:
        async with lock:
            _inflight.pop(key, None)
        if not future.done():
            future.set_exception(e)
        # Nobody may be awaiting this future; consume it so asyncio stays quiet.
        future.exception()
        raise

    async with lock:
        _cache[key] = result
        _cache_order.append(key)
        while len(_cache_order) > DECODE_CACHE_SIZE:
            _cache.pop(_cache_order.pop(0), None)
        _inflight.pop(key, None)
    if not future.done():
        future.set_result(result)
    return result


def decode_stats() -> dict:
    total = _stats["hit"] + _stats["miss"]
    return {
        "hit": _stats["hit"],
        "miss": _stats["miss"],
        "hit_rate": (_stats["hit"] / total) if total else 0.0,
        "cached": len(_cache),
    }
