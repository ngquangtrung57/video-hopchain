"""Entropy-aware token dropout logits processor (vLLM v1 API).

Two trigger modes are selected by ``trigger_mode``:

- ``"high"`` (default): perturb where ``max_prob > top_prob_threshold``.
- ``"band"``: perturb where ``tau_low < max_prob < tau_high``.

When triggered, with probability ``perturb_prob`` the top-``drop_top_k`` tokens
are masked so the sampler draws from the renormalised remainder. Applied only
to rollouts carrying ``SamplingParams.extra_args["rvrl_exploration"] = True``.

Configuration is global (per engine instance) and populated via
:func:`set_exploration_config` before vLLM engine construction by
``vllm_async_server.launch_server()``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import torch

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


logger = logging.getLogger(__file__)


_EXPLORATION_CFG: dict = {
    "enable": False,
    "trigger_mode": "high",        # "high" or "band"
    "top_prob_threshold": 0.9,     # used when trigger_mode == "high"
    "tau_low": 0.3,                # used when trigger_mode == "band"
    "tau_high": 0.7,               # used when trigger_mode == "band"
    "perturb_prob": 0.3,
    "drop_top_k": 1,
    "min_position": 8,
    "max_perturbations_per_seq": 32,
    "restrict_to_think_region": True,
    "think_open_token_id": 151667,
    "think_close_token_id": 151668,
    "seed": None,
    # Per-worker drop trace, written directly by the vLLM worker process.
    "drop_trace_dir": "",          # "" disables. One file per worker pid.
    "drop_trace_max_rows": 20000,  # per worker, then it stops recording

    "deterministic": False,
    "mask_from_loss": False,
    "prompt_exploration_prob": 1.0,
    # Scalar drop statistics (log-prob of the dropped token, probability mass
    # removed, forced log-prob penalty) are always accumulated. The expensive
    # extras below are opt-in:
    #   record_details -> per-drop rows (position, token id, log-probs, entropy)
    #                     kept in _DROPPED_DETAILS for the trainer to join
    #                     against the response.
    #   record_entropy -> full-vocab entropy at each dropped position.
    #   detail_topn    -> how many pre-drop candidates to keep per drop, so the
    #                     original probability of the token the sampler picked
    #                     instead stays recoverable. 0 disables.
    "record_details": False,
    "record_entropy": True,      # only consulted when record_details is True
    "detail_topn": 8,            # only consulted when record_details is True
    "detail_max_per_request": 512,
}


# Dropped-token positions keyed by the per-request UUID stamped from
# agent_loop. Populated inside ``apply`` when an explore request hits a
# trigger; consumed once by vllm_async_server at request completion via
# :func:`consume_dropped_positions`. Lives at module scope so it survives
# BatchUpdate moves and is reachable process-wide.
import os as _os_mod
_PID = _os_mod.getpid()
_DROPPED_POSITIONS_LOCK = threading.Lock()
_DROPPED_POSITIONS: dict[str, list[int]] = {}


# ---------------------------------------------------------------------------
# Cross-process transport for the loss mask.
#
# vLLM registers this logits processor by string path, so it is constructed in
# the spawned worker process while ``consume_dropped_positions`` runs in the
# server process; module globals are per-process and never meet. The worker
# buffers positions in memory and flushes one small node-local file per request
# when the request leaves the batch; the server reads and unlinks it. The
# directory comes from VERL_EXPLORATION_IPC_DIR.
_IPC_DIR_DEFAULT = "/tmp/rvrl_drop_pos"


def _ipc_dir() -> str:
    return _os_mod.environ.get("VERL_EXPLORATION_IPC_DIR", _IPC_DIR_DEFAULT)


def _ipc_path(req_uuid: str) -> str:
    safe = "".join(c for c in str(req_uuid) if c.isalnum() or c in "-_")
    return _os_mod.path.join(_ipc_dir(), safe)


def _ipc_publish(req_uuid: str, positions: list) -> bool:
    """Worker side. Write this request's drop positions where the server can
    read them. Atomic via write-then-rename so a partial file is never read."""
    if not req_uuid or not positions:
        return False
    try:
        d = _ipc_dir()
        _os_mod.makedirs(d, exist_ok=True)
        final = _ipc_path(req_uuid)
        tmp = final + f".{_PID}.tmp"
        with open(tmp, "w") as fh:
            fh.write(",".join(str(int(x)) for x in positions))
        _os_mod.replace(tmp, final)
        with _LP_COUNTERS_LOCK:
            _LP_COUNTERS["mask_ipc_published"] += 1
            _LP_COUNTERS["mask_ipc_positions"] += len(positions)
        return True
    except Exception:
        try:
            with _LP_COUNTERS_LOCK:
                _LP_COUNTERS["mask_ipc_publish_failed"] += 1
        except Exception:
            pass
        return False


def _ipc_consume(req_uuid: str) -> list:
    """Server side. Read and remove one request's positions. ``[]`` when absent."""
    if not req_uuid:
        return []
    path = _ipc_path(req_uuid)
    try:
        with open(path) as fh:
            raw = fh.read().strip()
    except FileNotFoundError:
        return []
    except Exception:
        return []
    finally:
        try:
            _os_mod.unlink(path)
        except Exception:
            pass
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


def ipc_sweep(max_age_s: float = 1800.0) -> int:
    """Remove position files older than ``max_age_s``.

    A parameter sync aborts in-flight requests, whose files the server then
    never consumes; without this sweep they accumulate.
    """
    import time as _time
    d = _ipc_dir()
    now = _time.time()
    n = 0
    try:
        for name in _os_mod.listdir(d):
            fp = _os_mod.path.join(d, name)
            try:
                if now - _os_mod.path.getmtime(fp) > max_age_s:
                    _os_mod.unlink(fp)
                    n += 1
            except Exception:
                continue
    except Exception:
        return 0
    return n


_DROPPED_DETAILS_LOCK = threading.Lock()
_DROPPED_DETAILS: dict[str, dict[str, list]] = {}
# Hard bound on orphaned detail records; eviction happens in ``apply``.
_DETAILS_TABLE_MAX = 8192


def consume_dropped_details(req_uuid: str) -> dict[str, list]:
    """Pop and return the per-drop analysis record for ``req_uuid``.

    A separate table from :data:`_DROPPED_POSITIONS`, which feeds the loss-mask
    path; everything here is analysis-only.

    Returns ``{}`` when detail recording is off or nothing was recorded. The
    returned dict holds equal-length parallel lists:

    ``pos``       response position of the drop (0-indexed, matches
                  ``_DROPPED_POSITIONS``)
    ``tok``       token id that was dropped (the pre-drop argmax)
    ``logp``      log-probability the model assigned to that dropped token
    ``ru_logp``   log-probability of the best SURVIVING token, i.e. what the
                  sampler is pushed onto
    ``mass``      total probability mass removed (sum over the k dropped)
    ``ent``       full-vocab entropy at that position (``record_entropy``)
    ``topn_tok`` / ``topn_logp``
                  pre-drop top-N candidates. The log-probs vLLM returns for an
                  exploration rollout are computed from the masked logits and
                  are therefore renormalised; these columns keep the original
                  policy's log-probability recoverable.
    """
    if not req_uuid:
        return {}
    with _DROPPED_DETAILS_LOCK:
        return _DROPPED_DETAILS.pop(req_uuid, {})


def consume_dropped_positions(req_uuid: str) -> list[int]:
    """Pop and return the list of dropped token positions for ``req_uuid``.

    Called once by ``vllm_async_server`` when finalizing an exploration
    request. If the LP never recorded any drops for this UUID (e.g. the
    request had no trigger), returns an empty list. Idempotent: a second
    call returns ``[]``.
    """
    if not req_uuid:
        return []
    # In-process table first: covers the case where the LP runs in THIS process
    # (unit tests, and any vLLM mode that does not spawn workers).
    with _DROPPED_POSITIONS_LOCK:
        local = _DROPPED_POSITIONS.pop(req_uuid, [])
    if local:
        return local
    # Cross-process path. The worker flushes the file when the request leaves
    # the batch, in the same engine step that produces the final output, so it
    # is normally already present; retry briefly in case this call gets in
    # first.
    import time as _time
    for _attempt in range(5):
        got = _ipc_consume(req_uuid)
        if got:
            return got
        _time.sleep(0.004)
    return []


# Per-token entropy histogram edges, in nats. Dense below 1.0, where most of
# the token mass sits; the last bucket is the open right tail.
_ENT_HIST_EDGES = (
    0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00,
    1.50, 2.00, 3.00, 4.00, 6.00, 8.00, 12.0,
)
_ENT_HIST_NBIN = len(_ENT_HIST_EDGES) + 1


# Module-level counters aggregated across all LP instances in this process.
# Reset by :func:`flush_lp_counters`. Updated under :data:`_LP_COUNTERS_LOCK`.
_LP_COUNTERS_LOCK = threading.Lock()
_LP_COUNTERS: dict = {
    # positions where active explore seq was scheduled (gate considered)
    "active_positions": 0,
    # positions skipped because pos < min_position
    "skipped_min_pos": 0,
    # positions skipped because outside <think> region (or top1 is think tag)
    "skipped_outside_think": 0,
    # positions where max_prob > tau (post-region/min_pos gate)
    "triggered": 0,
    # positions actually perturbed (subset of triggered, gated by perturb_prob)
    "perturbed": 0,
    # finished sequences observed (via removed/added events)
    "finished_seqs": 0,
    # Loss-mask transport health. "perturbed" large while "mask_ipc_published"
    # is 0 means the drop mask never reached the server and the forced tokens
    # are being trained on.
    "mask_ipc_published": 0,
    "mask_ipc_positions": 0,
    "mask_ipc_publish_failed": 0,
    # Token-class filter: positions that triggered on confidence but were
    # rejected because the top-1 token is whitespace, pure punctuation, a bare
    # digit, a special token or a mid-word subword.
    "skipped_undroppable_token": 0,
    # sequences that hit the per-seq perturbation cap before finishing
    "cap_hit_seqs": 0,
    # sum of perturbation counts at finish-time across finished seqs
    "perturb_count_sum": 0,
    # rolling sample of "active explore seqs" count per apply() invocation
    "active_seq_count_sum": 0,
    "active_seq_count_steps": 0,
    # ---- Drop statistics. Sums over PERTURBED positions only, so every mean
    # below divides by "perturbed".
    # sum of log p(dropped top-1) -- mean is the average log-prob we destroyed
    "dropped_logp_sum": 0.0,
    # sum of p(dropped top-1), for a mean in probability space
    "dropped_prob_sum": 0.0,
    # sum of total probability mass removed (all drop_top_k tokens)
    "dropped_mass_sum": 0.0,
    # sum of log p(best surviving token) -- where the sampler is pushed
    "runner_up_logp_sum": 0.0,
    # sum of [log p(dropped top-1) - log p(best survivor)]: the forced log-prob
    # penalty.
    "forced_logp_delta_sum": 0.0,
    # sum of full-vocab entropy at dropped positions (record_entropy only)
    "dropped_entropy_sum": 0.0,
    "dropped_entropy_n": 0,
    # sum of top-1 prob over TRIGGERED positions
    "trigger_top1_prob_sum": 0.0,
    # 10-bucket histogram of top-1 prob at DROP time, bucket i = [i/10,(i+1)/10)
    "drop_top1_hist": [0] * 10,
    # ---- Entropy instrumentation ----------------------------------------
    # Observation only: never gates a drop, never touches a gradient. Split by
    # wave, so comparing perturbed against clean rollouts on the same prompts
    # is a subtraction of two counters.
    #
    # "explore" = wave-2 rows carrying rvrl_exploration (the perturbed ones).
    # "anchor"  = wave-1 rows carrying rvrl_measure (clean controls).
    "ent_tok_sum_explore": 0.0,
    "ent_tok_sq_explore": 0.0,
    "ent_tok_n_explore": 0,
    "ent_tok_sum_anchor": 0.0,
    "ent_tok_sq_anchor": 0.0,
    "ent_tok_n_anchor": 0,
    # Per-token entropy histogram over _ENT_HIST_EDGES, which resolves the
    # right tail that a mean cannot.
    "ent_hist_explore": [0] * _ENT_HIST_NBIN,
    "ent_hist_anchor": [0] * _ENT_HIST_NBIN,
    # ---- at DROP sites ---------------------------------------------------
    # H_post is the entropy of the renormalised distribution the sampler draws
    # from once the mask is applied. With drop_top_k=1 it is exact in closed
    # form from H_orig and p_top1, so it costs no extra GPU work:
    #     H_post = (H_orig + p1*ln p1) / (1 - p1) + ln(1 - p1)
    "drop_H_post_sum": 0.0,
    "drop_H_post_sq_sum": 0.0,
    "drop_H_post_n": 0,
    # H_post - H_orig. Positive means the mask opened the position up; negative
    # means the remainder after the drop is close to deterministic.
    "drop_dH_sum": 0.0,
    # p2/(1-p1): the runner-up's share of all non-top-1 mass. ~1 is a binary
    # fork, ~0 a diffuse tail.
    "drop_resid_conc_sum": 0.0,
    # Population split of the above.
    "drop_H_post_near_forced": 0,   # H_post < 0.35 nats
    "drop_H_post_open": 0,          # H_post > 1.00 nats
    # ---- post-drop window ------------------------------------------------
    # Entropy over the W tokens following a drop.
    "post_drop_ent_sum": 0.0,
    "post_drop_ent_n": 0,
}


def set_exploration_config(cfg: dict) -> None:
    """Override module-level exploration config.

    Called by verl's vLLM launcher before the engine is constructed. The
    logits processor instance reads from this dict in ``__init__``.
    """
    global _EXPLORATION_CFG
    _EXPLORATION_CFG.update(cfg)


def merge_lp_counters(agg: dict, snap: dict) -> dict:
    """Element-wise sum ``snap`` into ``agg``, in place, returning ``agg``.

    Counter snapshots are summed at three places (vLLM worker -> server actor,
    server -> rollouter, rollouter/servers -> trainer); all three route through
    here so that list-valued counters (the histograms) are summed per bucket
    rather than raising.
    """
    if not isinstance(snap, dict):
        return agg
    for k, v in snap.items():
        if v is None:
            continue
        if isinstance(v, list):
            cur = agg.get(k)
            if not isinstance(cur, list):
                agg[k] = list(v)
            else:
                if len(cur) < len(v):
                    cur.extend([0] * (len(v) - len(cur)))
                for i, x in enumerate(v):
                    cur[i] += x
        else:
            cur = agg.get(k, 0)
            if isinstance(cur, list):
                continue  # type flip between snapshots; keep the list
            agg[k] = cur + v
    return agg


def flush_lp_counters() -> dict:
    """Atomically snapshot counters and reset to zero.

    Returns the previous counter values. Safe to call from any thread
    (including a Ray remote method). When the LP never fired, returns
    a dict of zero counters.
    """
    with _LP_COUNTERS_LOCK:
        snap = {k: (list(v) if isinstance(v, list) else v) for k, v in _LP_COUNTERS.items()}
        for k, v in _LP_COUNTERS.items():
            # Type-preserving reset: a blanket ``= 0`` would turn a histogram
            # list into an int and every later ``hist[b] += 1`` would raise.
            if isinstance(v, list):
                _LP_COUNTERS[k] = [0] * len(v)
            elif isinstance(v, float):
                _LP_COUNTERS[k] = 0.0
            else:
                _LP_COUNTERS[k] = 0
    return snap


class EntropyDropoutLP(LogitsProcessor):
    """vLLM v1 logits processor for controlled per-token exploration."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        # vLLM workers are spawned (not forked) inside a Ray actor, so the
        # in-memory ``_EXPLORATION_CFG`` set by the launcher in the parent
        # process is back at its defaults here. Recover it from the env var
        # ``vllm_async_server`` writes before engine init.
        import json as _json
        import os as _os
        _env_cfg = _os.environ.get("VERL_EXPLORATION_CFG")
        if _env_cfg and not _EXPLORATION_CFG["enable"]:
            try:
                set_exploration_config(_json.loads(_env_cfg))
            except Exception:
                pass
        cfg = _EXPLORATION_CFG

        self.enable: bool = bool(cfg["enable"])
        self.trigger_mode: str = str(cfg.get("trigger_mode", "high"))
        self.tau: float = float(cfg["top_prob_threshold"])
        self.tau_low: float = float(cfg.get("tau_low", 0.3))
        self.tau_high: float = float(cfg.get("tau_high", 0.7))
        self.p: float = float(cfg["perturb_prob"])
        self.drop_top_k: int = int(cfg["drop_top_k"])
        self.min_pos: int = int(cfg["min_position"])
        self.cap: int = int(cfg["max_perturbations_per_seq"])
        self.restrict_think: bool = bool(cfg["restrict_to_think_region"])
        self.think_open: int = int(cfg["think_open_token_id"])
        self.think_close: int = int(cfg["think_close_token_id"])
        # Deterministic drop: skip the per-token Bernoulli gate when True.
        self.deterministic: bool = bool(cfg.get("deterministic", False))
        # Whether the PROMPT already carries the opening <think> tag. Defaults
        # to the live VERL_THINK_PREFILL so the two cannot disagree when a
        # recipe omits the config key.
        self.think_prefilled: bool = bool(
            cfg.get("think_prefilled", _os.environ.get("VERL_THINK_PREFILL", "0") == "1")
        )
        # Token-class filter: restrict drops to top-1 tokens that carry a real
        # lexical alternative, excluding whitespace, punctuation, bare digits
        # and special tokens.
        self.skip_special_tokens: bool = bool(cfg.get("skip_special_tokens", True))
        self.skip_whitespace_punct: bool = bool(cfg.get("skip_whitespace_punct", False))
        self.skip_digit_tokens: bool = bool(cfg.get("skip_digit_tokens", False))
        # Stricter, off by default: require the dropped token to start a word.
        self.skip_subword_continuation: bool = bool(
            cfg.get("skip_subword_continuation", False)
        )
        self._filter_on: bool = (
            self.skip_special_tokens
            or self.skip_whitespace_punct
            or self.skip_digit_tokens
            or self.skip_subword_continuation
        )
        # Built lazily on the first apply(), when the vocab width is known from
        # the logits themselves rather than from config.
        self._droppable = None
        self._droppable_failed: bool = False
        self._vllm_config = vllm_config
        self.record_details: bool = bool(cfg.get("record_details", False))
        self.record_entropy: bool = bool(cfg.get("record_entropy", True))
        self.detail_topn: int = int(cfg.get("detail_topn", 8))
        self.detail_cap: int = int(cfg.get("detail_max_per_request", 512))
        seed = cfg.get("seed")

        self.device = device
        self.max_reqs: int = int(vllm_config.scheduler_config.max_num_seqs)

        self.active = torch.zeros(self.max_reqs, dtype=torch.bool, device=device)
        self.inside_think = torch.zeros(self.max_reqs, dtype=torch.bool, device=device)
        self.perturb_count = torch.zeros(self.max_reqs, dtype=torch.int32, device=device)
        self.pos_count = torch.zeros(self.max_reqs, dtype=torch.int32, device=device)

        # ---- entropy instrumentation state ------------------------------
        # `measure` is a superset of `active`: wave-1 anchors are measured but
        # never perturbed, which is what makes the wave-1 vs wave-2 entropy
        # comparison possible.
        self.measure_entropy: bool = bool(cfg.get("measure_entropy", False))
        self._ent_stride: int = max(1, int(cfg.get("entropy_measure_stride", 1)))
        self._post_win: int = max(0, int(cfg.get("entropy_post_drop_window", 16)))
        self._seq_trace_max: int = int(cfg.get("entropy_seq_trace_max_rows", 40000))
        self._seq_rows: int = 0
        self._meas_n: int = 0
        self.measure = torch.zeros(self.max_reqs, dtype=torch.bool, device=device)
        # Per-sequence running entropy, folded into the wave counters on removal.
        self.ent_sum = torch.zeros(self.max_reqs, dtype=torch.float32, device=device)
        self.ent_sq = torch.zeros(self.max_reqs, dtype=torch.float32, device=device)
        self.ent_n = torch.zeros(self.max_reqs, dtype=torch.int32, device=device)
        # Countdown of tokens still inside a post-drop observation window.
        self.post_left = torch.zeros(self.max_reqs, dtype=torch.int32, device=device)
        self._ent_edges = torch.tensor(
            _ENT_HIST_EDGES, dtype=torch.float32, device=device
        )

        self.output_tok_refs: dict[int, list[int]] = {}
        # slot -> per-request UUID (stamped by agent_loop via extra_args), so
        # ``apply`` records dropped positions under a stable key rather than
        # the volatile slot index.
        self.idx_to_req: dict[int, str] = {}
        # slot -> the drop just made, awaiting its replacement token. The
        # sampler runs after this processor returns, but output_tok_refs[idx]
        # is the live list vLLM appends to, so on the next apply() the token at
        # the recorded index is the replacement.
        self._pending: dict[int, dict] = {}
        # slot -> uuid for entropy measurement only. Held apart from
        # idx_to_req, which feeds the drop-mask IPC that decides which tokens
        # leave the loss; wave-1 anchor uuids must never reach it.
        self.idx_to_meas_req: dict[int, str] = {}
        self._trace_dir: str = str(cfg.get("drop_trace_dir", "") or "")
        self._trace_max: int = int(cfg.get("drop_trace_max_rows", 20000))
        self._trace_rows: int = 0
        # Counters are also snapshotted into the trace file periodically, so
        # they reach disk independently of the Ray metric path.
        self._apply_n: int = 0
        self._counter_every: int = int(cfg.get("counter_snapshot_every", 2000))
        self._trace_buf: list = []
        self._trace_fh = None
        self._trace_warned: bool = False

        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

    def is_argmax_invariant(self) -> bool:
        return False

    def _build_droppable(self, vocab_size: int, device) -> None:
        """Vocabulary-wide bool mask: may this token id be dropped?

        Built once per worker. On any failure the filter is disabled and the
        reason is logged, so an unbuildable filter is never mistaken for a
        filter that found nothing to skip.
        """
        self._droppable_failed = True          # until proven otherwise
        try:
            import string as _string
            tok = None
            mc = getattr(self._vllm_config, "model_config", None)
            name = getattr(mc, "tokenizer", None) or getattr(mc, "model", None)
            if name:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(name), trust_remote_code=True)
            if tok is None:
                raise RuntimeError("no tokenizer on vllm_config.model_config")

            pieces = tok.batch_decode([[i] for i in range(vocab_size)])
            special = set(getattr(tok, "all_special_ids", []) or [])
            # Qwen3-VL keeps control tokens (<|vision_start|>, <|im_end|>, the
            # think tags) outside all_special_ids, so anything decoding to a
            # <|...|> control string is treated as special too.
            punct = set(_string.punctuation)

            mask = torch.ones(vocab_size, dtype=torch.bool)
            n_skip = 0
            for i, raw in enumerate(pieces):
                drop_ok = True
                body = raw.strip()
                if self.skip_special_tokens and (
                    i in special or (raw.startswith("<|") and raw.endswith("|>"))
                ):
                    drop_ok = False
                elif body == "":
                    # whitespace-only, including the bare space token
                    drop_ok = False
                elif self.skip_whitespace_punct and all(c in punct for c in body):
                    drop_ok = False
                elif self.skip_digit_tokens and all(
                    (c.isdigit() or c in ".,") for c in body
                ):
                    # a corrupted digit changes the answer rather than the
                    # reasoning path
                    drop_ok = False
                elif self.skip_subword_continuation and not raw[:1].isspace():
                    drop_ok = False
                elif not any(c.isalpha() for c in body):
                    # nothing word-like left to substitute
                    drop_ok = False
                if not drop_ok:
                    mask[i] = False
                    n_skip += 1

            self._droppable = mask.to(device)
            self._droppable_failed = False
            logger.info(
                "droppable token filter built: vocab=%d blocked=%d", vocab_size, n_skip
            )
        except Exception as e:
            self._droppable = None
            logger.warning(
                "droppable token filter disabled (%s: %s); structural tokens "
                "(whitespace, punctuation, digits) remain eligible for dropping",
                type(e).__name__,
                e,
            )

    def _trace_write(self, rec: dict, exempt_from_cap: bool = False) -> None:
        """Append one record. Worker-local, never raises.

        ``exempt_from_cap`` is for the periodic counter snapshots: those rows are
        neither blocked by nor counted against ``drop_trace_max_rows``, so a full
        drop trace cannot silence them.
        """
        if not self._trace_dir:
            return
        if not exempt_from_cap and self._trace_rows >= self._trace_max:
            return
        try:
            if self._trace_fh is None:
                import os as _os, json as _json  # noqa: F401
                _os.makedirs(self._trace_dir, exist_ok=True)
                path = _os.path.join(self._trace_dir, f"drops_pid{_os.getpid()}.jsonl")
                self._trace_fh = open(path, "a", buffering=1)
                logger.info("drop trace -> %s", path)
            import json as _json
            self._trace_fh.write(_json.dumps(rec) + "\n")
            if exempt_from_cap:
                return
            self._trace_rows += 1
            if self._trace_rows >= self._trace_max and not self._trace_warned:
                self._trace_warned = True
                logger.warning(
                    "drop trace capped at %d rows; this worker stops recording "
                    "drop rows (rvrl/* counters are unaffected)",
                    self._trace_max,
                )
        except Exception:
            # Tracing must never be able to break generation.
            self._trace_dir = ""

    def _resolve_pending(self) -> None:
        """Fill in the token the sampler actually chose after each drop.

        Called at the TOP of apply(), i.e. after the sampler has run for the
        step in which the drop was made. ``output_tok_refs[idx]`` is the live
        list vLLM appends to, so the entry at the recorded length is the
        replacement. Anything still unresolved is left pending for a later call.
        """
        if not self._pending:
            return
        for idx in list(self._pending.keys()):
            rec = self._pending[idx]
            ref = self.output_tok_refs.get(idx)
            if ref is None:
                self._pending.pop(idx, None)
                continue
            n = rec.pop("_n_out", None)
            if n is None or len(ref) <= n:
                rec["_n_out"] = n
                continue
            rec["replacement_token_id"] = int(ref[n])
            rec["resolved"] = True
            self._pending.pop(idx, None)
            self._trace_write(rec)

    def _write_seq_entropy(self, idx: int) -> None:
        """Emit one per-rollout entropy row as the sequence leaves the batch.

        The row carries the rollout's uuid, so entropy can be joined to that
        rollout's own reward. Written with ``exempt_from_cap`` and counted
        against its own budget, so a saturated drop trace cannot starve it.
        """
        n = int(self.ent_n[idx].item())
        if n <= 0:
            return
        s = float(self.ent_sum[idx].item())
        q = float(self.ent_sq[idx].item())
        mean = s / n
        var = max(q / n - mean * mean, 0.0)
        if self._trace_dir and self._seq_rows < self._seq_trace_max:
            self._seq_rows += 1
            self._trace_write(
                {
                    "kind": "seq_entropy",
                    "pid": _PID,
                    "uuid": self.idx_to_meas_req.get(idx, ""),
                    "explore": bool(self.active[idx].item()),
                    "n_tok": n,
                    "ent_mean": round(mean, 5),
                    "ent_std": round(var ** 0.5, 5),
                    "n_drops": int(self.perturb_count[idx].item()),
                },
                exempt_from_cap=True,
            )
        self.ent_sum[idx] = 0.0
        self.ent_sq[idx] = 0.0
        self.ent_n[idx] = 0
        self.post_left[idx] = 0
        self.measure[idx] = False
        self.idx_to_meas_req.pop(idx, None)

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            return

        # On removal of an active explore seq, fold its final perturb_count
        # into module-level counters so we can compute per-seq stats.
        finished_active = 0
        finished_perturb_sum = 0
        finished_cap_hit = 0
        for idx in batch_update.removed:
            if bool(self.active[idx].item()):
                finished_active += 1
                pc = int(self.perturb_count[idx].item())
                finished_perturb_sum += pc
                if pc >= self.cap:
                    finished_cap_hit += 1
            # Last chance to resolve a pending drop for this slot: the live
            # token list is about to be released. If the drop was the last step
            # the replacement may exist; otherwise the row is recorded
            # unresolved so the trace never loses a drop.
            if idx in self._pending:
                try:
                    _rec = self._pending.pop(idx)
                    _ref = self.output_tok_refs.get(idx)
                    _n = _rec.pop("_n_out", None)
                    if _ref is not None and _n is not None and len(_ref) > _n:
                        _rec["replacement_token_id"] = int(_ref[_n])
                        _rec["resolved"] = True
                    self._trace_write(_rec)
                except Exception:
                    pass
            # The sequence is finished, so its entropy accumulators are final.
            # Done before active/measure are cleared, because the row records
            # which wave the rollout belonged to.
            if self.measure_entropy:
                try:
                    if bool(self.measure[idx].item()):
                        self._write_seq_entropy(idx)
                except Exception:
                    pass
            self.active[idx] = False
            self.inside_think[idx] = False
            self.output_tok_refs.pop(idx, None)
            # The request is leaving the batch, so its position list is final.
            # Publish it where the server process can read it; the server
            # finalizes the request immediately after this engine step, and
            # flushing once per request keeps this to one small write.
            _uuid_done = self.idx_to_req.get(idx)
            if _uuid_done:
                try:
                    with _DROPPED_POSITIONS_LOCK:
                        _pos_done = _DROPPED_POSITIONS.pop(_uuid_done, [])
                    if _pos_done:
                        _ipc_publish(_uuid_done, _pos_done)
                except Exception:
                    pass
            # Drop the slot->uuid binding; the dropped-positions list in
            # _DROPPED_POSITIONS stays alive until the consumer pops it.
            self.idx_to_req.pop(idx, None)
        if finished_active:
            try:
                with _LP_COUNTERS_LOCK:
                    _LP_COUNTERS["finished_seqs"] += finished_active
                    _LP_COUNTERS["perturb_count_sum"] += finished_perturb_sum
                    _LP_COUNTERS["cap_hit_seqs"] += finished_cap_hit
            except Exception:
                pass

        for idx, params, _prompt_ids, output_tok_ids in batch_update.added:
            extra = params.extra_args
            is_explore = bool(
                self.enable
                and extra is not None
                and extra.get("rvrl_exploration", False)
            )

            # Measurement tracks a superset of the perturbed rows. Wave-1
            # anchors carry `rvrl_measure` and never `rvrl_exploration`, so
            # they are measured and never dropped. A separate uuid map is used
            # because reusing idx_to_req would put anchor uuids on the
            # drop-mask IPC path, which decides what leaves the loss.
            if self.measure_entropy and extra is not None:
                _is_meas = bool(
                    extra.get("rvrl_measure", False) or extra.get("rvrl_exploration", False)
                )
                self.measure[idx] = _is_meas
                self.ent_sum[idx] = 0.0
                self.ent_sq[idx] = 0.0
                self.ent_n[idx] = 0
                self.post_left[idx] = 0
                if _is_meas:
                    _mu = extra.get("rvrl_req_uuid")
                    if _mu:
                        self.idx_to_meas_req[idx] = str(_mu)
                    else:
                        self.idx_to_meas_req.pop(idx, None)
                else:
                    self.idx_to_meas_req.pop(idx, None)

            if is_explore:
                self.active[idx] = True
                # Generation starts inside the think region when the prompt
                # was prefilled with the opening tag (VERL_THINK_PREFILL=1). In
                # that case the model never emits think_open, so the watcher in
                # apply() would never flip this True and the region gate would
                # reject every candidate. The think_close watcher still applies,
                # so drops stop at </think> and never reach the answer.
                self.inside_think[idx] = self.think_prefilled
                self.perturb_count[idx] = 0
                self.pos_count[idx] = 0
                self.output_tok_refs[idx] = output_tok_ids
                self._pending.pop(idx, None)   # never inherit a stale drop
                # Bind slot -> per-request UUID for the side-channel table.
                req_uuid = extra.get("rvrl_req_uuid") if extra is not None else None
                if req_uuid:
                    self.idx_to_req[idx] = str(req_uuid)
                else:
                    self.idx_to_req.pop(idx, None)
            else:
                self.active[idx] = False
                self.inside_think[idx] = False
                self.output_tok_refs.pop(idx, None)
                self.idx_to_req.pop(idx, None)
                self._pending.pop(idx, None)

        for a, b, direct in batch_update.moved:
            if direct == MoveDirectionality.SWAP:
                # The entropy accumulators are per-slot, so they ride along with
                # every slot move exactly like perturb_count; otherwise one
                # rollout's entropy is attributed to another whenever vLLM
                # reshuffles the batch.
                for t in (
                    self.active, self.inside_think, self.perturb_count, self.pos_count,
                    self.measure, self.ent_sum, self.ent_sq, self.ent_n, self.post_left,
                ):
                    tmp = t[a].clone()
                    t[a] = t[b]
                    t[b] = tmp
                ma = self.idx_to_meas_req.pop(a, None)
                mb = self.idx_to_meas_req.pop(b, None)
                if ma is not None:
                    self.idx_to_meas_req[b] = ma
                if mb is not None:
                    self.idx_to_meas_req[a] = mb
                ra = self.output_tok_refs.pop(a, None)
                rb = self.output_tok_refs.pop(b, None)
                if ra is not None:
                    self.output_tok_refs[b] = ra
                if rb is not None:
                    self.output_tok_refs[a] = rb
                # mirror idx_to_req swap
                ua = self.idx_to_req.pop(a, None)
                ub = self.idx_to_req.pop(b, None)
                if ua is not None:
                    self.idx_to_req[b] = ua
                if ub is not None:
                    self.idx_to_req[a] = ub
                # mirror _pending swap
                pa = self._pending.pop(a, None)
                pb = self._pending.pop(b, None)
                if pa is not None:
                    self._pending[b] = pa
                if pb is not None:
                    self._pending[a] = pb
            else:
                self.active[b] = self.active[a]
                self.inside_think[b] = self.inside_think[a]
                self.perturb_count[b] = self.perturb_count[a]
                self.pos_count[b] = self.pos_count[a]
                # mirror the entropy accumulators on the unidirectional move
                self.measure[b] = self.measure[a]
                self.ent_sum[b] = self.ent_sum[a]
                self.ent_sq[b] = self.ent_sq[a]
                self.ent_n[b] = self.ent_n[a]
                self.post_left[b] = self.post_left[a]
                self.measure[a] = False
                self.ent_sum[a] = 0.0
                self.ent_sq[a] = 0.0
                self.ent_n[a] = 0
                self.post_left[a] = 0
                _ma = self.idx_to_meas_req.pop(a, None)
                if _ma is not None:
                    self.idx_to_meas_req[b] = _ma
                else:
                    self.idx_to_meas_req.pop(b, None)
                self.active[a] = False
                self.inside_think[a] = False
                ra = self.output_tok_refs.pop(a, None)
                if ra is not None:
                    self.output_tok_refs[b] = ra
                else:
                    self.output_tok_refs.pop(b, None)
                # mirror idx_to_req unidirectional move
                ua = self.idx_to_req.pop(a, None)
                if ua is not None:
                    self.idx_to_req[b] = ua
                else:
                    self.idx_to_req.pop(b, None)
                # mirror _pending unidirectional move
                pa = self._pending.pop(a, None)
                if pa is not None:
                    self._pending[b] = pa
                else:
                    self._pending.pop(b, None)

    def _measure_entropy(self, logits: torch.Tensor, B: int):
        """Accumulate per-token policy entropy for every tracked row.

        Returns ``(probs, ent)`` so ``apply`` reuses both instead of paying for
        a second softmax and a second entropy pass at the drop sites.

        Entropy uses the stable identity ``H = logsumexp(z) - sum(softmax(z)*z)``,
        the same form as ``verl.utils.torch_functional.entropy_from_logits``.
        Scalars are packed into one stacked tensor and both histograms into one
        bincount, so the call costs two host syncs; unmeasured rows are routed
        to a discard bucket instead of being indexed out.

        Measured before any mask is applied, so a wave-2 row reports the entropy
        the policy had at that position. The post-mask quantity is exact in
        closed form at the drop site and needs no second pass.
        """
        probs = torch.softmax(logits, dim=-1)
        ent = (torch.logsumexp(logits, dim=-1) - (probs * logits).sum(dim=-1))
        ent = ent.float().clamp_min_(0.0)

        meas = self.measure[:B]
        expl = meas & self.active[:B]          # wave 2: perturbed
        anch = meas & (~self.active[:B])       # wave 1: clean control

        m_e = expl.to(ent.dtype)
        m_a = anch.to(ent.dtype)
        ent_sq = ent * ent

        # Per-sequence running totals; folded into the wave counters on removal
        # and written per-rollout so entropy can be joined to that rollout's
        # own reward.
        m_all = meas.to(ent.dtype)
        self.ent_sum[:B] += ent * m_all
        self.ent_sq[:B] += ent_sq * m_all
        self.ent_n[:B] += meas.to(torch.int32)

        # Post-drop observation window: entropy over the W tokens following a
        # drop.
        if self._post_win > 0:
            inwin = self.post_left[:B] > 0
            m_w = inwin.to(ent.dtype)
            self.post_left[:B] = (
                self.post_left[:B] - inwin.to(torch.int32)
            ).clamp_min_(0)
        else:
            m_w = torch.zeros_like(m_e)

        stats = torch.stack([
            (ent * m_e).sum(), (ent_sq * m_e).sum(), m_e.sum(),
            (ent * m_a).sum(), (ent_sq * m_a).sum(), m_a.sum(),
            (ent * m_w).sum(), m_w.sum(),
        ])

        # One bincount for both waves: shift explore rows into the upper half of
        # the bucket range, and send unmeasured rows to a trailing discard bin.
        b = torch.bucketize(ent, self._ent_edges)
        b = b + expl.to(torch.int64) * _ENT_HIST_NBIN
        b = torch.where(meas, b, torch.full_like(b, 2 * _ENT_HIST_NBIN))
        hist = torch.bincount(b, minlength=2 * _ENT_HIST_NBIN + 1)

        sv = stats.detach().to("cpu").tolist()          # sync 1
        hv = hist.detach().to("cpu").tolist()           # sync 2

        with _LP_COUNTERS_LOCK:
            _LP_COUNTERS["ent_tok_sum_explore"] += sv[0]
            _LP_COUNTERS["ent_tok_sq_explore"] += sv[1]
            _LP_COUNTERS["ent_tok_n_explore"] += int(sv[2])
            _LP_COUNTERS["ent_tok_sum_anchor"] += sv[3]
            _LP_COUNTERS["ent_tok_sq_anchor"] += sv[4]
            _LP_COUNTERS["ent_tok_n_anchor"] += int(sv[5])
            _LP_COUNTERS["post_drop_ent_sum"] += sv[6]
            _LP_COUNTERS["post_drop_ent_n"] += int(sv[7])
            _ha = _LP_COUNTERS["ent_hist_anchor"]
            _he = _LP_COUNTERS["ent_hist_explore"]
            for _b in range(_ENT_HIST_NBIN):
                _ha[_b] += int(hv[_b])
                _he[_b] += int(hv[_ENT_HIST_NBIN + _b])
        return probs, ent

    def apply(self, logits: torch.Tensor) -> torch.Tensor:

        # The sampler has run for the step in which any pending drop was made,
        # so the replacement token is readable. Done first, before any early
        # return below, or drops made on the last active step never get their
        # replacement.
        if self._pending:
            try:
                self._resolve_pending()
            except Exception:
                pass
        B = logits.size(0)

        # Entropy measurement runs before the n_active early return: wave-1
        # anchors are by construction not active, so returning first would
        # leave the explore wave with no control arm to compare against.
        probs = None
        ent_all = None
        if self.measure_entropy:
            self._meas_n += 1
            if self._meas_n % self._ent_stride == 0:
                try:
                    probs, ent_all = self._measure_entropy(logits, B)
                except Exception:
                    # Instrumentation must never be able to break generation.
                    probs, ent_all = None, None

        active_slice = self.active[:B]
        n_active = int(active_slice.sum().item())
        if n_active == 0:
            return logits

        # accumulate active-seq sample (per scheduling step)
        try:
            with _LP_COUNTERS_LOCK:
                _LP_COUNTERS["active_seq_count_sum"] += n_active
                _LP_COUNTERS["active_seq_count_steps"] += 1
                _LP_COUNTERS["active_positions"] += n_active
        except Exception:
            pass

        # Periodic counter snapshot into the trace file.
        self._apply_n += 1
        if self._trace_dir and self._counter_every > 0 and (
            self._apply_n % self._counter_every == 0
        ):
            try:
                import copy as _copy
                with _LP_COUNTERS_LOCK:
                    _snap = _copy.deepcopy(_LP_COUNTERS)
                _snap["kind"] = "counters"
                _snap["apply_n"] = self._apply_n
                _snap["pid"] = _PID
                # Written through the same handle as the drop rows, but neither
                # counted against drop_trace_max_rows nor silenced by it.
                self._trace_write(_snap, exempt_from_cap=True)
            except Exception:
                pass

        if self.restrict_think and self.output_tok_refs:
            for idx, toks in self.output_tok_refs.items():
                if idx >= B or not toks:
                    continue
                last = toks[-1]
                if last == self.think_open:
                    self.inside_think[idx] = True
                elif last == self.think_close:
                    self.inside_think[idx] = False

        self.pos_count[:B] = torch.where(
            active_slice,
            self.pos_count[:B] + 1,
            self.pos_count[:B],
        )

        # tally early-skip reasons (positions where active but masked out before LP probe)
        skipped_min_pos = int(
            (active_slice & (self.pos_count[:B] < self.min_pos)).sum().item()
        )
        if self.restrict_think:
            region_ok = self.inside_think[:B]
            skipped_outside_think = int(
                (active_slice & (~region_ok) & (self.pos_count[:B] >= self.min_pos)).sum().item()
            )
        else:
            region_ok = torch.ones_like(active_slice)
            skipped_outside_think = 0
        try:
            with _LP_COUNTERS_LOCK:
                _LP_COUNTERS["skipped_min_pos"] += skipped_min_pos
                _LP_COUNTERS["skipped_outside_think"] += skipped_outside_think
        except Exception:
            pass

        gate = (
            active_slice
            & region_ok
            & (self.pos_count[:B] >= self.min_pos)
            & (self.perturb_count[:B] < self.cap)
        )
        if not gate.any():
            return logits

        # Reuse the softmax already computed by the entropy measurement.
        if probs is None:
            probs = torch.softmax(logits, dim=-1)
        # Take ONE more than we drop: column ``drop_top_k`` is the best token
        # that SURVIVES the mask, i.e. where the sampler gets pushed. Needed for
        # the forced log-prob penalty. Masking below still slices [:drop_top_k],
        # so widening the topk changes nothing about which tokens are dropped.
        _k_probe = min(self.drop_top_k + 1, probs.size(-1))
        top_probs, top_idx = torch.topk(probs, _k_probe, dim=-1)

        top1_prob = top_probs[:, 0]
        if self.trigger_mode == "band":
            # Medium-confidence trigger: fire between the two band edges.
            trigger = (top1_prob > self.tau_low) & (top1_prob < self.tau_high) & gate
        else:  # "high"
            trigger = (top1_prob > self.tau) & gate

        if self.restrict_think:
            top1 = top_idx[:, 0]
            trigger = trigger & (top1 != self.think_open) & (top1 != self.think_close)

        # Reject positions whose top-1 token is structural rather than a real
        # lexical choice.
        if self._filter_on and not self._droppable_failed and self._droppable is None:
            self._build_droppable(int(logits.size(-1)), logits.device)
        if self._droppable is not None:
            _before = int(trigger.sum().item())
            trigger = trigger & self._droppable[top_idx[:, 0]]
            _after = int(trigger.sum().item())
            if _before != _after:
                try:
                    with _LP_COUNTERS_LOCK:
                        _LP_COUNTERS["skipped_undroppable_token"] += (_before - _after)
                except Exception:
                    pass

        n_trigger = int(trigger.sum().item())
        if n_trigger == 0:
            return logits

        if self.deterministic:
            # Skip the per-token Bernoulli gate: every triggered position is
            # dropped.
            do_perturb = trigger
        else:
            roll = torch.rand(B, generator=self.gen, device=logits.device)
            do_perturb = trigger & (roll < self.p)
        rows = do_perturb.nonzero(as_tuple=True)[0]
        n_perturb = int(rows.numel())

        try:
            trig_top1_sum = float(top1_prob[trigger].sum().item()) if n_trigger else 0.0
            with _LP_COUNTERS_LOCK:
                _LP_COUNTERS["triggered"] += n_trigger
                _LP_COUNTERS["perturbed"] += n_perturb
                _LP_COUNTERS["trigger_top1_prob_sum"] += trig_top1_sum
        except Exception:
            pass

        if n_perturb == 0:
            return logits

        # ---- drop statistics ---------------------------------------------
        # Computed before the mask is applied, while ``probs`` still holds the
        # original policy distribution. All sums are over perturbed rows only.
        drop_detail = None
        try:
            _p = top_probs[rows]                      # (n_perturb, _k_probe)
            _dropped_p = _p[:, : self.drop_top_k]     # what we remove
            _top1_p = _p[:, 0].clamp_min(1e-12)
            _mass = _dropped_p.sum(dim=-1)
            if _k_probe > self.drop_top_k:
                _ru_p = _p[:, self.drop_top_k].clamp_min(1e-12)
            else:
                _ru_p = torch.full_like(_top1_p, 1e-12)
            _top1_logp = _top1_p.log()
            _ru_logp = _ru_p.log()

            _ent_sum = 0.0
            _ent_n = 0
            _ent_vec = None
            if ent_all is not None:
                # The whole-batch measurement already produced H at every row,
                # so the drop sites are a gather rather than a second
                # full-vocab reduction over the perturbed rows.
                _ent_vec = ent_all[rows]
                _ent_sum = float(_ent_vec.sum().item())
                _ent_n = int(_ent_vec.numel())
            elif self.record_details and self.record_entropy:
                _pr = probs[rows]
                _ent_vec = -(_pr * _pr.clamp_min(1e-12).log()).sum(dim=-1)
                _ent_sum = float(_ent_vec.sum().item())
                _ent_n = int(_ent_vec.numel())

            # Entropy of the renormalised distribution the sampler faces after
            # the mask. Masking the top-k and renormalising is an exact
            # transform of the distribution just measured, so no second softmax
            # is needed:
            #     H_post = (H + sum_{dropped} p ln p) / (1 - M) + ln(1 - M)
            # with M the removed mass. This block follows `record_entropy`, not
            # `measure_entropy`, which gates the per-row per-step measurement.
            _hpost_v = None
            if _ent_vec is not None:
                _keep = (1.0 - _mass).clamp_min(1e-6)
                _dplogp = (
                    _dropped_p * _dropped_p.clamp_min(1e-12).log()
                ).sum(dim=-1)
                _hpost_v = (
                    (_ent_vec + _dplogp) / _keep + _keep.log()
                ).clamp_min(0.0)
                # Runner-up's share of all surviving mass: ~1 is a binary fork,
                # ~0 a diffuse tail.
                _resid_conc = (_ru_p / _keep).clamp(0.0, 1.0)
                _hp_stats = torch.stack([
                    _hpost_v.sum(),
                    (_hpost_v * _hpost_v).sum(),
                    (_hpost_v - _ent_vec).sum(),
                    _resid_conc.sum(),
                    (_hpost_v < 0.35).to(_hpost_v.dtype).sum(),
                    (_hpost_v > 1.00).to(_hpost_v.dtype).sum(),
                ]).detach().to("cpu").tolist()
                with _LP_COUNTERS_LOCK:
                    _LP_COUNTERS["drop_H_post_sum"] += _hp_stats[0]
                    _LP_COUNTERS["drop_H_post_sq_sum"] += _hp_stats[1]
                    _LP_COUNTERS["drop_dH_sum"] += _hp_stats[2]
                    _LP_COUNTERS["drop_resid_conc_sum"] += _hp_stats[3]
                    _LP_COUNTERS["drop_H_post_near_forced"] += int(_hp_stats[4])
                    _LP_COUNTERS["drop_H_post_open"] += int(_hp_stats[5])
                    _LP_COUNTERS["drop_H_post_n"] += _ent_n

            _hist = torch.bincount(
                (_p[:, 0] * 10).clamp_(0, 9).to(torch.int64), minlength=10
            ).tolist()

            with _LP_COUNTERS_LOCK:
                _LP_COUNTERS["dropped_logp_sum"] += float(_top1_logp.sum().item())
                _LP_COUNTERS["dropped_prob_sum"] += float(_p[:, 0].sum().item())
                _LP_COUNTERS["dropped_mass_sum"] += float(_mass.sum().item())
                _LP_COUNTERS["runner_up_logp_sum"] += float(_ru_logp.sum().item())
                _LP_COUNTERS["forced_logp_delta_sum"] += float(
                    (_top1_logp - _ru_logp).sum().item()
                )
                _LP_COUNTERS["dropped_entropy_sum"] += _ent_sum
                _LP_COUNTERS["dropped_entropy_n"] += _ent_n
                _h = _LP_COUNTERS["drop_top1_hist"]
                for _b in range(10):
                    _h[_b] += int(_hist[_b])

            if self.record_details and self.idx_to_req:
                # One host sync for the whole block, then plain Python lists.
                drop_detail = {
                    "tok": top_idx[rows, 0].detach().to("cpu").tolist(),
                    "logp": _top1_logp.detach().to("cpu").tolist(),
                    "ru_logp": _ru_logp.detach().to("cpu").tolist(),
                    "mass": _mass.detach().to("cpu").tolist(),
                    "ent": (
                        _ent_vec.detach().to("cpu").tolist()
                        if _ent_vec is not None
                        else [float("nan")] * n_perturb
                    ),
                    "h_post": (
                        _hpost_v.detach().to("cpu").tolist()
                        if _hpost_v is not None
                        else [float("nan")] * n_perturb
                    ),
                }
                if self.detail_topn > 0:
                    _n = min(self.detail_topn, probs.size(-1))
                    _tp, _ti = torch.topk(probs[rows], _n, dim=-1)
                    drop_detail["topn_tok"] = _ti.detach().to("cpu").tolist()
                    drop_detail["topn_logp"] = (
                        _tp.clamp_min(1e-12).log().detach().to("cpu").tolist()
                    )
        except Exception:
            # Metrics must never be able to break generation.
            drop_detail = None

        # Stash one pending record per perturbed row. The replacement is
        # resolved on the next apply() call (see _resolve_pending); recording
        # n_out here is what makes that join exact.
        if self._trace_dir and drop_detail is not None:
            try:
                _rows_l = rows.detach().to("cpu").tolist()
                for _i, _idx in enumerate(_rows_l):
                    _ref = self.output_tok_refs.get(int(_idx))
                    if _ref is None:
                        continue
                    self._pending[int(_idx)] = {
                        "pid": _PID,
                        "uuid": self.idx_to_req.get(int(_idx), ""),
                        "gen_pos": len(_ref),
                        "dropped_token_id": int(drop_detail["tok"][_i]),
                        "dropped_logp": round(float(drop_detail["logp"][_i]), 5),
                        "dropped_prob": round(float(__import__("math").exp(drop_detail["logp"][_i])), 5),
                        "runner_up_logp": round(float(drop_detail["ru_logp"][_i]), 5),
                        "mass_removed": round(float(drop_detail["mass"][_i]), 5),
                        "entropy": round(float(drop_detail["ent"][_i]), 5),
                        "entropy_post_mask": round(float(drop_detail["h_post"][_i]), 5),
                        "topn_tok": drop_detail.get("topn_tok", [[]])[_i]
                                    if "topn_tok" in drop_detail else [],
                        "topn_logp": [round(float(x), 4) for x in drop_detail["topn_logp"][_i]]
                                     if "topn_logp" in drop_detail else [],
                        "replacement_token_id": None,
                        "resolved": False,
                        "_n_out": len(_ref),
                    }
            except Exception:
                pass

        logits[rows.unsqueeze(1), top_idx[rows, : self.drop_top_k]] = float("-inf")
        self.perturb_count[rows] += 1
        # Open the post-drop observation window on these rows.
        if self.measure_entropy and self._post_win > 0:
            self.post_left[rows] = self._post_win

        # Record dropped-token positions for downstream loss masking. The
        # position recorded is the 1-indexed response position (it matches
        # pos_count at this point in apply()), stored under the per-request
        # UUID so the consumer can pop it once the request finalizes. Only paid
        # when idx_to_req is non-empty, i.e. agent_loop stamped UUIDs for
        # explore requests.
        if self.idx_to_req:
            rows_cpu = rows.detach().to("cpu").tolist()
            pos_at_drop = self.pos_count[rows].detach().to("cpu").tolist()
            with _DROPPED_POSITIONS_LOCK:
                for r, p in zip(rows_cpu, pos_at_drop):
                    uuid = self.idx_to_req.get(int(r))
                    if uuid:
                        _DROPPED_POSITIONS.setdefault(uuid, []).append(int(p) - 1)

            # Mirror the per-drop analysis rows into the separate details
            # table, keyed identically. Positions here are the same 0-indexed
            # response positions written above, so the two tables always join.
            if drop_detail is not None:
                with _DROPPED_DETAILS_LOCK:
                    # Bound the table: a parameter sync aborts in-flight
                    # requests whose UUIDs are then never consumed, so the
                    # orphans would otherwise accumulate. Oldest-first
                    # eviction, using dict insertion order.
                    if len(_DROPPED_DETAILS) > _DETAILS_TABLE_MAX:
                        for _stale in list(_DROPPED_DETAILS)[: len(_DROPPED_DETAILS) // 4]:
                            _DROPPED_DETAILS.pop(_stale, None)
                    for _i, (r, p) in enumerate(zip(rows_cpu, pos_at_drop)):
                        uuid = self.idx_to_req.get(int(r))
                        if not uuid:
                            continue
                        rec = _DROPPED_DETAILS.setdefault(
                            uuid,
                            {k: [] for k in ("pos", "tok", "logp", "ru_logp", "mass", "ent")},
                        )
                        if len(rec["pos"]) >= self.detail_cap:
                            continue
                        rec["pos"].append(int(p) - 1)
                        rec["tok"].append(int(drop_detail["tok"][_i]))
                        rec["logp"].append(float(drop_detail["logp"][_i]))
                        rec["ru_logp"].append(float(drop_detail["ru_logp"][_i]))
                        rec["mass"].append(float(drop_detail["mass"][_i]))
                        rec["ent"].append(float(drop_detail["ent"][_i]))
                        if "topn_tok" in drop_detail:
                            rec.setdefault("topn_tok", []).append(drop_detail["topn_tok"][_i])
                            rec.setdefault("topn_logp", []).append(
                                [float(x) for x in drop_detail["topn_logp"][_i]]
                            )
        return logits
