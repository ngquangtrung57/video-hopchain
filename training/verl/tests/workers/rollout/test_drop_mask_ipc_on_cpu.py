"""CPU tests for the loss-mask transport and the token-class filter.

vLLM registers the logits processor by string path, so it is built in the
spawned worker process while ``consume_dropped_positions`` runs in the server
process; module globals are per-process and never meet, so drop positions cross
that boundary through one small file per request.

These tests simulate the process boundary the only way a CPU test can: publish,
then clear the in-process table, then consume. A test that skipped the clear
would also pass when the file transport is absent.

Run: pytest tests/workers/rollout/test_drop_mask_ipc_on_cpu.py -q
"""

import os
import types

import pytest
import torch

from verl.workers.rollout.exploration import entropy_dropout_lp as E

VOCAB = 10


@pytest.fixture(autouse=True)
def _isolate(tmp_path):
    os.environ["VERL_EXPLORATION_IPC_DIR"] = str(tmp_path / "ipc")
    E._DROPPED_POSITIONS.clear()
    E._DROPPED_DETAILS.clear()
    E.flush_lp_counters()
    yield
    E._DROPPED_POSITIONS.clear()
    E._DROPPED_DETAILS.clear()
    E.flush_lp_counters()
    os.environ.pop("VERL_EXPLORATION_IPC_DIR", None)


def _make_lp(**overrides):
    cfg = dict(
        enable=True, trigger_mode="high", top_prob_threshold=0.5, perturb_prob=1.0,
        drop_top_k=1, min_position=0, max_perturbations_per_seq=1000,
        restrict_to_think_region=False, deterministic=True, record_details=False,
        record_entropy=False, detail_topn=0, seed=0,
        skip_special_tokens=False, skip_whitespace_punct=False,
        skip_digit_tokens=False, skip_subword_continuation=False,
    )
    cfg.update(overrides)
    E.set_exploration_config(cfg)
    vllm_config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=4)
    )
    lp = E.EntropyDropoutLP(vllm_config, torch.device("cpu"), False)
    lp.enable = True
    lp.tau = cfg["top_prob_threshold"]
    lp.deterministic = True
    lp.restrict_think = cfg["restrict_to_think_region"]
    E.flush_lp_counters()
    return lp


def _logits(top=3, runner=7):
    x = torch.full((1, VOCAB), -20.0)
    x[0, top] = 5.0
    x[0, runner] = 3.0
    return x


def _fire(lp, uuid="u1", logits=None):
    lp.active[0] = True
    lp.idx_to_req[0] = uuid
    lp.output_tok_refs[0] = []
    return lp.apply((logits if logits is not None else _logits()).clone())


# --------------------------------------------------------------- IPC transport

def test_publish_then_consume_across_the_process_boundary():
    """The whole point: consume must work when the in-process table is EMPTY."""
    assert E._ipc_publish("req-a", [3, 9, 14]) is True
    E._DROPPED_POSITIONS.clear()          # <- simulates the other process
    assert E.consume_dropped_positions("req-a") == [3, 9, 14]


def test_consume_is_idempotent_and_unlinks():
    E._ipc_publish("req-b", [1, 2])
    E._DROPPED_POSITIONS.clear()
    assert E.consume_dropped_positions("req-b") == [1, 2]
    assert not os.path.exists(E._ipc_path("req-b"))
    assert E.consume_dropped_positions("req-b") == []


def test_consume_prefers_the_local_table_when_same_process():
    """Unit tests and any non-spawning vLLM mode keep the fast in-memory path."""
    E._DROPPED_POSITIONS["req-c"] = [42]
    E._ipc_publish("req-c", [1, 1, 1])
    assert E.consume_dropped_positions("req-c") == [42]


def test_missing_uuid_returns_empty_not_raises():
    assert E.consume_dropped_positions("never-existed") == []
    assert E.consume_dropped_positions("") == []


def test_publish_refuses_empty_position_list():
    assert E._ipc_publish("req-d", []) is False
    assert not os.path.exists(E._ipc_path("req-d"))


def test_removed_slot_publishes_for_the_server():
    """End to end: a drop, then the request leaves the batch, then the server
    (with no shared memory) can still read the positions."""
    lp = _make_lp()
    _fire(lp, "req-e")
    assert E._DROPPED_POSITIONS.get("req-e"), "drop was not recorded at all"

    class _Upd:
        added, moved = [], []
        removed = [0]
    lp.update_state(_Upd())

    E._DROPPED_POSITIONS.clear()          # <- the server has its own globals
    assert E.consume_dropped_positions("req-e") == [0]


def test_publish_counter_makes_a_dead_mask_visible():
    """A run with perturbations but zero published files is the failure this
    whole change set exists to remove. The counter is what surfaces it."""
    lp = _make_lp()
    _fire(lp, "req-f")

    class _Upd:
        added, moved = [], []
        removed = [0]
    lp.update_state(_Upd())
    snap = E.flush_lp_counters()
    assert snap["perturbed"] == 1
    assert snap["mask_ipc_published"] == 1
    assert snap["mask_ipc_positions"] == 1
    assert snap["mask_ipc_publish_failed"] == 0


def test_sweep_removes_orphans_only():
    E._ipc_publish("orphan", [1])
    E._ipc_publish("fresh", [2])
    os.utime(E._ipc_path("orphan"), (0, 0))     # ancient
    assert E.ipc_sweep(max_age_s=60.0) == 1
    assert not os.path.exists(E._ipc_path("orphan"))
    assert os.path.exists(E._ipc_path("fresh"))


# ---------------------------------------------------------- token-class filter

def test_undroppable_top1_is_not_dropped():
    lp = _make_lp()
    block = torch.ones(VOCAB, dtype=torch.bool)
    block[3] = False                       # token 3 is the top-1
    lp._droppable = block
    out = _fire(lp)
    snap = E.flush_lp_counters()
    assert snap["triggered"] == 0
    assert snap["perturbed"] == 0
    assert snap["skipped_undroppable_token"] == 1
    assert torch.isfinite(out[0, 3]), "a blocked token must survive the mask"


def test_droppable_top1_still_drops():
    lp = _make_lp()
    lp._droppable = torch.ones(VOCAB, dtype=torch.bool)
    out = _fire(lp)
    snap = E.flush_lp_counters()
    assert snap["perturbed"] == 1
    assert snap["skipped_undroppable_token"] == 0
    assert not torch.isfinite(out[0, 3])


def test_filter_absent_preserves_legacy_behaviour():
    """All skip flags off => no token-class filter is built and none is applied."""
    lp = _make_lp()
    assert lp._filter_on is False
    out = _fire(lp)
    assert E.flush_lp_counters()["perturbed"] == 1
    assert not torch.isfinite(out[0, 3])


def test_build_failure_disables_the_filter_loudly(capsys):
    """No tokenizer reachable => filter OFF and it SAYS SO. A filter that cannot
    be built must never be mistaken for a filter that found nothing to skip."""
    lp = _make_lp(skip_whitespace_punct=True)
    lp._vllm_config = types.SimpleNamespace()      # no model_config at all
    lp._build_droppable(VOCAB, torch.device("cpu"))
    assert lp._droppable is None
    assert lp._droppable_failed is True
    assert "DISABLED" in capsys.readouterr().out


def test_failed_build_is_not_retried_every_token():
    lp = _make_lp(skip_whitespace_punct=True)
    lp._vllm_config = types.SimpleNamespace()
    _fire(lp)
    assert lp._droppable_failed is True
    calls = []
    lp._build_droppable = lambda *a, **k: calls.append(1)
    _fire(lp, "u2")
    assert calls == [], "a failed build must not be retried on every apply()"
