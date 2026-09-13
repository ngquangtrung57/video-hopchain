"""CPU tests for the worker-side drop trace.

The trace records the dropped token together with the token the sampler chose
in its place. That join works because ``output_tok_refs[idx]`` is the live list
vLLM appends to: the drop records its length, and on the next apply() the token
at that index is the replacement.

The trace is written by the worker process, because the logits processor is
built there from a string path while ``consume_dropped_details`` runs in the
server process and their module globals never meet.

Run: pytest tests/workers/rollout/test_drop_trace_on_cpu.py -q
"""

import json
import types

import pytest
import torch

from verl.workers.rollout.exploration import entropy_dropout_lp as E

VOCAB = 10
THINK_OPEN, THINK_CLOSE = 151667, 151668


@pytest.fixture(autouse=True)
def _isolate():
    E._DROPPED_POSITIONS.clear()
    E._DROPPED_DETAILS.clear()
    E.flush_lp_counters()
    yield
    E._DROPPED_POSITIONS.clear()
    E._DROPPED_DETAILS.clear()
    E.flush_lp_counters()


class _Upd:
    def __init__(self, added=(), removed=(), moved=()):
        self.added = list(added)
        self.removed = list(removed)
        self.moved = list(moved)


def _lp(tmpdir, **over):
    cfg = dict(
        enable=True, trigger_mode="high", top_prob_threshold=0.5, perturb_prob=1.0,
        drop_top_k=1, min_position=0, max_perturbations_per_seq=1000,
        restrict_to_think_region=False, deterministic=True,
        record_details=True, record_entropy=True, detail_topn=4, seed=0,
        drop_trace_dir=str(tmpdir), drop_trace_max_rows=1000,
    )
    cfg.update(over)
    E.set_exploration_config(cfg)
    lp = E.EntropyDropoutLP(
        types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_seqs=4)),
        torch.device("cpu"), False)
    for k, v in cfg.items():
        attr = {"top_prob_threshold": "tau", "perturb_prob": "p", "min_position": "min_pos",
                "max_perturbations_per_seq": "cap",
                "restrict_to_think_region": "restrict_think",
                "drop_trace_dir": "_trace_dir", "drop_trace_max_rows": "_trace_max"}.get(k, k)
        if hasattr(lp, attr):
            setattr(lp, attr, v)
    E.flush_lp_counters()
    return lp


def _params(uuid):
    return types.SimpleNamespace(extra_args={"rvrl_exploration": True, "rvrl_req_uuid": uuid})


def _logits():
    x = torch.full((1, VOCAB), -20.0)
    x[0, 3] = 5.0    # top-1, ~0.88 -> will be dropped
    x[0, 7] = 3.0    # runner-up -> where the sampler must land
    return x


def _rows(tmpdir):
    out = []
    for f in tmpdir.iterdir():
        for line in f.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def test_trace_records_dropped_and_replacement(tmp_path):
    """The whole point: dropped token AND what the sampler chose instead."""
    lp = _lp(tmp_path)
    toks = []                                   # the LIVE list vLLM appends to
    lp.update_state(_Upd(added=[(0, _params("u1"), [], toks)]))

    lp.apply(_logits())                         # drop fires at gen position 0
    assert _rows(tmp_path) == [], "must not be written before the sampler runs"
    assert 0 in lp._pending

    toks.append(7)                              # sampler picked the runner-up
    lp.apply(_logits())                         # next step resolves it

    rows = _rows(tmp_path)
    assert len(rows) >= 1
    r = rows[0]
    assert r["dropped_token_id"] == 3           # what we deleted
    assert r["replacement_token_id"] == 7       # what took its place
    assert r["resolved"] is True
    assert r["gen_pos"] == 0
    assert r["uuid"] == "u1"
    assert r["dropped_logp"] > r["runner_up_logp"]   # we removed the likelier token
    assert 0.0 < r["dropped_prob"] <= 1.0


def test_replacement_is_the_token_at_the_recorded_index(tmp_path):
    """Guards the join: a drop at position N resolves to output[N], not output[-1]."""
    lp = _lp(tmp_path)
    toks = [111, 222]                           # sequence already has output
    lp.update_state(_Upd(added=[(0, _params("u2"), [], toks)]))
    lp.apply(_logits())                         # drop at gen_pos 2
    toks.append(7)                              # replacement for THIS drop
    toks.append(999)                            # a later, unrelated token
    lp.apply(_logits())

    r = [x for x in _rows(tmp_path) if x["uuid"] == "u2"][0]
    assert r["gen_pos"] == 2
    assert r["replacement_token_id"] == 7, "must not pick up the later token"


def test_slot_reuse_does_not_misattribute(tmp_path):
    """vLLM reuses slots. A stale pending must never be joined to a new request."""
    lp = _lp(tmp_path)
    toks_a = []
    lp.update_state(_Upd(added=[(0, _params("uA"), [], toks_a)]))
    lp.apply(_logits())
    assert 0 in lp._pending
    # slot 0 is reassigned before the drop ever resolved
    toks_b = []
    lp.update_state(_Upd(added=[(0, _params("uB"), [], toks_b)]))
    assert 0 not in lp._pending, "stale pending survived slot reuse"
    toks_b.append(7)
    lp.apply(_logits())
    for r in _rows(tmp_path):
        if r["resolved"]:
            assert r["uuid"] != "uA" or r["replacement_token_id"] != 7


def test_removed_slot_flushes_unresolved(tmp_path):
    """A drop on the final step must still be recorded, not silently lost."""
    lp = _lp(tmp_path)
    toks = []
    lp.update_state(_Upd(added=[(0, _params("u3"), [], toks)]))
    lp.apply(_logits())
    lp.update_state(_Upd(removed=[0]))          # finished before resolving
    rows = [r for r in _rows(tmp_path) if r["uuid"] == "u3"]
    assert len(rows) == 1
    assert rows[0]["dropped_token_id"] == 3
    assert rows[0]["resolved"] is False
    assert rows[0]["replacement_token_id"] is None


def test_trace_disabled_writes_nothing(tmp_path):
    lp = _lp(tmp_path, drop_trace_dir="")
    toks = []
    lp.update_state(_Upd(added=[(0, _params("u4"), [], toks)]))
    lp.apply(_logits())
    toks.append(7)
    lp.apply(_logits())
    assert _rows(tmp_path) == []
    assert E.flush_lp_counters()["perturbed"] == 2   # dropping itself unaffected


def test_cap_stops_writing(tmp_path):
    lp = _lp(tmp_path, drop_trace_max_rows=2)
    toks = []
    lp.update_state(_Upd(added=[(0, _params("u5"), [], toks)]))
    for _ in range(8):
        lp.apply(_logits())
        toks.append(7)
    lp.apply(_logits())
    assert len(_rows(tmp_path)) <= 2


# ---------------------------------------------------------------------------
# A full drop trace must not silence the periodic counter snapshots: those rows
# are exempt from drop_trace_max_rows, so they keep reaching disk once the drop
# rows stop.
# ---------------------------------------------------------------------------


def test_counter_snapshot_survives_a_capped_drop_trace(tmp_path):
    lp = _lp(tmp_path, drop_trace_max_rows=3)

    for i in range(10):
        lp._trace_write({"row": i})
    assert lp._trace_rows == 3, "cap must still stop ordinary drop rows"

    lp._trace_write({"kind": "counters", "perturbed": 42,
                     "mask_ipc_published": 7}, exempt_from_cap=True)

    rows = _rows(tmp_path)
    drops = [r for r in rows if r.get("kind") != "counters"]
    counters = [r for r in rows if r.get("kind") == "counters"]

    assert len(drops) == 3, "drop rows must respect the cap"
    assert len(counters) == 1, "THE BUG: a full drop trace silenced the health channel"
    assert counters[0]["mask_ipc_published"] == 7
    assert lp._trace_rows == 3, "exempt rows must not consume cap budget"


def test_exempt_rows_do_not_consume_the_cap(tmp_path):
    lp = _lp(tmp_path, drop_trace_max_rows=5)
    for _ in range(20):
        lp._trace_write({"kind": "counters"}, exempt_from_cap=True)
    assert lp._trace_rows == 0
    for i in range(5):
        lp._trace_write({"row": i})
    assert lp._trace_rows == 5
    assert len([r for r in _rows(tmp_path) if r.get("kind") == "counters"]) == 20


def test_disabled_dir_still_wins_over_exemption(tmp_path):
    lp = _lp(tmp_path, drop_trace_dir="", drop_trace_max_rows=5)
    lp._trace_write({"kind": "counters"}, exempt_from_cap=True)
    assert lp._trace_fh is None
