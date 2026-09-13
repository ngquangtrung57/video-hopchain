"""CPU tests for the exploration drop-analysis metrics.

Covers three things that can break silently:

1. The drop statistics must describe the original, pre-mask distribution. They
   are computed inside ``apply`` just before the logits are set to -inf, so an
   ordering mistake would measure the masked distribution instead.
2. ``flush_lp_counters`` must be type-preserving. The histogram counter is a
   list, and a blanket ``= 0`` reset would turn it into an int so that the next
   ``hist[b] += 1`` raises.
3. ``merge_lp_counters`` must sum lists element-wise. The call sites that
   aggregate counters across vLLM workers swallow exceptions, so a
   ``TypeError`` there would make every ``rvrl/lp_*`` metric vanish silently.

Plus the guarantee that with ``record_details`` off nothing is recorded and the
loss-mask table (``_DROPPED_POSITIONS``) is untouched.

Run: pytest tests/workers/rollout/test_exploration_drop_metrics_on_cpu.py -q
"""

import math
import types

import pytest
import torch

from verl.workers.rollout.exploration import entropy_dropout_lp as E

VOCAB = 10


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """The LP keeps its tables and counters at module scope, shared by every
    test in the process. Without this reset a test that consumes only the
    DETAILS table leaves its POSITIONS behind and the next test sees them.
    """
    E._DROPPED_POSITIONS.clear()
    E._DROPPED_DETAILS.clear()
    E.flush_lp_counters()
    yield
    E._DROPPED_POSITIONS.clear()
    E._DROPPED_DETAILS.clear()
    E.flush_lp_counters()


def _make_lp(**overrides):
    cfg = dict(
        enable=True,
        trigger_mode="high",
        top_prob_threshold=0.5,
        perturb_prob=1.0,
        drop_top_k=1,
        min_position=0,
        max_perturbations_per_seq=1000,
        restrict_to_think_region=False,
        deterministic=True,
        record_details=True,
        record_entropy=True,
        detail_topn=4,
        seed=0,
    )
    cfg.update(overrides)
    E.set_exploration_config(cfg)
    vllm_config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=4)
    )
    lp = E.EntropyDropoutLP(vllm_config, torch.device("cpu"), False)
    # Explicit assignment: set_exploration_config only mutates module state, and
    # __init__ may have read a stale value from a previous test's env var.
    for k, v in cfg.items():
        attr = {
            "top_prob_threshold": "tau",
            "perturb_prob": "p",
            "min_position": "min_pos",
            "max_perturbations_per_seq": "cap",
            "restrict_to_think_region": "restrict_think",
        }.get(k, k)
        if hasattr(lp, attr):
            setattr(lp, attr, v)
    E.flush_lp_counters()
    return lp


def _two_seq_logits():
    logits = torch.full((2, VOCAB), -20.0)
    logits[0, 3] = 5.0   # top1 prob ~0.88
    logits[0, 7] = 3.0   # runner-up
    logits[1, 2] = 2.0   # top1 prob ~0.5
    logits[1, 5] = 1.9
    return logits


def _run(lp, logits):
    lp.active[0] = True
    lp.active[1] = True
    lp.idx_to_req[0] = "uA"
    lp.idx_to_req[1] = "uB"
    return lp.apply(logits.clone())


def test_drop_stats_match_original_distribution():
    lp = _make_lp()
    logits = _two_seq_logits()
    probs = torch.softmax(logits, dim=-1)
    out = _run(lp, logits)
    snap = E.flush_lp_counters()

    assert snap["perturbed"] == 2
    assert snap["triggered"] == 2

    det = E.consume_dropped_details("uA")
    assert len(det["pos"]) == 1
    # dropped token is the PRE-drop argmax
    assert det["tok"][0] == int(probs[0].argmax())
    # its log-prob is measured on the original distribution
    assert det["logp"][0] == pytest.approx(float(probs[0].max().log()), abs=1e-4)
    # runner-up is the second highest, i.e. where the sampler is pushed
    second = float(probs[0].sort(descending=True).values[1].log())
    assert det["ru_logp"][0] == pytest.approx(second, abs=1e-4)
    # forced penalty is positive by construction
    assert det["logp"][0] - det["ru_logp"][0] > 0
    # mass removed equals the top-1 probability when drop_top_k == 1
    assert det["mass"][0] == pytest.approx(float(probs[0].max()), abs=1e-4)
    # entropy is the full-vocab entropy of the original distribution
    H = float(-(probs[0] * probs[0].clamp_min(1e-12).log()).sum())
    assert det["ent"][0] == pytest.approx(H, abs=1e-4)
    # the mask itself still happened
    assert math.isinf(out[0, int(probs[0].argmax())].item())
    assert out[0, 7].item() == 3.0


def test_topn_lets_us_recover_the_original_logprob():
    lp = _make_lp()
    _run(lp, _two_seq_logits())
    det = E.consume_dropped_details("uA")
    assert len(det["topn_tok"][0]) == 4
    # first candidate is the dropped token, and the recorded log-probs are
    # descending -- this is what makes the sampled token's ORIGINAL log-prob
    # recoverable offline (vLLM only returns the renormalised one).
    assert det["topn_tok"][0][0] == det["tok"][0]
    lps = det["topn_logp"][0]
    assert all(lps[i] >= lps[i + 1] for i in range(len(lps) - 1))
    assert lps[0] == pytest.approx(det["logp"][0], abs=1e-4)


def test_counter_sum_matches_per_drop_records():
    lp = _make_lp()
    _run(lp, _two_seq_logits())
    snap = E.flush_lp_counters()
    a = E.consume_dropped_details("uA")
    b = E.consume_dropped_details("uB")
    expect = (a["logp"][0] - a["ru_logp"][0]) + (b["logp"][0] - b["ru_logp"][0])
    assert snap["forced_logp_delta_sum"] == pytest.approx(expect, abs=1e-4)
    assert snap["dropped_logp_sum"] == pytest.approx(a["logp"][0] + b["logp"][0], abs=1e-4)
    assert sum(snap["drop_top1_hist"]) == 2


def test_record_details_off_records_nothing_but_keeps_positions():
    """Back-compat: the loss-mask path must be unaffected by the analysis flag."""
    lp = _make_lp(record_details=False)
    _run(lp, _two_seq_logits())
    assert E.consume_dropped_details("uA") == {}
    # positions -- which feed exploration_drop_mask -- are still recorded
    assert E.consume_dropped_positions("uA") == [0]
    snap = E.flush_lp_counters()
    # cheap aggregate stats are collected regardless of the flag
    assert snap["perturbed"] == 2
    assert snap["dropped_logp_sum"] < 0.0
    # entropy is the only stat gated off with details
    assert snap["dropped_entropy_n"] == 0


def test_flush_is_type_preserving():
    E._LP_COUNTERS["perturbed"] = 5
    E._LP_COUNTERS["dropped_logp_sum"] = -3.5
    E._LP_COUNTERS["drop_top1_hist"][3] = 7
    snap = E.flush_lp_counters()

    assert snap["perturbed"] == 5 and snap["drop_top1_hist"][3] == 7
    assert E._LP_COUNTERS["perturbed"] == 0
    assert isinstance(E._LP_COUNTERS["dropped_logp_sum"], float)
    hist = E._LP_COUNTERS["drop_top1_hist"]
    assert isinstance(hist, list) and len(hist) == 10 and sum(hist) == 0
    # the snapshot must be a copy, not an alias
    E._LP_COUNTERS["drop_top1_hist"][3] = 99
    assert snap["drop_top1_hist"][3] == 7
    # and the reset value must still support +=  (the original crash)
    E._LP_COUNTERS["drop_top1_hist"][0] += 1


def test_merge_sums_lists_elementwise():
    agg = {}
    E.merge_lp_counters(agg, {"perturbed": 3, "dropped_logp_sum": -1.5,
                              "drop_top1_hist": [1, 0, 0, 0, 0, 0, 0, 0, 0, 2]})
    E.merge_lp_counters(agg, {"perturbed": 4, "dropped_logp_sum": -2.5,
                              "drop_top1_hist": [0, 5, 0, 0, 0, 0, 0, 0, 0, 3]})
    assert agg["perturbed"] == 7
    assert agg["dropped_logp_sum"] == pytest.approx(-4.0)
    assert agg["drop_top1_hist"] == [1, 5, 0, 0, 0, 0, 0, 0, 0, 5]


def test_merge_is_defensive():
    agg = {"perturbed": 7}
    E.merge_lp_counters(agg, {"perturbed": None})
    E.merge_lp_counters(agg, "not a dict")
    assert agg["perturbed"] == 7
    ragged = {}
    E.merge_lp_counters(ragged, {"h": [1, 2]})
    E.merge_lp_counters(ragged, {"h": [1, 2, 3]})
    assert ragged["h"] == [2, 4, 3]


def test_details_table_pop_once():
    E._DROPPED_DETAILS["u1"] = {"pos": [1, 2]}
    assert E.consume_dropped_details("u1") == {"pos": [1, 2]}
    assert E.consume_dropped_details("u1") == {}
    assert E.consume_dropped_details("") == {}


# ---------------------------------------------------------------------------
# think_prefilled: the region gate vs VERL_THINK_PREFILL
#
# With prefill on, <think> sits in the PROMPT, so the model never emits token
# 151667 and the watcher in apply() cannot flip inside_think True. Combined with
# restrict_to_think_region that would make exploration a permanent no-op, so the
# region flag has to start True instead.
# ---------------------------------------------------------------------------

THINK_OPEN, THINK_CLOSE = 151667, 151668


class _Upd:
    """Minimal BatchUpdate stand-in."""

    def __init__(self, added):
        self.added = added
        self.removed = []
        self.moved = []


def _region_lp(prefilled):
    lp = _make_lp(restrict_to_think_region=True, think_prefilled=prefilled,
                  record_entropy=False, detail_topn=0)
    lp.restrict_think = True
    lp.think_prefilled = prefilled
    return lp


def _explore_params(uuid):
    return types.SimpleNamespace(
        extra_args={"rvrl_exploration": True, "rvrl_req_uuid": uuid}
    )


def _one_seq_logits():
    x = torch.full((1, VOCAB), -20.0)
    x[0, 3] = 5.0
    x[0, 7] = 3.0
    return x


def test_prefilled_region_starts_open_and_drop_fires():
    lp = _region_lp(True)
    # output_tok_ids is EMPTY: the opening tag was in the prompt, not the output
    lp.update_state(_Upd([(0, _explore_params("u1"), [THINK_OPEN], [])]))
    assert bool(lp.inside_think[0].item()) is True
    lp.apply(_one_seq_logits())
    snap = E.flush_lp_counters()
    assert snap["perturbed"] == 1
    assert snap["skipped_outside_think"] == 0


def test_non_prefilled_behaviour_is_unchanged():
    """Regression guard: without prefill the model must still emit <think> first."""
    lp = _region_lp(False)
    lp.update_state(_Upd([(0, _explore_params("u2"), [], [])]))
    assert bool(lp.inside_think[0].item()) is False
    lp.apply(_one_seq_logits())
    snap = E.flush_lp_counters()
    assert snap["perturbed"] == 0
    assert snap["skipped_outside_think"] == 1


def test_close_tag_still_ends_the_region_under_prefill():
    """Drops must never reach the answer, prefill or not."""
    lp = _region_lp(True)
    toks = []
    lp.update_state(_Upd([(0, _explore_params("u3"), [THINK_OPEN], toks)]))
    toks.append(THINK_CLOSE)          # model closes the think block
    lp.apply(_one_seq_logits())
    assert E.flush_lp_counters()["perturbed"] == 0
