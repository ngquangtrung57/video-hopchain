"""CPU tests for the entropy instrumentation.

The exploration trigger is log-prob based (``p_top1`` inside a band), while
entropy is a property of the whole next-token distribution. These tests pin the
five ways that instrumentation can silently produce a plausible but wrong
number:

1. ``H_post`` -- the entropy of the renormalised distribution the sampler
   actually draws from after the mask -- is computed in CLOSED FORM from
   ``H_orig`` and the removed mass, to avoid a second full-vocab softmax. If the
   algebra is wrong the metric still looks reasonable, so it is checked against
   an explicit mask-and-renormalise.
2. Entropy must be measured on the ORIGINAL distribution, before the mask. The
   measurement runs at the top of ``apply`` and the mask at the bottom; an
   ordering mistake would silently report post-mask entropy as policy entropy.
3. Wave-1 anchors must be MEASURED but never PERTURBED. They are the control
   arm; if the measure stamp leaked into the drop decision the comparison would
   be against a treated control, and the drop-mask IPC would also be polluted.
4. The per-slot accumulators must ride along with vLLM slot moves, exactly like
   ``perturb_count``. Otherwise one rollout's entropy is attributed to another
   whenever the batch is reshuffled, which no aggregate metric would reveal.
5. With ``measure_entropy`` off, no entropy work runs and the drop behaviour is
   unchanged.

Run: pytest tests/workers/rollout/test_exploration_entropy_metrics_on_cpu.py -q
"""

import types

import pytest
import torch

from verl.workers.rollout.exploration import entropy_dropout_lp as E

VOCAB = 64


@pytest.fixture(autouse=True)
def _isolate_module_state():
    E._DROPPED_POSITIONS.clear()
    E._DROPPED_DETAILS.clear()
    E.flush_lp_counters()
    yield
    E._DROPPED_POSITIONS.clear()
    E._DROPPED_DETAILS.clear()
    E.flush_lp_counters()


def _make_lp(max_seqs=4, **overrides):
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
        measure_entropy=True,
        entropy_measure_stride=1,
        entropy_post_drop_window=3,
    )
    cfg.update(overrides)
    E.set_exploration_config(cfg)
    vllm_config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_seqs)
    )
    lp = E.EntropyDropoutLP(vllm_config, torch.device("cpu"), False)
    for k, v in cfg.items():
        attr = {
            "top_prob_threshold": "tau",
            "perturb_prob": "p",
            "min_position": "min_pos",
            "max_perturbations_per_seq": "cap",
            "restrict_to_think_region": "restrict_think",
            "entropy_measure_stride": "_ent_stride",
            "entropy_post_drop_window": "_post_win",
        }.get(k, k)
        if hasattr(lp, attr):
            setattr(lp, attr, v)
    E.flush_lp_counters()
    return lp


def _logits(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    lg = torch.randn(n, VOCAB, generator=g) * 2.0
    # Force row 0 to a confident two-way fork so it always triggers a drop.
    lg[0] = -20.0
    lg[0, 3] = 5.0
    lg[0, 7] = 3.2
    return lg


def _brute_post_mask_entropy(logits_row: torch.Tensor, k: int = 1) -> float:
    """Explicitly mask the top-k, renormalise, and measure. Ground truth."""
    lg = logits_row.clone()
    top = torch.topk(lg, k).indices
    lg[top] = float("-inf")
    p = torch.softmax(lg, dim=-1)
    return float(-(p * p.clamp_min(1e-30).log()).sum())


# --------------------------------------------------------------------------
# 1. the closed form


def test_post_mask_entropy_closed_form_matches_brute_force():
    lp = _make_lp()
    lg = _logits()
    lp.active[0] = True
    lp.measure[0] = True
    lp.idx_to_req[0] = "uA"
    lp.apply(lg.clone())

    det = E.consume_dropped_details("uA")
    assert len(det["pos"]) == 1
    snap = E.flush_lp_counters()

    assert snap["drop_H_post_n"] == 1
    got = snap["drop_H_post_sum"]
    want = _brute_post_mask_entropy(lg[0])
    assert got == pytest.approx(want, abs=1e-4)

    # And it must differ from the pre-mask entropy, or the metric is redundant.
    p = torch.softmax(lg[0], dim=-1)
    h_orig = float(-(p * p.clamp_min(1e-30).log()).sum())
    assert abs(got - h_orig) > 1e-3
    assert snap["drop_dH_sum"] == pytest.approx(got - h_orig, abs=1e-4)


def test_residual_concentration_is_runner_up_share_of_surviving_mass():
    lp = _make_lp()
    lg = _logits()
    lp.active[0] = True
    lp.measure[0] = True
    lp.idx_to_req[0] = "uA"
    lp.apply(lg.clone())
    snap = E.flush_lp_counters()

    p = torch.softmax(lg[0], dim=-1)
    srt = p.sort(descending=True).values
    want = float(srt[1] / (1.0 - srt[0]))
    assert snap["drop_resid_conc_sum"] == pytest.approx(want, abs=1e-4)
    # Row 0 is built as a clean two-way fork, so nearly all surviving mass is
    # on the runner-up.
    assert want > 0.8


# --------------------------------------------------------------------------
# 2. measured before the mask


def test_entropy_is_measured_on_the_premask_distribution():
    lp = _make_lp()
    lg = _logits()
    lp.active[0] = True
    lp.measure[0] = True
    lp.idx_to_req[0] = "uA"
    lp.apply(lg.clone())
    snap = E.flush_lp_counters()

    p = torch.softmax(lg[0], dim=-1)
    h_orig = float(-(p * p.clamp_min(1e-30).log()).sum())
    assert snap["ent_tok_n_explore"] == 1
    assert snap["ent_tok_sum_explore"] == pytest.approx(h_orig, abs=1e-4)
    # The two quantities are far apart on this row, so reporting the post-mask
    # value by mistake would be caught here rather than looking plausible.
    assert abs(_brute_post_mask_entropy(lg[0]) - h_orig) > 0.3


def test_near_forced_substitution_is_detected():
    """A drop can reduce local entropy, and H_post has to report that.

    Row 0 is a two-way split (p1=0.858, p2=0.142). Deleting the top token leaves
    essentially all remaining mass on a single token, so the step is a
    deterministic rank-2 substitution: H 0.408 -> ~0 nats. p_top1 alone cannot
    tell that apart from a genuine fork.
    """
    lp = _make_lp()
    lg = _logits()
    lp.active[0] = True
    lp.measure[0] = True
    lp.idx_to_req[0] = "uA"
    lp.apply(lg.clone())
    snap = E.flush_lp_counters()

    assert snap["drop_H_post_n"] == 1
    assert snap["drop_H_post_sum"] < 0.35          # near-forced
    assert snap["drop_H_post_near_forced"] == 1
    assert snap["drop_H_post_open"] == 0
    # Entropy went DOWN: the mask closed the position rather than opening it.
    assert snap["drop_dH_sum"] < 0.0


# --------------------------------------------------------------------------
# 3. anchors are measured, never perturbed


def test_anchor_rows_are_measured_but_never_dropped():
    lp = _make_lp()
    lg = _logits()
    lp.active[0] = False        # anchor: measured, not explored
    lp.measure[0] = True
    lp.active[1] = True
    lp.measure[1] = True
    lp.idx_to_req[1] = "uB"

    out = lp.apply(lg.clone())
    snap = E.flush_lp_counters()

    # Anchor entropy landed in the anchor bucket, not the explore bucket.
    assert snap["ent_tok_n_anchor"] == 1
    assert snap["ent_tok_n_explore"] == 1
    assert snap["ent_tok_sum_anchor"] > 0.0

    # Row 0 was never masked: its argmax logit survives untouched.
    assert out[0, 3].item() == pytest.approx(5.0)
    assert torch.isfinite(out[0]).all()
    # And it never reached the drop-mask table.
    assert "uA" not in E._DROPPED_POSITIONS


def test_measure_stamp_alone_does_not_activate_dropping():
    """`rvrl_measure` must not be mistaken for `rvrl_exploration`."""
    lp = _make_lp()
    upd = types.SimpleNamespace(
        removed=[],
        moved=[],
        added=[(
            0,
            types.SimpleNamespace(extra_args={"rvrl_measure": True, "rvrl_req_uuid": "m0"}),
            [1, 2],
            [],
        )],
    )
    lp.update_state(upd)
    assert bool(lp.measure[0]) is True
    assert bool(lp.active[0]) is False
    assert lp.idx_to_meas_req[0] == "m0"
    # The drop-mask IPC map must stay clean: it decides which tokens are
    # excluded from the loss.
    assert 0 not in lp.idx_to_req


# --------------------------------------------------------------------------
# 4. slot moves


def test_entropy_accumulators_follow_slot_moves():
    lp = _make_lp()
    lp.measure[0] = True
    lp.ent_sum[0] = 4.0
    lp.ent_sq[0] = 9.0
    lp.ent_n[0] = 2
    lp.post_left[0] = 3
    lp.idx_to_meas_req[0] = "u0"

    upd = types.SimpleNamespace(
        removed=[], added=[],
        moved=[(0, 2, E.MoveDirectionality.UNIDIRECTIONAL)],
    )
    lp.update_state(upd)

    assert float(lp.ent_sum[2]) == pytest.approx(4.0)
    assert float(lp.ent_sq[2]) == pytest.approx(9.0)
    assert int(lp.ent_n[2]) == 2
    assert int(lp.post_left[2]) == 3
    assert bool(lp.measure[2]) is True
    assert lp.idx_to_meas_req[2] == "u0"
    # Source slot is cleared, or the next request in it inherits stale entropy.
    assert int(lp.ent_n[0]) == 0
    assert float(lp.ent_sum[0]) == 0.0
    assert bool(lp.measure[0]) is False
    assert 0 not in lp.idx_to_meas_req


def test_entropy_accumulators_swap():
    lp = _make_lp()
    lp.ent_n[0], lp.ent_n[1] = 5, 9
    lp.ent_sum[0], lp.ent_sum[1] = 1.0, 2.0
    lp.idx_to_meas_req[0] = "a"
    lp.idx_to_meas_req[1] = "b"
    upd = types.SimpleNamespace(
        removed=[], added=[], moved=[(0, 1, E.MoveDirectionality.SWAP)]
    )
    lp.update_state(upd)
    assert int(lp.ent_n[0]) == 9 and int(lp.ent_n[1]) == 5
    assert float(lp.ent_sum[0]) == pytest.approx(2.0)
    assert lp.idx_to_meas_req[0] == "b" and lp.idx_to_meas_req[1] == "a"


# --------------------------------------------------------------------------
# 5. post-drop window and the off switch


def test_post_drop_window_counts_following_tokens_only():
    lp = _make_lp(entropy_post_drop_window=2)
    lp._post_win = 2
    lg = _logits()
    lp.active[0] = True
    lp.measure[0] = True
    lp.idx_to_req[0] = "uA"

    lp.apply(lg.clone())                 # drop happens; window armed to 2
    assert int(lp.post_left[0]) == 2
    s1 = E.flush_lp_counters()
    # The measurement precedes the drop within a call, so the drop step itself
    # is not inside its own window.
    assert s1["post_drop_ent_n"] == 0

    lp.apply(lg.clone())                 # first token after the drop
    s2 = E.flush_lp_counters()
    assert s2["post_drop_ent_n"] == 1
    assert s2["post_drop_ent_sum"] > 0.0


def test_measure_off_disables_the_expensive_per_token_measurement():
    """`measure_entropy=false` must cost nothing per decode step.

    It gates the per-row, per-step measurement -- the only part with a real
    price. The drop-site H_post follows `record_entropy` instead, because it is
    an exact transform of an entropy already being computed.
    """
    lp = _make_lp(measure_entropy=False)
    lp.measure_entropy = False
    lg = _logits()
    lp.active[0] = True
    lp.measure[0] = True          # even if a row is marked, nothing is measured
    lp.idx_to_req[0] = "uA"
    out = lp.apply(lg.clone())
    snap = E.flush_lp_counters()

    # The drop still happens exactly as before ...
    assert snap["perturbed"] == 1
    assert torch.isinf(out[0, 3])
    # ... the per-step measurement is fully off ...
    assert snap["ent_tok_n_explore"] == 0
    assert snap["ent_tok_n_anchor"] == 0
    assert snap["post_drop_ent_n"] == 0
    assert sum(snap["ent_hist_explore"]) == 0
    assert int(lp.ent_n[0]) == 0
    assert int(lp.post_left[0]) == 0     # window never armed
    # ... but H_post still rides along with record_entropy, free.
    assert snap["drop_H_post_n"] == 1


def test_record_entropy_off_disables_h_post_too():
    lp = _make_lp(measure_entropy=False, record_entropy=False)
    lp.measure_entropy = False
    lp.record_entropy = False
    lg = _logits()
    lp.active[0] = True
    lp.idx_to_req[0] = "uA"
    lp.apply(lg.clone())
    snap = E.flush_lp_counters()

    assert snap["perturbed"] == 1        # dropping is unaffected
    assert snap["drop_H_post_n"] == 0
    assert snap["dropped_entropy_n"] == 0


def test_histogram_and_counter_roundtrip_are_type_preserving():
    """The entropy histograms are lists; a blanket reset would break them."""
    lp = _make_lp()
    lg = _logits()
    lp.active[0] = True
    lp.measure[0] = True
    lp.measure[1] = True
    lp.idx_to_req[0] = "uA"
    lp.apply(lg.clone())

    snap = E.flush_lp_counters()
    assert isinstance(snap["ent_hist_explore"], list)
    assert isinstance(snap["ent_hist_anchor"], list)
    assert len(snap["ent_hist_explore"]) == E._ENT_HIST_NBIN
    assert sum(snap["ent_hist_explore"]) == 1
    assert sum(snap["ent_hist_anchor"]) == 1
    # Reset must preserve the list type, or the next bincount fold raises.
    assert isinstance(E._LP_COUNTERS["ent_hist_explore"], list)
    assert sum(E._LP_COUNTERS["ent_hist_explore"]) == 0

    # merge must sum the histograms element-wise across worker snapshots.
    agg = {}
    E.merge_lp_counters(agg, snap)
    E.merge_lp_counters(agg, snap)
    assert sum(agg["ent_hist_explore"]) == 2
    assert agg["ent_tok_n_explore"] == 2 * snap["ent_tok_n_explore"]


def test_stride_skips_measurement_without_skipping_drops():
    lp = _make_lp(entropy_measure_stride=2)
    lp._ent_stride = 2
    lg = _logits()
    lp.active[0] = True
    lp.measure[0] = True
    lp.idx_to_req[0] = "uA"

    lp.apply(lg.clone())          # _meas_n = 1 -> skipped
    s1 = E.flush_lp_counters()
    assert s1["ent_tok_n_explore"] == 0
    assert s1["perturbed"] == 1   # the drop still happened

    lp.apply(lg.clone())          # _meas_n = 2 -> measured
    s2 = E.flush_lp_counters()
    assert s2["ent_tok_n_explore"] == 1


# --------------------------------------------------------------------------
# 6. config transport parity
#
# The vLLM worker is SPAWNED, not forked, so it reimports the LP module fresh
# and loses the in-memory config. `_exp_cfg_dict` in vllm_async_server.py, mirrored
# through VERL_EXPLORATION_CFG, is the ONLY channel that reaches it. A key the LP
# reads but the server never ships is not a smaller feature: it is a knob pinned
# to its default forever, with no error and no log line anywhere.
#
# A regression test is the only thing that catches that, because nothing about
# a missing dict key looks wrong at runtime.


def _read_source(rel: str) -> str:
    import pathlib
    here = pathlib.Path(__file__).resolve()
    root = next(p for p in here.parents if (p / "verl" / "workers").is_dir())
    return (root / rel).read_text()


def test_worker_config_ships_every_key_the_lp_reads():
    import re

    lp = _read_source("verl/workers/rollout/exploration/entropy_dropout_lp.py")
    srv = _read_source("verl/workers/rollout/vllm_rollout/vllm_async_server.py")

    read = set(re.findall(r'cfg\.get\(\s*"([a-z_]+)"', lp))
    read |= set(re.findall(r'cfg\[\s*"([a-z_]+)"\s*\]', lp))
    assert "measure_entropy" in read, "test is not finding the LP's config reads"

    m = re.search(r"_exp_cfg_dict\s*=\s*\{(.*?)\n            \}", srv, re.S)
    assert m, "could not locate _exp_cfg_dict in vllm_async_server.py"
    shipped = set(re.findall(r'"([a-z_]+)"\s*:', m.group(1)))

    missing = sorted(read - shipped)
    assert not missing, (
        "these keys are read by the logits processor but never shipped to the "
        f"spawned worker, so they are permanently stuck at their defaults: {missing}"
    )


def test_worker_config_keys_exist_on_the_dataclass():
    """A shipped key that the dataclass does not define reads as its getattr
    fallback forever -- the same silent failure from the other direction."""
    import dataclasses
    import re

    from verl.workers.config.rollout import ExplorationConfig

    srv = _read_source("verl/workers/rollout/vllm_rollout/vllm_async_server.py")
    m = re.search(r"_exp_cfg_dict\s*=\s*\{(.*?)\n            \}", srv, re.S)
    shipped = set(re.findall(r'"([a-z_]+)"\s*:', m.group(1)))

    fields = {f.name for f in dataclasses.fields(ExplorationConfig)}
    # "enable" is set literally to True inside the guarded branch, not read off
    # the config object, so it is legitimately absent from the field crosscheck.
    unknown = sorted(shipped - fields - {"enable"})
    assert not unknown, f"shipped keys with no ExplorationConfig field: {unknown}"


def test_entropy_instrumentation_defaults_on_for_exploration_runs():
    """The instrumentation must be on by default wherever exploration runs.

    Guards the three places a default lives; they drift apart silently because
    Hydra reads the yaml, the dataclass backs it, and the server ships it.
    """
    import re

    from verl.workers.config.rollout import ExplorationConfig

    cfg = ExplorationConfig()
    assert cfg.measure_entropy is True
    assert cfg.entropy_measure_stride == 1
    assert cfg.entropy_post_drop_window > 0

    yaml = _read_source("verl/trainer/config/rollout/rollout.yaml")
    assert re.search(r"^\s+measure_entropy:\s*true\s*$", yaml, re.M), (
        "rollout.yaml must default measure_entropy true; Hydra reads the yaml, "
        "not the dataclass, so a false here silently wins"
    )
