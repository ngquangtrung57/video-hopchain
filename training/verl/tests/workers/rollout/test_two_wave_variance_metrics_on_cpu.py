"""CPU tests for the two-wave variance instrumentation.

The two-wave design exists to manufacture reward variance in prompt groups that
would otherwise give GRPO a zero advantage. Measuring whether it works has four
failure modes that all produce a plausible-looking number, and each one is pinned
here:

1. ``group_var_after`` alone is not evidence. Ordinary temperature sampling
   produces variance too. The ANOVA split (``between_share``) and the Bernoulli
   null (``var_null``) separate the wave contrast from resampling noise, and both
   are exact identities that a sign slip would quietly break.
2. Reward variance is a 0/1 OUTCOME measure. A group can collapse to one
   behaviour with its reward variance intact. The common-prefix (``lcp``) fields
   measure where the group actually fans out, and they carry a -1.0 "not
   computable" sentinel that must never be averaged as if it were a real 0.
3. Any rate reported only on the triggered groups is unfalsifiable. The
   ungated ``__tw_w2_solves_new__`` supplies the control arm, so it must NOT be
   gated on ``explored``.
4. The ``rollout_n < k_explore`` bucket does not line up with the waves. The
   per-row ``__tw_wave__`` / ``__tw_perturbed__`` labels replace it, and every
   consumer must prefer them whenever they are present.

Run: pytest tests/workers/rollout/test_two_wave_variance_metrics_on_cpu.py -q
"""

import asyncio
import types

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

# FullyAsyncTrainer is @ray.remote-decorated, so the module attribute is an
# ActorClass wrapper whose methods route through Ray. Unwrap to the plain class:
# both methods under test are pure Python and take only a `self.config` read.
_Trainer = FullyAsyncTrainer.__ray_metadata__.modified_class

STATS = AgentLoopWorker._two_wave_group_stats
GRPO = _Trainer._collect_grpo_variance_metrics
ROLLOUT_METRICS = _Trainer._collect_rvrl_rollout_metrics

# The 10 group-composition keys. Their values must not move.
LEGACY_KEYS = {
    "__tw_all_correct__", "__tw_all_wrong__", "__tw_mixed__", "__tw_triggered__",
    "__tw_var_manufactured__", "__tw_group_var_after__", "__tw_cracked_all_wrong__",
    "__tw_broke_all_correct__", "__tw_w1_acc__", "__tw_w2_acc__",
}


class _Out:
    """Minimal stand-in for AgentLoopOutput as _two_wave_group_stats reads it."""

    def __init__(self, acc, ids=None, aborted=False):
        self.extra_fields = {"reward_extra_info": {"accuracy": acc}}
        self.reward_score = acc
        self.response_ids = list(ids) if ids is not None else [7, 7, 7]
        self.stop_reason = "aborted" if aborted else "stop"


def _cfg(**kw):
    base = dict(
        variance_metric="accuracy",
        variance_threshold=0.0,
        explore_max_mean=1.0,
        anchor_fraction=0.5,
        two_wave_toggle_enable=False,
        measure_entropy=True,
    )
    base.update(kw)
    return types.SimpleNamespace(exploration=types.SimpleNamespace(**base))


def _stats(w1_accs, w2_accs, explored=True, w1_ids=None, w2_ids=None):
    w1 = [_Out(a, None if w1_ids is None else w1_ids[i]) for i, a in enumerate(w1_accs)]
    w2 = [_Out(a, None if w2_ids is None else w2_ids[i]) for i, a in enumerate(w2_accs)]
    return STATS(None, w1, w2, explored, _cfg())


def _popvar(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


# --------------------------------------------------------------------------
# 1. Variance structure: the ANOVA split and the Bernoulli null
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "w1,w2",
    [
        ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        ([1.0, 0.0], [0.25, 0.75, 1.0]),
        ([0.3, 0.9, 0.1, 0.5], [0.2, 0.8]),
    ],
)
def test_anova_split_is_an_exact_identity(w1, w2):
    """total = within + between, so between_share is well defined and in [0, 1].

    Guards the size weighting: an unweighted `between` (plain variance of the two
    wave means) is right only when the waves are equal size, and anchor_fraction
    makes them unequal for most settings of n.
    """
    s = _stats(w1, w2)
    total = s["__tw_group_var_after__"]
    n1, n2, n = len(w1), len(w2), len(w1) + len(w2)
    m = sum(w1 + w2) / n
    within = (n1 * _popvar(w1) + n2 * _popvar(w2)) / n
    between = (n1 * (sum(w1) / n1 - m) ** 2 + n2 * (sum(w2) / n2 - m) ** 2) / n

    assert total == pytest.approx(within + between, abs=1e-12)
    assert s["__tw_between_share__"] == pytest.approx(between / total, abs=1e-9)
    assert 0.0 <= s["__tw_between_share__"] <= 1.0


def test_between_share_is_one_when_each_wave_is_internally_flat():
    """The signature of a working perturbation: wave-1 flat, wave-2 displaced."""
    s = _stats([0.0, 0.0], [1.0, 1.0])
    assert s["__tw_between_share__"] == pytest.approx(1.0)
    assert s["__tw_group_var_after__"] == pytest.approx(0.25)


def test_between_share_is_zero_when_the_wave_means_coincide():
    """Variance that is only resampling noise must not be credited to the design."""
    s = _stats([0.0, 1.0], [1.0, 0.0])
    assert s["__tw_group_var_after__"] > 0.0
    assert s["__tw_between_share__"] == pytest.approx(0.0, abs=1e-12)


def test_between_share_is_zero_not_nan_on_a_fully_degenerate_group():
    s = _stats([1.0, 1.0], [1.0, 1.0])
    assert s["__tw_group_var_after__"] == 0.0
    assert s["__tw_between_share__"] == 0.0


def test_var_null_is_exactly_zero_on_a_degenerate_wave1():
    """The triggered population is degenerate by construction, so on it the whole
    of group_var_after is excess over the null. That is what makes the causal
    claim clean, and it holds only if var_null is exactly 0 here."""
    for accs in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]):
        s = _stats(accs, [1.0, 0.0, 1.0])
        assert s["__tw_var_null__"] == 0.0


def test_var_null_uses_the_finite_sample_bernoulli_expectation():
    """p(1-p)(N-1)/N, not p(1-p). At n=8 the correction is 12.5% -- large enough
    that dropping it would show up as a persistent positive var_gain on the
    untriggered control arm and be misread as an effect."""
    s = _stats([1.0, 0.0], [1.0, 0.0])  # p = 0.5, N = 4
    assert s["__tw_var_null__"] == pytest.approx(0.25 * 3 / 4)


# --------------------------------------------------------------------------
# 2. Behavioural divergence (common prefix against the wave-1 reference)
# --------------------------------------------------------------------------


def test_lcp_measures_the_divergence_point_from_the_wave1_reference():
    ref = [1, 2, 3, 4, 5, 6]
    s = _stats(
        [0.0, 0.0],
        [0.0, 0.0],
        w1_ids=[ref, [1, 2, 3, 9, 9, 9]],            # diverges at 3
        w2_ids=[[1, 9, 9, 9], [1, 2, 9, 9]],          # diverges at 1 and 2
    )
    assert s["__tw_lcp_w1__"] == pytest.approx(3.0)
    assert s["__tw_lcp_w2__"] == pytest.approx(1.5)
    # Negative delta = the perturbation makes wave-2 leave the anchor earlier
    # than sampling noise alone does. This sign is the whole point of the metric.
    assert s["__tw_lcp_w2__"] - s["__tw_lcp_w1__"] < 0


def test_lcp_reference_is_excluded_from_its_own_mean():
    """The reference has a full-length prefix match with itself. Including it
    inflates lcp_w1 toward the response length and would make the wave-2 delta
    look negative on every group, perturbed or not."""
    ref = [1, 2, 3, 4, 5, 6, 7, 8]
    s = _stats([0.0, 0.0], [0.0], w1_ids=[ref, [1, 2, 9, 9]], w2_ids=[[1, 2, 9, 9]])
    assert s["__tw_lcp_w1__"] == pytest.approx(2.0)  # not (8 + 2) / 2


def test_lcp_is_the_negative_sentinel_when_not_computable():
    """A single valid wave-1 rollout leaves no non-reference row to average, and
    0 is a legitimate LCP value, so the sentinel has to be negative."""
    s = _stats([0.0], [0.0, 0.0], w1_ids=[[1, 2, 3]], w2_ids=[[9], [9]])
    assert s["__tw_lcp_w1__"] == -1.0
    assert s["__tw_lcp_w2__"] == 0.0  # computable, and genuinely zero


def test_lcp_never_scans_past_the_cap():
    from verl.experimental.agent_loop.agent_loop import _TW_LCP_CAP

    long = list(range(_TW_LCP_CAP + 500))
    s = _stats([0.0, 0.0], [0.0], w1_ids=[long, long], w2_ids=[long])
    assert s["__tw_lcp_w1__"] == float(_TW_LCP_CAP)
    assert s["__tw_lcp_w2__"] == float(_TW_LCP_CAP)


# --------------------------------------------------------------------------
# 3. Coverage and the control arm
# --------------------------------------------------------------------------


def test_solves_new_is_not_gated_on_the_trigger():
    """cracked_all_wrong is gated on `triggered` and therefore has no control
    rate. __tw_w2_solves_new__ is the ungated sibling: an untriggered group whose
    (unperturbed) second wave finds a solution the first wave missed is exactly
    the baseline that the triggered rate has to beat."""
    s = _stats([0.0, 0.0], [1.0, 0.0], explored=False)
    assert s["__tw_triggered__"] is False
    assert s["__tw_cracked_all_wrong__"] is False  # legacy field stays gated
    assert s["__tw_w2_solves_new__"] is True       # new field does not


def test_solves_new_is_false_when_wave1_already_solved_it():
    s = _stats([1.0, 0.0], [1.0, 1.0], explored=True)
    assert s["__tw_w2_solves_new__"] is False


def test_pass_at_n_fields_track_the_two_denominators():
    s = _stats([0.0, 0.0], [0.0, 1.0])
    assert s["__tw_pass_any_w1__"] is False
    assert s["__tw_pass_any__"] is True  # coverage_gain = 1 - 0 for this group


# --------------------------------------------------------------------------
# 4. Contracts the downstream aggregation depends on
# --------------------------------------------------------------------------


def test_key_set_is_identical_regardless_of_the_explore_decision():
    """The dispatcher stamps these onto every rollout of every prompt. A key
    present on some prompts and absent on others breaks the trainer's cross-prompt
    concat, so the two branches must agree exactly."""
    assert set(_stats([0.0, 0.0], [1.0, 1.0], explored=True)) == set(
        _stats([0.0, 1.0], [1.0, 0.0], explored=False)
    )


def test_every_new_field_is_a_plain_scalar():
    """numpy object arrays of Python bool/float are what non_tensor_batch carries;
    a nested list or None would survive to the trainer and fail there instead."""
    for v in _stats([0.0, 0.0], [1.0, 1.0]).values():
        assert isinstance(v, (bool, float, int)), v


def test_earlier_fields_are_untouched_by_the_variance_fields():
    s = _stats([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], explored=True)
    assert LEGACY_KEYS <= set(s)
    assert s["__tw_all_wrong__"] is True
    assert s["__tw_all_correct__"] is False
    assert s["__tw_mixed__"] is False
    assert s["__tw_var_manufactured__"] is True
    assert s["__tw_cracked_all_wrong__"] is True
    assert s["__tw_w1_acc__"] == pytest.approx(0.0)
    assert s["__tw_w2_acc__"] == pytest.approx(1.0 / 3)


def test_aborted_rollouts_are_dropped_from_the_variance_structure_too():
    """The variance-structure fields must drop aborted rollouts like the
    composition fields do; otherwise an aborted row enters as a phantom 0 and
    manufactures between-wave variance out of a partial rollout."""
    w1 = [_Out(1.0), _Out(1.0)]
    w2 = [_Out(1.0), _Out(0.0, aborted=True)]
    s = STATS(None, w1, w2, True, _cfg())
    assert s["__tw_group_var_after__"] == 0.0
    assert s["__tw_between_share__"] == 0.0
    assert s["__tw_w2_solves_new__"] is False


# --------------------------------------------------------------------------
# 5. Per-row wave labels produced by the dispatcher
# --------------------------------------------------------------------------


class _Batch:
    def __init__(self, n):
        self.non_tensor_batch = {"raw_prompt": [f"p{i}" for i in range(n)]}
        self.meta_info = {}
        self._n = n

    def __len__(self):
        return self._n


def _dispatch(w1_accs, w2_accs, **cfgkw):
    """Drive the real _dispatch_two_wave with a stubbed _run_agent_loop."""
    n = len(w1_accs) + len(w2_accs)
    accs = list(w1_accs) + list(w2_accs)
    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    seen_explore = []

    async def _fake_run(sampling_params, trajectory, *, trace=True, **kwargs):
        i = trajectory["rollout_n"]
        seen_explore.append(
            bool((sampling_params.get("extra_args") or {}).get("rvrl_exploration", False))
        )
        return _Out(accs[i])

    worker._run_agent_loop = _fake_run
    outs = asyncio.run(
        worker._dispatch_two_wave(
            _Batch(n),
            {},
            [{"step": 0, "sample_index": 0, "rollout_n": i, "validate": False} for i in range(n)],
            set(),
            None,
            _cfg(**cfgkw),
            False,
        )
    )
    return outs, seen_explore


def test_wave_labels_follow_anchor_fraction_not_k_explore():
    """anchor_fraction, not k_explore, decides the split. The labels are stamped
    by the code that made the decision, so they cannot drift apart from it."""
    outs, _ = _dispatch([0.0, 0.0], [0.0, 0.0])
    assert [o.extra_fields["__tw_wave__"] for o in outs] == [1, 1, 2, 2]


def test_perturbed_flag_marks_exactly_the_rollouts_that_carried_the_stamp():
    """__tw_perturbed__ must agree row-for-row with the rvrl_exploration stamp
    that actually reached the sampler. Everything downstream that attributes GRPO
    advantage to the method rests on this."""
    outs, stamped = _dispatch([0.0, 0.0], [0.0, 0.0])  # degenerate wave-1 -> explore
    flags = [o.extra_fields["__tw_perturbed__"] for o in outs]
    assert flags == [False, False, True, True]
    assert stamped == flags


def test_no_row_is_marked_perturbed_when_wave1_has_variance():
    outs, stamped = _dispatch([0.0, 1.0], [0.0, 0.0])  # var > threshold -> no explore
    assert [o.extra_fields["__tw_perturbed__"] for o in outs] == [False] * 4
    assert not any(stamped)
    # The wave split is still labelled: it is an observation, not a treatment.
    assert [o.extra_fields["__tw_wave__"] for o in outs] == [1, 1, 2, 2]


def test_wave1_rows_are_never_perturbed_even_when_the_group_triggers():
    outs, _ = _dispatch([0.0] * 3, [0.0] * 3, anchor_fraction=0.5)
    for o in outs:
        if o.extra_fields["__tw_wave__"] == 1:
            assert o.extra_fields["__tw_perturbed__"] is False


# --------------------------------------------------------------------------
# 6. Trainer side: GRPO gradient reachability
# --------------------------------------------------------------------------


def _batch(adv_rows, lengths=None, **ntb):
    """adv_rows: one scalar advantage per row, broadcast over that row's tokens."""
    n = len(adv_rows)
    lengths = lengths or [4] * n
    width = max(lengths)
    adv = torch.zeros(n, width)
    mask = torch.zeros(n, width, dtype=torch.long)
    for i, (a, ln) in enumerate(zip(adv_rows, lengths)):
        adv[i, :ln] = a
        mask[i, :ln] = 1
    return types.SimpleNamespace(
        batch={"advantages": adv, "response_mask": mask},
        non_tensor_batch=dict(ntb),
    )


def _grpo(batch, k_explore=None):
    stub = types.SimpleNamespace()
    if k_explore is not None:
        stub.config = types.SimpleNamespace(
            actor_rollout_ref=types.SimpleNamespace(
                rollout=types.SimpleNamespace(
                    exploration=types.SimpleNamespace(k_explore=k_explore)
                )
            )
        )
    m = {}
    GRPO(stub, batch, m)
    return m


def test_zero_adv_group_frac_counts_groups_that_bought_nothing():
    """Two of four groups scored flat, so half the rollout budget produced no
    update. This is the number the two-wave design exists to move."""
    m = _grpo(
        _batch(
            [1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0],
            uid=["a", "a", "b", "b", "c", "c", "d", "d"],
        )
    )
    assert m["rvrl/grpo/n_groups"] == 4
    assert m["rvrl/grpo/zero_adv_group_frac"] == pytest.approx(0.5)
    assert m["rvrl/grpo/eff_sample_frac"] == pytest.approx(0.5)


def test_a_group_is_alive_if_any_single_row_carries_advantage():
    m = _grpo(_batch([0.0, 0.0, 0.0, 1e-3], uid=["a"] * 4))
    assert m["rvrl/grpo/zero_adv_group_frac"] == 0.0
    assert m["rvrl/grpo/eff_sample_frac"] == pytest.approx(0.25)


def test_grpo_block_emits_on_a_baseline_run_with_no_exploration_keys():
    """The block runs outside the exploration gate precisely so the baseline arm
    produces these numbers. A control that only exists on the treatment arm is
    not a control."""
    m = _grpo(_batch([1.0, -1.0], uid=["a", "a"]))
    assert "rvrl/grpo/zero_adv_group_frac" in m
    assert "rvrl/grpo/adv_abs_mean" in m
    assert not any(k.endswith("perturbed_share") for k in m)


def test_advantage_mass_share_uses_the_two_wave_flag_and_ignores_the_legacy_bucket():
    """Under two-wave the legacy rollout_n < k_explore bucket is INVERTED. With
    both keys present the per-row flag must win, or the metric reports the
    anchors' gradient share as the method's."""
    b = _batch(
        [4.0, 4.0, 1.0, 1.0],
        uid=["a"] * 4,
        __tw_perturbed__=[False, False, True, True],
        __rollout_n__=[0, 1, 2, 3],
    )
    m = _grpo(b, k_explore=2)  # legacy bucket would select rows 0,1 -> share 0.8
    assert m["rvrl/grpo/perturbed_row_frac"] == pytest.approx(0.5)
    assert m["rvrl/grpo/adv_mass_perturbed_share"] == pytest.approx(2.0 / 10.0)


def test_legacy_bucket_is_the_fallback_only_when_the_flag_is_absent():
    b = _batch([4.0, 4.0, 1.0, 1.0], uid=["a"] * 4, __rollout_n__=[0, 1, 2, 3])
    m = _grpo(b, k_explore=2)
    assert m["rvrl/grpo/adv_mass_perturbed_share"] == pytest.approx(8.0 / 10.0)


def test_token_weighted_share_is_actually_length_weighted():
    """Row share and token share answer different questions: a perturbed rollout
    twice as long contributes twice the gradient the optimiser integrates. If the
    weighting were dropped the two keys would be identical and the distinction
    silently lost."""
    b = _batch(
        [1.0, 1.0],
        lengths=[10, 30],
        uid=["a", "a"],
        __tw_perturbed__=[False, True],
    )
    m = _grpo(b)
    assert m["rvrl/grpo/adv_mass_perturbed_share"] == pytest.approx(0.5)
    assert m["rvrl/grpo/adv_token_mass_perturbed_share"] == pytest.approx(0.75)


def test_grpo_block_is_a_noop_without_advantages():
    b = types.SimpleNamespace(batch={"response_mask": torch.ones(2, 3)}, non_tensor_batch={})
    assert _grpo(b) == {}


# --------------------------------------------------------------------------
# 7. Trainer side: two-wave aggregation
# --------------------------------------------------------------------------


def _agg(n_rows=4, **ntb):
    """Run _collect_rvrl_rollout_metrics over a fabricated non_tensor_batch."""
    base = {
        "__rollout_n__": list(range(n_rows)),
        "__tw_all_wrong__": [True] * n_rows,
        "__tw_all_correct__": [False] * n_rows,
        "__tw_mixed__": [False] * n_rows,
    }
    base.update(ntb)
    batch = types.SimpleNamespace(batch={}, non_tensor_batch=base)
    m = {}
    ROLLOUT_METRICS(types.SimpleNamespace(), batch, m, 2)
    return m


def test_aggregation_reports_each_metric_against_its_control_arm():
    """A triggered-only rate has no control, so every variance-structure metric
    is emitted on both arms or not at all."""
    m = _agg(
        __tw_triggered__=[True, True, False, False],
        __tw_between_share__=[1.0, 1.0, 0.2, 0.2],
        __tw_w2_solves_new__=[True, True, False, False],
    )
    assert m["rvrl/two_wave/between_share_triggered"] == pytest.approx(1.0)
    assert m["rvrl/two_wave/between_share_untriggered"] == pytest.approx(0.2)
    assert m["rvrl/two_wave/solves_new_frac_triggered"] == pytest.approx(1.0)
    assert m["rvrl/two_wave/solves_new_frac_untriggered"] == pytest.approx(0.0)
    assert m["rvrl/two_wave/n_triggered_rows"] == 2
    assert m["rvrl/two_wave/n_untriggered_rows"] == 2


def test_var_gain_is_the_excess_over_the_null_on_each_arm():
    m = _agg(
        __tw_triggered__=[True, True, False, False],
        __tw_group_var_after__=[0.25, 0.25, 0.20, 0.20],
        __tw_var_null__=[0.0, 0.0, 0.1875, 0.1875],
    )
    assert m["rvrl/two_wave/var_gain_triggered"] == pytest.approx(0.25)
    # Wave-2 is unperturbed on the control arm, so this is a calibration check on
    # var_null and must sit near 0. A large value here invalidates the treated arm.
    assert m["rvrl/two_wave/var_gain_untriggered"] == pytest.approx(0.0125)


def test_lcp_aggregation_drops_the_uncomputable_sentinel():
    """-1.0 means "no reference rollout", not "diverged immediately". Averaging it
    would drag lcp_w1_mean down and flip the sign of the delta."""
    m = _agg(
        __tw_triggered__=[True, True, True, True],
        __tw_lcp_w1__=[100.0, 200.0, -1.0, -1.0],
        __tw_lcp_w2__=[40.0, 60.0, 5.0, -1.0],
    )
    assert m["rvrl/two_wave/lcp_w1_mean"] == pytest.approx(150.0)
    assert m["rvrl/two_wave/lcp_delta_triggered"] == pytest.approx(-100.0)


def test_coverage_gain_is_pass_at_n_minus_pass_at_n_anchor():
    m = _agg(
        __tw_triggered__=[True] * 4,
        __tw_pass_any__=[True, True, True, False],
        __tw_pass_any_w1__=[True, False, False, False],
    )
    assert m["rvrl/two_wave/pass_at_n_mean"] == pytest.approx(0.75)
    assert m["rvrl/two_wave/pass_at_n_anchor_mean"] == pytest.approx(0.25)
    assert m["rvrl/two_wave/coverage_gain"] == pytest.approx(0.5)


def test_best_of_group_groups_by_uid_not_by_the_constant_sample_index():
    """extra_info.index is absent from the curated parquets, so __sample_index__ is
    0 on every row and using it collapsed the whole training batch into a single
    group. uid is stamped per prompt and is the key GRPO itself groups by."""
    ntb = {
        "__rollout_n__": [0, 1, 0, 1],
        "__sample_index__": [0, 0, 0, 0],
        "uid": ["uid_a", "uid_a", "uid_b", "uid_b"],
    }
    batch = types.SimpleNamespace(
        batch={"token_level_scores": torch.tensor([[1.0], [0.0], [0.0], [1.0]])},
        non_tensor_batch=ntb,
    )
    m = {}
    ROLLOUT_METRICS(types.SimpleNamespace(), batch, m, 1)
    # Per uid: group a's best is an explore row (hit), group b's is not (miss).
    # The old constant key gave one group of four and scored 1.0.
    assert m["rvrl/best_of_group_explore_share"] == pytest.approx(0.5)


def test_best_of_group_emits_a_correctly_labelled_sibling_under_two_wave():
    ntb = {
        "__rollout_n__": [0, 1, 2, 3],
        "uid": ["uid_a"] * 4,
        "__tw_perturbed__": [False, False, True, True],
    }
    batch = types.SimpleNamespace(
        batch={"token_level_scores": torch.tensor([[0.0], [0.0], [1.0], [0.0]])},
        non_tensor_batch=ntb,
    )
    m = {}
    ROLLOUT_METRICS(types.SimpleNamespace(), batch, m, 2)
    assert m["rvrl/two_wave/best_of_group_perturbed_share"] == pytest.approx(1.0)
    # The rollout_n-bucketed key keeps its own meaning and is unaffected.
    assert m["rvrl/best_of_group_explore_share"] == pytest.approx(0.0)


def test_aggregation_is_skipped_cleanly_on_a_run_without_two_wave_keys():
    m = _agg_no_tw = {}
    batch = types.SimpleNamespace(batch={}, non_tensor_batch={"__rollout_n__": [0, 1]})
    ROLLOUT_METRICS(types.SimpleNamespace(), batch, m, 1)
    assert m["rvrl/n_explore_in_batch"] == 1
    assert not any(k.startswith("rvrl/two_wave/") for k in m)


def test_numpy_object_arrays_from_non_tensor_batch_are_handled():
    """non_tensor_batch stores these as dtype=object numpy arrays, not lists."""
    m = _agg(
        __tw_triggered__=np.array([True, True, False, False], dtype=object),
        __tw_between_share__=np.array([1.0, 1.0, 0.0, 0.0], dtype=object),
    )
    assert m["rvrl/two_wave/between_share_triggered"] == pytest.approx(1.0)
