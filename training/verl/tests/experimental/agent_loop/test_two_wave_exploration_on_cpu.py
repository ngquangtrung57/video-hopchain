# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""CPU-only unit tests for two-wave variance-targeted exploration.

Feature under test, in ``verl/experimental/agent_loop/agent_loop.py``::

    AgentLoopWorker._dispatch_two_wave(...)   # wave split + conditional explore stamp
    AgentLoopWorker._wave1_low_variance(...)  # variance gate

Importing ``verl.experimental.agent_loop.agent_loop`` pulls in torch, ray and
transformers, which is too heavy for this test and can block on CUDA
initialization. The methods under test are pure Python: they use only
``asyncio``, ``uuid.uuid4``, ``getattr`` and plain dict/list arithmetic. So this
file extracts their sources verbatim from ``agent_loop.py`` with the ``ast``
module and binds them to a light ``FakeWorker``, and the assertions run against
the real method bodies with no heavy imports.

To run against the imported class instead, replace ``_load_methods_from_source``
below with::

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    FakeWorker._dispatch_two_wave = AgentLoopWorker._dispatch_two_wave
    FakeWorker._wave1_low_variance = AgentLoopWorker._wave1_low_variance

Everything else (the ``FakeWorker`` stub of ``_run_agent_loop``, the fake batch,
the fake config) stays identical.

Run: pytest tests/experimental/agent_loop/test_two_wave_exploration_on_cpu.py -q
"""

from __future__ import annotations

import ast
import asyncio
import os
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

# --------------------------------------------------------------------------- #
# Locate the engine source file (repo-relative, no heavy import).
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# tests/experimental/agent_loop/ -> repo root is three levels up.
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_AGENT_LOOP_SRC = os.path.join(
    _REPO_ROOT, "verl", "experimental", "agent_loop", "agent_loop.py"
)

_METHOD_NAMES = (
    "_dispatch_two_wave",
    "_wave1_low_variance",
    "_two_wave_group_stats",
    "_two_wave_strategy_on",  # duty cycle; called via self. from _dispatch_two_wave
)


def _load_methods_from_source() -> dict[str, Any]:
    """Extract the method sources from ``AgentLoopWorker`` and exec them.

    Returns ``{method_name: function_object}`` holding the verbatim bodies from
    ``agent_loop.py``, executed in a namespace that provides only ``asyncio``
    and ``uuid4`` plus annotation stand-ins.
    """
    assert os.path.exists(_AGENT_LOOP_SRC), f"engine file missing: {_AGENT_LOOP_SRC}"
    src = open(_AGENT_LOOP_SRC, encoding="utf-8").read()
    tree = ast.parse(src)

    method_nodes: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AgentLoopWorker":
            for member in node.body:
                if (
                    isinstance(member, (ast.AsyncFunctionDef, ast.FunctionDef))
                    and member.name in _METHOD_NAMES
                ):
                    method_nodes[member.name] = member
            break

    missing = set(_METHOD_NAMES) - set(method_nodes)
    assert not missing, f"could not locate methods {missing} in AgentLoopWorker"

    # Namespace supplies only what the method bodies/annotations reference.
    # ``DataProto`` appears solely in _dispatch_two_wave's annotation -> any
    # object is fine. ``Any`` is used in annotations too.
    ns: dict[str, Any] = {
        "asyncio": asyncio,
        "uuid4": uuid4,
        "Any": Any,
        "DataProto": object,  # annotation-only placeholder
    }

    funcs: dict[str, Any] = {}
    for name, node in method_nodes.items():
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, filename=_AGENT_LOOP_SRC, mode="exec")
        local_ns: dict[str, Any] = {}
        exec(code, ns, local_ns)  # noqa: S102 - executing trusted in-repo source
        funcs[name] = local_ns[name]
    return funcs


_REAL_METHODS = _load_methods_from_source()


# --------------------------------------------------------------------------- #
# Test doubles.
# --------------------------------------------------------------------------- #
class _StubOutput:
    """Minimal stand-in for ``_InternalAgentLoopOutput`` (the awaited result)."""

    def __init__(self, accuracy: float | None, reward_score: float | None = None):
        # Mirror the real attribute access in _wave1_low_variance:
        #   metric == "accuracy" -> extra_fields["reward_extra_info"]["accuracy"]
        #   metric == "reward"    -> self.reward_score
        rei: dict[str, Any] = {}
        if accuracy is not None:
            rei["accuracy"] = accuracy
        self.extra_fields: dict[str, Any] = {"reward_extra_info": rei}
        self.reward_score = reward_score if reward_score is not None else accuracy
        self.stop_reason = None  # not aborted


class FakeBatch:
    """Stand-in for DataProto: only ``len()`` and ``non_tensor_batch`` are used."""

    def __init__(self, n: int, prompt_ordinal: int | None = None):
        self._n = n
        # make_task pulls per-ordinal kwargs from non_tensor_batch (skipping
        # __do_sample__). Keep one harmless object key so the dict-comp runs.
        self.non_tensor_batch: dict[str, Any] = {
            "agent_name": ["single_turn_agent"] * n,
        }
        # The toggle reads the per-prompt ordinal from meta_info. Omitted
        # (empty dict) => the dispatcher must raise when the toggle is enabled.
        self.meta_info: dict[str, Any] = {}
        if prompt_ordinal is not None:
            self.meta_info["rvrl_prompt_ordinal"] = prompt_ordinal

    def __len__(self) -> int:
        return self._n


class FakeWorker:
    """Carries the REAL two-wave methods + a recording ``_run_agent_loop`` stub.

    ``accuracies`` drives the per-ordinal stub output value; ``recorded`` holds
    the ``sampling_params`` (esp. ``extra_args``) seen per ordinal.
    """

    # Bind the verbatim engine methods.
    _dispatch_two_wave = _REAL_METHODS["_dispatch_two_wave"]
    _wave1_low_variance = _REAL_METHODS["_wave1_low_variance"]
    _two_wave_group_stats = _REAL_METHODS["_two_wave_group_stats"]
    _two_wave_strategy_on = _REAL_METHODS["_two_wave_strategy_on"]

    def __init__(self, accuracies: list[float | None], metric: str = "accuracy"):
        self._accuracies = accuracies
        self._metric = metric
        # recorded[ordinal] = the sampling_params dict passed to _run_agent_loop
        self.recorded: dict[int, dict[str, Any]] = {}

    async def _run_agent_loop(self, sample_sampling_params, trajectory_info, trace=False, **kwargs):
        ordinal = int(trajectory_info["__ordinal__"])
        # Record exactly what the dispatcher built for this ordinal.
        self.recorded[ordinal] = sample_sampling_params
        acc = self._accuracies[ordinal]
        if self._metric == "reward":
            return _StubOutput(accuracy=None, reward_score=acc)
        return _StubOutput(accuracy=acc)


def _make_trajectory_info(n: int) -> list[dict[str, Any]]:
    # _dispatch_two_wave indexes trajectory_info[i]; we tag an __ordinal__ so the
    # stub can map back to the per-ordinal accuracy. Other keys mimic the real
    # shape but are unused by the methods under test.
    return [
        {"__ordinal__": i, "sample_index": 0, "rollout_n": i, "trajectory_id": f"t{i}"}
        for i in range(n)
    ]


def _make_config(**overrides) -> SimpleNamespace:
    """Fake rollout config with an ``.exploration`` namespace (no real load)."""
    exploration = SimpleNamespace(
        enable=True,
        k_explore=4,
        two_wave_enable=True,
        anchor_fraction=0.5,
        variance_threshold=0.0,
        variance_metric="accuracy",
        explore_max_mean=1.0,
        prompt_exploration_prob=1.0,
        # Duty cycle, disabled by default.
        two_wave_toggle_enable=False,
        two_wave_toggle_period=2,
        two_wave_toggle_phase=1,
    )
    for k, v in overrides.items():
        setattr(exploration, k, v)
    return SimpleNamespace(exploration=exploration)


def _has_explore_stamp(params: dict[str, Any]) -> bool:
    extra = params.get("extra_args") or {}
    return bool(extra.get("rvrl_exploration"))


async def _run_dispatch(worker: FakeWorker, n: int, config, prompt_ordinal: int | None = None) -> list:
    batch = FakeBatch(n, prompt_ordinal=prompt_ordinal)
    trajectory_info = _make_trajectory_info(n)
    traced_indices: set = set()
    sampling_params = {"temperature": 1.0, "top_p": 1.0}
    return await worker._dispatch_two_wave(
        batch,
        sampling_params,
        trajectory_info,
        traced_indices,
        None,  # per_sample_do_sample
        config,
        False,  # validate
    )


# --------------------------------------------------------------------------- #
# F1: _dispatch_two_wave behavior, n=8, anchor_fraction=0.5 => n_anchor=4.
# --------------------------------------------------------------------------- #
def test_wave1_never_stamped_when_wave1_degenerate_all_correct():
    """(i) wave-1 (0..3) never stamped; (ii) all-equal wave-1 -> wave-2 stamped."""
    worker = FakeWorker(accuracies=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    config = _make_config()  # explore_max_mean=1.0 admits mean 1.0
    outputs = asyncio.run(_run_dispatch(worker, n=8, config=config))

    assert len(outputs) == 8
    # (i) wave-1 ordinals 0..3 never carry rvrl_exploration.
    for ordinal in range(4):
        assert not _has_explore_stamp(worker.recorded[ordinal]), (
            f"wave-1 ordinal {ordinal} should NOT be stamped"
        )
    # (ii) variance 0 (all 1.0) and mean<=explore_max_mean -> wave-2 stamped.
    for ordinal in range(4, 8):
        assert _has_explore_stamp(worker.recorded[ordinal]), (
            f"wave-2 ordinal {ordinal} SHOULD be stamped (degenerate wave-1)"
        )
        # uuid present alongside the marker.
        assert "rvrl_req_uuid" in worker.recorded[ordinal]["extra_args"]
    # passive metadata stamped on every output.
    for o in outputs:
        assert o.extra_fields["__two_wave_explored__"] is True
        assert o.extra_fields["__wave1_var__"] == 0.0


def test_wave2_stamped_when_wave1_all_zero():
    """All-equal wave-1 (all 0.0) -> variance 0 -> wave-2 stamped."""
    worker = FakeWorker(accuracies=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    config = _make_config()
    outputs = asyncio.run(_run_dispatch(worker, n=8, config=config))

    for ordinal in range(4):
        assert not _has_explore_stamp(worker.recorded[ordinal])
    for ordinal in range(4, 8):
        assert _has_explore_stamp(worker.recorded[ordinal])
    for o in outputs:
        assert o.extra_fields["__two_wave_explored__"] is True


def test_wave2_not_stamped_when_wave1_mixed():
    """(iii) mixed wave-1 ([1,0,1,0] -> var>0) -> wave-2 NOT stamped."""
    worker = FakeWorker(accuracies=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    config = _make_config()
    outputs = asyncio.run(_run_dispatch(worker, n=8, config=config))

    # No ordinal — wave-1 or wave-2 — is stamped.
    for ordinal in range(8):
        assert not _has_explore_stamp(worker.recorded[ordinal]), (
            f"ordinal {ordinal} should NOT be stamped when wave-1 has variance"
        )
    for o in outputs:
        assert o.extra_fields["__two_wave_explored__"] is False
        assert o.extra_fields["__wave1_var__"] > 0.0


# --------------------------------------------------------------------------- #
# (iv) validation gate: two-wave disabled on validate. Test the gate condition
# from generate_sequences directly (no heavy scaffolding required).
# --------------------------------------------------------------------------- #
def _generate_sequences_two_wave_gate(config, validate: bool, batch_len: int) -> bool:
    """Re-express the exact ``two_wave`` gate boolean from generate_sequences.

    Mirrors agent_loop.py::generate_sequences:
        exploration_enabled = (not validate and exploration is not None
                               and enable and k_explore > 0)
        two_wave = (exploration_enabled and two_wave_enable
                    and not validate and len(batch) >= 2)
    """
    exploration_enabled = (
        not validate
        and getattr(config, "exploration", None) is not None
        and config.exploration.enable
        and config.exploration.k_explore > 0
    )
    return bool(
        exploration_enabled
        and getattr(config.exploration, "two_wave_enable", False)
        and not validate
        and batch_len >= 2
    )


def test_two_wave_gate_disabled_on_validate():
    """(iv) validate=True => two_wave gate False (two-wave off during validation)."""
    config = _make_config()
    assert _generate_sequences_two_wave_gate(config, validate=True, batch_len=8) is False
    # Sanity: same config, validate=False => gate True.
    assert _generate_sequences_two_wave_gate(config, validate=False, batch_len=8) is True


def test_two_wave_gate_off_when_n_lt_2():
    """Gate also off for n<2 (single rollout has no wave to split)."""
    config = _make_config()
    assert _generate_sequences_two_wave_gate(config, validate=False, batch_len=1) is False
    assert _generate_sequences_two_wave_gate(config, validate=False, batch_len=2) is True


def test_two_wave_gate_off_when_two_wave_disabled():
    config = _make_config(two_wave_enable=False)
    assert _generate_sequences_two_wave_gate(config, validate=False, batch_len=8) is False


# --------------------------------------------------------------------------- #
# Wave-split index math: n_anchor = max(1, min(n-1, round(n*anchor_fraction))).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n,anchor_fraction,expected_n_anchor",
    [
        (2, 0.5, 1),  # round(1.0)=1
        (7, 0.5, 4),  # round(3.5)=4 (banker's rounding: 3.5 -> 4)
        (8, 0.5, 4),  # round(4.0)=4
        (16, 0.5, 8),  # round(8.0)=8
    ],
)
def test_wave_split_index_math(n, anchor_fraction, expected_n_anchor):
    """n_anchor matches the clamp formula and both waves are non-empty."""
    n_anchor = max(1, min(n - 1, int(round(n * anchor_fraction))))
    assert n_anchor == expected_n_anchor
    assert 1 <= n_anchor <= n - 1  # never empty wave-1, never empty wave-2

    # And drive the real dispatcher: wave-1 = [0, n_anchor) is ALWAYS unstamped.
    worker = FakeWorker(accuracies=[1.0] * n)
    config = _make_config()
    asyncio.run(_run_dispatch(worker, n=n, config=config))
    for ordinal in range(n_anchor):
        assert not _has_explore_stamp(worker.recorded[ordinal])

    # Wave-2 = [n_anchor, n) is stamped ONLY when wave-1 has >=2 valid scores
    # (variance is otherwise undefined and _wave1_low_variance fail-safes to
    # no-explore). For n=2 -> n_anchor=1 -> single-rollout wave-1 -> no stamp;
    # this is correct degenerate-edge behavior, not a bug.
    expect_wave2_stamp = n_anchor >= 2
    for ordinal in range(n_anchor, n):
        assert _has_explore_stamp(worker.recorded[ordinal]) is expect_wave2_stamp


# --------------------------------------------------------------------------- #
# Direct table test of _wave1_low_variance contract.
# --------------------------------------------------------------------------- #
def _wave1_outputs(accuracies, metric="accuracy", aborted_mask=None):
    outs = []
    for idx, acc in enumerate(accuracies):
        if metric == "reward":
            o = _StubOutput(accuracy=None, reward_score=acc)
        else:
            o = _StubOutput(accuracy=acc)
        if aborted_mask and aborted_mask[idx]:
            o.stop_reason = "aborted"
        outs.append(o)
    return outs


def test_wave1_low_variance_all_equal_explores():
    """All-equal accuracies => variance 0 => explore True, var 0.0."""
    worker = FakeWorker(accuracies=[])  # unused here
    config = _make_config()
    explore, var = worker._wave1_low_variance(_wave1_outputs([1.0, 1.0, 1.0, 1.0]), config)
    assert explore is True
    assert var == 0.0

    explore0, var0 = worker._wave1_low_variance(_wave1_outputs([0.0, 0.0, 0.0]), config)
    assert explore0 is True
    assert var0 == 0.0


def test_wave1_low_variance_mixed_no_explore():
    """Mixed accuracies => variance > threshold => explore False."""
    worker = FakeWorker(accuracies=[])
    config = _make_config()
    explore, var = worker._wave1_low_variance(_wave1_outputs([1.0, 0.0, 1.0, 0.0]), config)
    assert explore is False
    assert var == pytest.approx(0.25)  # population var of [1,0,1,0]


def test_wave1_low_variance_fewer_than_two_valid():
    """<2 valid scores (most aborted/None) => fail-safe (False, 0.0)."""
    worker = FakeWorker(accuracies=[])
    config = _make_config()
    # Three of four aborted -> only one valid score remains.
    outs = _wave1_outputs([1.0, 1.0, 1.0, 1.0], aborted_mask=[True, True, True, False])
    explore, var = worker._wave1_low_variance(outs, config)
    assert explore is False
    assert var == 0.0

    # All accuracies None -> zero valid scores.
    none_outs = _wave1_outputs([None, None, None, None])
    explore2, var2 = worker._wave1_low_variance(none_outs, config)
    assert explore2 is False
    assert var2 == 0.0


def test_wave1_low_variance_reward_metric_path():
    """metric='reward' pulls from .reward_score, not reward_extra_info."""
    worker = FakeWorker(accuracies=[])
    config = _make_config(variance_metric="reward")
    # All-equal reward scores => explore.
    explore, var = worker._wave1_low_variance(
        _wave1_outputs([0.5, 0.5, 0.5], metric="reward"), config
    )
    assert explore is True
    assert var == 0.0
    # Mixed reward scores => no explore.
    explore_m, var_m = worker._wave1_low_variance(
        _wave1_outputs([0.9, 0.1, 0.9], metric="reward"), config
    )
    assert explore_m is False
    assert var_m > 0.0


def test_wave1_low_variance_explore_max_mean_gate():
    """explore_max_mean=0.5: all-correct (mean 1.0) skipped, all-wrong still explores."""
    worker = FakeWorker(accuracies=[])
    config = _make_config(explore_max_mean=0.5)
    # All-correct: variance 0 but mean 1.0 > 0.5 => explore False.
    explore_hi, var_hi = worker._wave1_low_variance(_wave1_outputs([1.0, 1.0, 1.0, 1.0]), config)
    assert explore_hi is False
    assert var_hi == 0.0
    # All-wrong: variance 0 and mean 0.0 <= 0.5 => explore True.
    explore_lo, var_lo = worker._wave1_low_variance(_wave1_outputs([0.0, 0.0, 0.0, 0.0]), config)
    assert explore_lo is True
    assert var_lo == 0.0


# --------------------------------------------------------------------------- #
# _two_wave_group_stats: group classification + exploration-effect diagnostics.
# --------------------------------------------------------------------------- #
def test_group_stats_all_correct_broken_by_wave2():
    """Degenerate all-correct wave-1; wave-2 (explored) finds a wrong => broke + variance manufactured."""
    worker = FakeWorker(accuracies=[])
    config = _make_config()
    w1 = _wave1_outputs([1.0, 1.0, 1.0, 1.0])
    w2 = _wave1_outputs([1.0, 0.0, 1.0, 1.0])  # one wrong introduced
    s = worker._two_wave_group_stats(w1, w2, explored=True, config=config)
    assert s["__tw_all_correct__"] is True
    assert s["__tw_all_wrong__"] is False
    assert s["__tw_mixed__"] is False
    assert s["__tw_triggered__"] is True
    assert s["__tw_group_var_after__"] > 0.0
    assert s["__tw_var_manufactured__"] is True
    assert s["__tw_broke_all_correct__"] is True
    assert s["__tw_cracked_all_wrong__"] is False
    assert s["__tw_w1_acc__"] == 1.0
    assert s["__tw_w2_acc__"] == pytest.approx(0.75)


def test_group_stats_all_wrong_cracked_by_wave2():
    """Degenerate all-wrong wave-1; wave-2 finds a correct => cracked + variance manufactured."""
    worker = FakeWorker(accuracies=[])
    config = _make_config()
    w1 = _wave1_outputs([0.0, 0.0, 0.0, 0.0])
    w2 = _wave1_outputs([0.0, 0.0, 1.0, 0.0])  # one correct discovered
    s = worker._two_wave_group_stats(w1, w2, explored=True, config=config)
    assert s["__tw_all_wrong__"] is True
    assert s["__tw_all_correct__"] is False
    assert s["__tw_triggered__"] is True
    assert s["__tw_var_manufactured__"] is True
    assert s["__tw_cracked_all_wrong__"] is True
    assert s["__tw_broke_all_correct__"] is False
    assert s["__tw_w1_acc__"] == 0.0
    assert s["__tw_w2_acc__"] == pytest.approx(0.25)


def test_group_stats_all_correct_robust_no_manufacture():
    """All-correct wave-1 AND wave-2 stays all-correct => no variance manufactured (self-limiting)."""
    worker = FakeWorker(accuracies=[])
    config = _make_config()
    w1 = _wave1_outputs([1.0, 1.0, 1.0, 1.0])
    w2 = _wave1_outputs([1.0, 1.0, 1.0, 1.0])  # robustly solved: perturbation changes nothing
    s = worker._two_wave_group_stats(w1, w2, explored=True, config=config)
    assert s["__tw_all_correct__"] is True
    assert s["__tw_group_var_after__"] == 0.0
    assert s["__tw_var_manufactured__"] is False
    assert s["__tw_broke_all_correct__"] is False


def test_group_stats_mixed_not_triggered():
    """Mixed wave-1 => mixed True, not triggered (explored False), nothing manufactured."""
    worker = FakeWorker(accuracies=[])
    config = _make_config()
    w1 = _wave1_outputs([1.0, 0.0, 1.0, 0.0])
    w2 = _wave1_outputs([1.0, 1.0, 0.0, 0.0])
    s = worker._two_wave_group_stats(w1, w2, explored=False, config=config)
    assert s["__tw_mixed__"] is True
    assert s["__tw_all_correct__"] is False
    assert s["__tw_all_wrong__"] is False
    assert s["__tw_triggered__"] is False
    assert s["__tw_var_manufactured__"] is False
    assert s["__tw_cracked_all_wrong__"] is False
    assert s["__tw_broke_all_correct__"] is False


def test_group_stats_aborted_rollouts_filtered():
    """Aborted wave-2 rollouts are dropped from the wave-2 accuracy / crack check."""
    worker = FakeWorker(accuracies=[])
    config = _make_config()
    w1 = _wave1_outputs([0.0, 0.0, 0.0, 0.0])
    # The only 'correct' wave-2 rollout is aborted -> must NOT count as a crack.
    w2 = _wave1_outputs([0.0, 1.0, 0.0, 0.0], aborted_mask=[False, True, False, False])
    s = worker._two_wave_group_stats(w1, w2, explored=True, config=config)
    assert s["__tw_all_wrong__"] is True
    assert s["__tw_cracked_all_wrong__"] is False  # the correct one was aborted
    assert s["__tw_w2_acc__"] == 0.0


def test_dispatch_stamps_group_stats_keys():
    """End-to-end: _dispatch_two_wave stamps the __tw_* diagnostic keys on every output."""
    worker = FakeWorker(accuracies=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    config = _make_config()
    outputs = asyncio.run(_run_dispatch(worker, n=8, config=config))
    required = {
        "__tw_all_correct__", "__tw_all_wrong__", "__tw_mixed__", "__tw_triggered__",
        "__tw_var_manufactured__", "__tw_group_var_after__", "__tw_cracked_all_wrong__",
        "__tw_broke_all_correct__", "__tw_w1_acc__", "__tw_w2_acc__",
    }
    for o in outputs:
        assert required.issubset(o.extra_fields.keys())
    # wave-1 all-correct, wave-2 has a wrong => broke + manufactured.
    assert outputs[0].extra_fields["__tw_all_correct__"] is True
    assert outputs[0].extra_fields["__tw_var_manufactured__"] is True
    assert outputs[0].extra_fields["__tw_broke_all_correct__"] is True


# --------------------------------------------------------------------------- #
# F6: the two-wave duty-cycle toggle.
#
# The toggle gates the STRATEGY, not the exploration. Strategy ON => the
# variance rule decides, so high variance still explores nothing. Strategy OFF
# => plain GRPO: no wave split, no exploration stamp.
# --------------------------------------------------------------------------- #

_ALL_CORRECT_8 = [1.0] * 8
_MIXED_W1_8 = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def _toggle_cfg(**kw):
    base = dict(two_wave_toggle_enable=True, two_wave_toggle_period=2, two_wave_toggle_phase=1)
    base.update(kw)
    return _make_config(**base)


def test_toggle_disabled_runs_the_strategy_on_every_prompt():
    """Toggle off => the variance rule alone decides and the ordinal is ignored."""
    for accs, expect_stamp in ((_ALL_CORRECT_8, True), (_MIXED_W1_8, False)):
        worker = FakeWorker(accs)
        outputs = asyncio.run(_run_dispatch(worker, 8, _make_config(), prompt_ordinal=None))
        for i in range(4):
            assert not _has_explore_stamp(worker.recorded[i])
        for i in range(4, 8):
            assert _has_explore_stamp(worker.recorded[i]) is expect_stamp
        assert outputs[0].extra_fields["__two_wave_explored__"] is expect_stamp
        assert outputs[0].extra_fields["__tw_strategy_on__"] is True


@pytest.mark.parametrize("ordinal", [0, 1, 2, 3, 17, 128])
def test_toggle_period_1_runs_the_strategy_on_every_prompt(ordinal):
    """period=1 => the strategy runs on every prompt, whatever the ordinal."""
    worker = FakeWorker(_ALL_CORRECT_8)
    outputs = asyncio.run(
        _run_dispatch(worker, 8, _toggle_cfg(two_wave_toggle_period=1), prompt_ordinal=ordinal)
    )
    assert outputs[0].extra_fields["__tw_strategy_on__"] is True
    assert all(_has_explore_stamp(worker.recorded[i]) for i in range(4, 8))


@pytest.mark.parametrize(
    "ordinal,strategy_on",
    [(1, True), (2, False), (3, True), (4, False), (5, True), (100, False), (101, True)],
)
def test_toggle_period_2_alternates(ordinal, strategy_on):
    """period=2 phase=1 => odd ordinals run the strategy, even ones do not."""
    worker = FakeWorker(_ALL_CORRECT_8)
    outputs = asyncio.run(_run_dispatch(worker, 8, _toggle_cfg(), prompt_ordinal=ordinal))
    assert outputs[0].extra_fields["__tw_strategy_on__"] is strategy_on
    # wave-1 is degenerate, so the stamp tracks the toggle exactly here.
    assert all(_has_explore_stamp(worker.recorded[i]) is strategy_on for i in range(4, 8))


def test_toggle_off_is_plain_grpo_even_when_degenerate():
    """Strategy OFF + fully degenerate wave-1 => NO ordinal carries a stamp."""
    worker = FakeWorker(_ALL_CORRECT_8)
    outputs = asyncio.run(_run_dispatch(worker, 8, _toggle_cfg(), prompt_ordinal=2))
    assert all(not _has_explore_stamp(worker.recorded[i]) for i in range(8))
    assert outputs[0].extra_fields["__two_wave_explored__"] is False
    assert outputs[0].extra_fields["__tw_strategy_on__"] is False
    assert outputs[0].extra_fields["__tw_triggered__"] is False


def test_toggle_on_but_high_variance_explores_nothing():
    """The user's explicit case: the strategy ran, and the variance rule declined."""
    worker = FakeWorker(_MIXED_W1_8)
    outputs = asyncio.run(_run_dispatch(worker, 8, _toggle_cfg(), prompt_ordinal=1))
    assert outputs[0].extra_fields["__tw_strategy_on__"] is True
    assert all(not _has_explore_stamp(worker.recorded[i]) for i in range(8))
    assert outputs[0].extra_fields["__two_wave_explored__"] is False
    assert outputs[0].extra_fields["__wave1_var__"] > 0.0


def test_toggle_off_still_reports_observation_stats():
    """Strategy-off prompts keep full telemetry, so their share stays measurable."""
    on = asyncio.run(_run_dispatch(FakeWorker(_ALL_CORRECT_8), 8, _toggle_cfg(), prompt_ordinal=1))
    off = asyncio.run(_run_dispatch(FakeWorker(_ALL_CORRECT_8), 8, _toggle_cfg(), prompt_ordinal=2))
    # Degeneracy is measured on BOTH branches -- that is what makes
    # rvrl/two_wave/suppressed_by_toggle_share meaningful.
    for outs in (on, off):
        assert outs[0].extra_fields["__tw_all_correct__"] is True
        assert outs[0].extra_fields["__wave1_var__"] == 0.0
    assert off[0].extra_fields["__tw_triggered__"] is False
    assert on[0].extra_fields["__tw_triggered__"] is True


def test_toggle_branches_stamp_identical_key_sets():
    """INVARIANT: a key on one branch and not the other breaks the trainer's concat."""
    on = asyncio.run(_run_dispatch(FakeWorker(_ALL_CORRECT_8), 8, _toggle_cfg(), prompt_ordinal=1))
    off = asyncio.run(_run_dispatch(FakeWorker(_ALL_CORRECT_8), 8, _toggle_cfg(), prompt_ordinal=2))
    assert set(on[0].extra_fields.keys()) == set(off[0].extra_fields.keys())
    for outs in (on, off):
        keys = set(outs[0].extra_fields.keys())
        assert all(set(o.extra_fields.keys()) == keys for o in outs)


def test_toggle_raises_when_ordinal_missing():
    """Fail loud: a missing ordinal must not silently run the strategy everywhere."""
    worker = FakeWorker(_ALL_CORRECT_8)
    with pytest.raises(RuntimeError, match="rvrl_prompt_ordinal"):
        asyncio.run(_run_dispatch(worker, 8, _toggle_cfg(), prompt_ordinal=None))


def test_strategy_on_helper_phase_wraps():
    """phase is taken modulo period, so an out-of-range phase is still valid."""
    worker = FakeWorker(_ALL_CORRECT_8)
    cfg = _toggle_cfg(two_wave_toggle_period=3, two_wave_toggle_phase=7)  # 7 % 3 == 1
    assert worker._two_wave_strategy_on(1, cfg) is True
    assert worker._two_wave_strategy_on(4, cfg) is True
    assert worker._two_wave_strategy_on(2, cfg) is False
    assert worker._two_wave_strategy_on(3, cfg) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
