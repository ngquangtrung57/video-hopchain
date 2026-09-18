"""verl custom reward function.

Scores a rollout as 0.8 * accuracy + 0.2 * format, dispatching on
extra_info.reward_type.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boxed_reward.math_verify_reward_type_boxed import (
    acc_reward,
    format_reward,
    _extract_reward_type,
)
from reward_utils import score_sequence_answer
from math_verify import parse, verify
# lmms-eval is optional; fall back to the local MCQ extractor when it is absent.
try:
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
except Exception as _exc:  # pragma: no cover - depends on deployment layout
    import warnings as _warnings

    from reward_utils import parse_mcq as _parse_mcq

    _warnings.warn(
        f"lmms_eval unavailable ({_exc.__class__.__name__}: {_exc}); "
        "MCQ fallback extraction uses reward_utils.parse_mcq instead. "
        "Only the no-<answer>-tag fallback path is affected.",
        RuntimeWarning,
        stacklevel=2,
    )

    def extract_mcq_answer(predict_str: str) -> str:
        """Local stand-in for lmms-eval's priority-ranked MCQ extractor."""
        return _parse_mcq(predict_str)

FORMAT_WEIGHT = 0.2

# Minimum non-whitespace characters inside <think>...</think> for the format
# term to count.
FORMAT_MIN_THINK_CHARS = 100

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_THINK_CONTENT_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_boxed(text):
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def _raw_reward_type(extra_info):
    """reward_type read straight off extra_info, with no handler-table filter."""
    if not isinstance(extra_info, dict):
        return None
    reward_type = extra_info.get("reward_type")
    if not isinstance(reward_type, str):
        return None
    return reward_type.strip().lower()


def _fallback_accuracy(solution_str, ground_truth, extra_info):
    reward_type = _extract_reward_type(extra_info)
    gt = ground_truth.strip()
    boxed = _extract_boxed(solution_str)

    if reward_type == "multiple_choice":
        # Prefer \boxed content normalized to a bare letter.
        if boxed:
            letter = re.sub(r"[^A-Za-z]", "", boxed).upper()
            if len(letter) == 1 and letter.isalpha():
                return 1.0 if letter == gt.upper() else 0.0
        # lmms-eval priority-ranked extractor on the full output.
        extracted = extract_mcq_answer(solution_str)
        if extracted and extracted.upper() == gt.upper():
            return 1.0
        return 0.0

    # Non-MCQ types: require content inside \boxed{}.
    if not boxed:
        return 0.0

    if reward_type in ("numeric", "counting"):
        try:
            gold = parse(gt)
            pred = parse(boxed)
            if gold is not None and pred is not None and verify(gold, pred):
                return 1.0
        except Exception:
            pass
        return 1.0 if boxed == gt else 0.0

    if reward_type in ("string_match", "list_string_match", "search"):
        return 1.0 if boxed.lower() == gt.lower() else 0.0

    return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  sandbox_fusion_url=None, concurrent_semaphore=None):
    solution_str = str(solution_str)
    ground_truth = str(ground_truth)

    # Under VERL_THINK_PREFILL the rollout prompt already carries the opening tag, so it is
    # absent from the decoded response; prepend the same string before grading the format.
    if os.environ.get("VERL_THINK_PREFILL", "0") == "1":
        _pf = os.environ.get("VERL_THINK_PREFILL_STR", "<think>")
        if not solution_str.lstrip().startswith(_pf):
            solution_str = _pf + solution_str

    # "sequence": ground_truth is a comma-separated 1-indexed clip order, which acc_reward
    # has no handler for, so score it here and report step accuracy alongside accuracy.
    reward_type = _raw_reward_type(extra_info)
    step_accuracy = None
    if reward_type == "sequence":
        accuracy, step_accuracy = score_sequence_answer(solution_str, ground_truth)
    else:
        accuracy = acc_reward(solution_str, ground_truth, extra_info=extra_info)
    formatting = format_reward(solution_str, extra_info=extra_info)

    # Length floor on the format term.
    if formatting > 0.0:
        m = _THINK_CONTENT_RE.search(solution_str)
        think_len = len(m.group(1).strip()) if m else 0
        if think_len < FORMAT_MIN_THINK_CHARS:
            formatting = 0.0

    if accuracy == 0.0 and formatting == 0.0 and reward_type != "sequence":
        accuracy = _fallback_accuracy(solution_str, ground_truth, extra_info)

    score = (1.0 - FORMAT_WEIGHT) * accuracy + FORMAT_WEIGHT * formatting

    result = {
        "score": score,
        "accuracy": accuracy,
        "format": formatting,
    }
    if step_accuracy is not None:
        result["step_accuracy"] = step_accuracy
    return result
