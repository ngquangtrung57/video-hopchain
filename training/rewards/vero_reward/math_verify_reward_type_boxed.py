"""Reward helpers that require explicit reward types and support numeric tolerance."""

from __future__ import annotations

import ast
import re
from typing import Any

import sympy
from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
from .text_normalization import normalize_text_for_match

__all__ = [
    "format_reward",
    "acc_reward",
]


_PARSE_TIMEOUT = 10
_VERIFY_TIMEOUT = 10
_NUMERIC_DEFAULT_REL_TOL = 1e-2  # 1% relative tolerance for numeric eq when extra_info.tolerance is unset
_NUMERIC_ABS_TOL = 1e-6


def _numeric_short_circuit(predicted: str, truth: str, rel_tol: float = _NUMERIC_DEFAULT_REL_TOL) -> bool:
    """Return True when predicted ~= truth as floats (relative or absolute).

    Catches the common case where math_verify's strict mode rejects
    `0.89` vs `0.8882` or `\\frac{8}{3}` vs `2.6667` despite numerical
    equivalence. Cheap and side-effect-free.
    """
    try:
        a = float(predicted.strip())
        b = float(truth.strip())
    except (TypeError, ValueError):
        return False
    diff = abs(a - b)
    if diff <= _NUMERIC_ABS_TOL:
        return True
    if abs(b) > _NUMERIC_ABS_TOL and diff <= rel_tol * abs(b):
        return True
    return False

_FORMAT_PATTERN = re.compile(
    r"<think>(?:(?!<think>|</think>).)*</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL,
)
_THINK_PATTERN = re.compile(r"<think>(?P<think>(?:(?!<think>|</think>).)*)</think>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_REWARD_TYPES = {
    "string_match",
    "multiple_choice",
    "numeric",
    "list_string_match",
    "counting",
    "search",
}

_FORMAT_BOXED_WEIGHT = 0.5

_NUMERIC_EXTRACTION_TARGETS = [
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
]


def _safe_truncate(text: Any, limit: int = 200) -> str:
    try:
        rendered = str(text)
    except Exception:
        rendered = "<unprintable>"
    if len(rendered) > limit:
        return rendered[: limit - 3] + "..."
    return rendered


def _log_timeout(context: str, truth: Any, pred: Any) -> None:
    truth_preview = _safe_truncate(truth)
    pred_preview = _safe_truncate(pred)
    print(f"[math_verify timeout] {context} | truth={truth_preview!r} | pred={pred_preview!r}")


def _is_single_letter_choice(text: str) -> bool:
    trimmed = text.strip()
    return len(trimmed) == 1 and trimmed.isalpha()


def _has_single_tag_pair(text: str, open_tag: str, close_tag: str) -> bool:
    return text.count(open_tag) == 1 and text.count(close_tag) == 1


def _extract_answer(predict_str: str) -> str:
    answer_match = _ANSWER_PATTERN.search(predict_str)
    candidate = answer_match.group(1).strip() if answer_match else predict_str.strip()
    return candidate


def _extract_boxed_contents(text: str) -> list[str]:
    if not text:
        return []

    contents: list[str] = []
    idx = 0
    while True:
        start = text.find(r"\boxed", idx)
        if start == -1:
            break
        cursor = start + len(r"\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            idx = cursor
            continue
        cursor += 1
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            content = text[content_start : cursor - 1].strip()
            content = _strip_tex_text_wrapper(content)
            contents.append(content)
            idx = cursor
        else:
            break
    return contents


def _extract_last_boxed_bracket_content(text: str) -> str | None:
    """Return payload from the last malformed '\\boxed[...]' block, if present."""
    if not text:
        return None

    idx = 0
    last_content: str | None = None
    while True:
        start = text.find(r"\boxed", idx)
        if start == -1:
            break
        cursor = start + len(r"\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        if text[cursor] != "[":
            idx = cursor + 1
            continue
        cursor += 1
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            cursor += 1
        if depth == 0:
            last_content = text[content_start : cursor - 1].strip()
            idx = cursor
        else:
            break
    return last_content


def _strip_tex_text_wrapper(text: str) -> str:
    if not text:
        return text
    stripped = text.lstrip()
    if not stripped.startswith(r"\text"):
        return text
    idx = len(r"\text")
    while idx < len(stripped) and stripped[idx].isspace():
        idx += 1
    if idx >= len(stripped) or stripped[idx] != "{":
        return text
    idx += 1
    depth = 1
    content_start = idx
    while idx < len(stripped) and depth > 0:
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        idx += 1
    if depth != 0:
        return text
    content = stripped[content_start : idx - 1].strip()
    trailing = stripped[idx:].strip()
    if trailing:
        return text
    return content


def _extract_tolerance(extra_info: Any) -> float | None:
    if not isinstance(extra_info, dict):
        return None
    tolerance = extra_info.get("tolerance")
    if tolerance is None:
        return None
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        return None
    if tolerance_value < 0:
        return None
    return tolerance_value


def _extract_reward_type(extra_info: Any) -> str | None:
    if not isinstance(extra_info, dict):
        return None
    reward_type = extra_info.get("reward_type")
    if not isinstance(reward_type, str):
        return None
    reward_type = reward_type.strip().lower()
    allowed = {
        "string_match",
        "multiple_choice",
        "numeric",
        "list_string_match",
        "counting",
        "search",
    }
    if reward_type in allowed:
        return reward_type
    return None


def _string_match_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
    normalized_pred = _normalize_text_for_match(stripped_predicted)
    normalized_truth = _normalize_text_for_match(truth)
    if not normalized_pred or not normalized_truth:
        return 0.0
    return 1.0 if normalized_pred == normalized_truth else 0.0


def _multiple_choice_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
    try:
        parsed = parse(
            truth,
            extraction_config=[
                StringExtractionConfig(
                    strings=(
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "J",
                        "K",
                        "L",
                        "M",
                        "N",
                        "O",
                        "P",
                        "Q",
                        "R",
                        "S",
                        "T",
                        "U",
                        "V",
                        "W",
                        "X",
                        "Y",
                        "Z",
                    )
                )
            ],
            parsing_timeout=_PARSE_TIMEOUT,
        )
    except TimeoutException:
        _log_timeout("multiple_choice", truth, stripped_predicted)
        parsed = None
    except Exception:
        parsed = None

    if parsed is None:
        parsed_values = []
    elif isinstance(parsed, (list, tuple, set)):
        parsed_values = list(parsed)
    else:
        parsed_values = [parsed]

    normalized_truth = str(parsed_values[0]) if parsed_values else truth
    return _string_match_reward(stripped_predicted, normalized_truth, **_)


def _default_acc_reward(
    stripped_predicted: str, truth: str, use_boxed: bool = False, extra_info=None  # noqa: ARG001
) -> float:

    if stripped_predicted.strip().lower() == truth.strip().lower():
        return 1.0

    if _numeric_short_circuit(stripped_predicted, truth):
        return 1.0

    truth_for_parse = truth

    try:
        gold_parsed = parse(truth_for_parse, parsing_timeout=_PARSE_TIMEOUT)
        mcq_mode = False
        if not gold_parsed and _is_single_letter_choice(truth):
            gold_parsed = parse(
                truth,
                extraction_config=[StringExtractionConfig()],
                parsing_timeout=_PARSE_TIMEOUT,
            )
            mcq_mode = bool(gold_parsed)

        pred_parsed = parse(stripped_predicted, parsing_timeout=_PARSE_TIMEOUT)
        if not pred_parsed and mcq_mode:
            pred_parsed = parse(
                stripped_predicted,
                extraction_config=[StringExtractionConfig()],
                parsing_timeout=_PARSE_TIMEOUT,
            )

        gold_target = gold_parsed if gold_parsed else truth
        pred_target = pred_parsed if pred_parsed else stripped_predicted
    except Exception:
        gold_parsed = None
        pred_parsed = None
        gold_target = truth
        pred_target = stripped_predicted

    try:
        verified = verify(
            gold_target, 
            pred_target, 
            float_rounding=6,
            strict=True,
            allow_set_relation_comp=False,
            timeout_seconds=_VERIFY_TIMEOUT
            )
    except TimeoutException:
        _log_timeout("default_verify", truth, stripped_predicted)
        verified = False
    except Exception:
        verified = False

    return 1.0 if verified else 0.0


def _normalize_text_for_match(text: Any) -> str:
    return normalize_text_for_match(text)


def _coerce_truth_to_list(truth: Any) -> list[str]:
    if isinstance(truth, (list, tuple, set)):
        return [str(item) for item in truth]
    if isinstance(truth, str):
        candidate = truth.strip()
        if not candidate:
            return []
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
        except Exception:
            pass
        return [candidate]
    return [str(truth)]


def _list_string_match_reward(stripped_predicted: str, truth: str, **_: Any) -> float:
    normalized_pred = _normalize_text_for_match(stripped_predicted)
    if not normalized_pred:
        return 0.0

    truth_items = _coerce_truth_to_list(truth)
    if not truth_items:
        return 0.0

    for truth_item in truth_items:
        if normalized_pred == _normalize_text_for_match(truth_item):
            return 1.0
    return 0.0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        pass
    try:
        expr = sympy.sympify(value)
        if expr.is_real or expr.is_number:
            evaluated = expr.evalf()
            return float(evaluated)
    except Exception:
        pass
    return None


def _numeric_candidates(text: str) -> list[float]:
    candidates: list[float] = []

    direct = _coerce_float(text)
    if direct is not None:
        candidates.append(direct)

    try:
        parsed = parse(text, _NUMERIC_EXTRACTION_TARGETS, parsing_timeout=_PARSE_TIMEOUT)
    except TimeoutException:
        _log_timeout("numeric_parse", text, text)
        parsed = None
    except Exception:
        parsed = None

    parsed_values: list[Any]
    if parsed is None:
        parsed_values = []
    elif isinstance(parsed, (list, tuple, set)):
        parsed_values = list(parsed)
    else:
        parsed_values = [parsed]

    for value in parsed_values:
        coerced = _coerce_float(value)
        if coerced is not None:
            candidates.append(coerced)

    unique_candidates: list[float] = []
    for val in candidates:
        if not any(abs(val - existing) <= 1e-12 for existing in unique_candidates):
            unique_candidates.append(val)
    return unique_candidates


def grade_numeric(gold: str, pred: str, precision: int = 6, tolerance: float | None = None) -> int:
    """
    Returns:
        1 if pred matches gold numerically,
        0 if it does not,
       -1 if parsing/comparison failed.
    """

    if tolerance is not None:
        try:
            tol = float(tolerance)
        except (TypeError, ValueError):
            return -1
        if tol < 0:
            return -1

        gold_candidates = _numeric_candidates(gold)
        pred_candidates = _numeric_candidates(pred)
        if not gold_candidates or not pred_candidates:
            return -1

        for gold_value in gold_candidates:
            for pred_value in pred_candidates:
                diff = abs(gold_value - pred_value)
                if abs(gold_value) <= 1e-12:
                    if diff <= tol:
                        return 1
                elif diff <= tol * abs(gold_value):
                    return 1
        return 0

    try:
        gold_parsed = parse(
            gold, _NUMERIC_EXTRACTION_TARGETS, parsing_timeout=_PARSE_TIMEOUT
        )
        pred_parsed = parse(
            pred, _NUMERIC_EXTRACTION_TARGETS, parsing_timeout=_PARSE_TIMEOUT
        )
    except TimeoutException:
        _log_timeout("numeric_parse_verify", gold, pred)
        return -1
    except Exception:
        return -1

    if gold_parsed is None or pred_parsed is None:
        return -1

    try:
        verified = verify(
            gold_parsed,
            pred_parsed,
            float_rounding=precision,
            strict=True,
            allow_set_relation_comp=False,
            timeout_seconds=_VERIFY_TIMEOUT,
        )
    except TimeoutException:
        _log_timeout("numeric_verify", gold, pred)
        return -1
    except Exception:
        return -1

    return 1 if verified else 0


def _numeric_reward(stripped_predicted: str, truth: str, *, extra_info: Any | None = None, **_: Any) -> float:
    # Cheap exact/near-equal short-circuit before invoking math_verify (which is
    # strict and can timeout on long CoT). Catches:
    #   - identical strings
    #   - rounded decimals (`0.89` vs `0.8882`) within 1% relative tolerance
    #   - fractions vs decimals (`\frac{8}{3}` vs `2.6667`) via float() coercion
    if stripped_predicted.strip() == truth.strip():
        return 1.0
    # Resolve extra_info.tolerance before the short-circuit so a tighter tolerance is
    # honoured; an unset tolerance falls back to _NUMERIC_DEFAULT_REL_TOL.
    tolerance = _extract_tolerance(extra_info)
    if tolerance is None:
        tolerance = _NUMERIC_DEFAULT_REL_TOL
    if _numeric_short_circuit(stripped_predicted, truth, rel_tol=tolerance):
        return 1.0
    result = grade_numeric(truth, stripped_predicted, precision=6, tolerance=tolerance)
    return 1.0 if result == 1 else 0.0


def format_reward(predict_str: str, extra_info: Any | None = None) -> float:
    if not predict_str:
        return 0.0

    if not _has_single_tag_pair(predict_str, "<think>", "</think>"):
        return 0.0
    if not _has_single_tag_pair(predict_str, "<answer>", "</answer>"):
        return 0.0

    if not re.fullmatch(_FORMAT_PATTERN, predict_str):
        return 0.0

    # Require non-whitespace reasoning content in the think block.
    think_match = _THINK_PATTERN.search(predict_str)
    if not think_match or not think_match.group("think").strip():
        return 0.0

    answer_text = _extract_answer(predict_str).strip()
    if not answer_text:
        return 0.0

    reward_type = _extract_reward_type(extra_info)
    if reward_type in _BOXED_REWARD_TYPES:
        boxed_values = _extract_boxed_contents(answer_text)
        if len(boxed_values) == 1 and boxed_values[0]:
            return 1.0
        return 1.0 - _FORMAT_BOXED_WEIGHT
    return 1.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = False, extra_info=None) -> float:  # noqa: ARG001
    raw_predicted = _extract_answer(predict_str)
    stripped_predicted = raw_predicted.strip()
    reward_type = _extract_reward_type(extra_info)
    if not reward_type:
        print(f"[acc_reward] Missing or invalid reward_type in extra_info: {extra_info!r}")
    if reward_type in _BOXED_REWARD_TYPES:
        boxed_values = _extract_boxed_contents(stripped_predicted)
        if boxed_values and boxed_values[-1]:
            stripped_predicted = boxed_values[-1].strip()
        elif r"\boxed" in stripped_predicted:
            malformed_bracket = _extract_last_boxed_bracket_content(stripped_predicted)
            if malformed_bracket:
                stripped_predicted = malformed_bracket
    if not stripped_predicted:
        return 0.0

    truth = ground_truth.strip()
    reward_handlers = {
        "string_match": _string_match_reward,
        "multiple_choice": _multiple_choice_reward,
        "numeric": _numeric_reward,
        "list_string_match": _list_string_match_reward,
        "counting": _string_match_reward,
        "search": _string_match_reward,
    }
    handler = reward_handlers.get(reward_type)
    if not handler:
        return _default_acc_reward(stripped_predicted, truth, use_boxed=use_boxed, extra_info=extra_info)

    try:
        return handler(stripped_predicted, truth, extra_info=extra_info)
    except Exception:
        return 0.0
