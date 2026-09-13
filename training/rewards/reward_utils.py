"""Utility functions for reward computation."""

import re


def extract_boxed_answer(predict_str: str) -> str:
    """Extract the answer from \boxed{} format.

    Args:
        predict_str (str): The prediction string containing the boxed answer.

    Returns:
        str: The extracted answer from \boxed{}, or an empty string if not found.
    """
    # Find all occurrences of \boxed{
    boxed_start = "\\boxed{"
    start_indices = []

    # Find all positions where \boxed{ starts
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1

    if not start_indices:
        return ""

    # For each \boxed{ occurrence, find the matching closing brace
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1  # Position at the opening brace of \boxed{

        while pos < len(predict_str):
            char = predict_str[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    content_start = start_pos + len(boxed_start)
                    content = predict_str[content_start:pos]
                    results.append(content)
                    break
            pos += 1

    # Return the last (rightmost) match if multiple found
    return results[-1] if results else ""


def parse_mcq(predict_str: str) -> str:
    """
    Parse multiple choice answers from various formats.
    Handles formats like: "A", "A.", "A)", "(A)", "The answer is A", "A: xxx", etc.
    """
    if not predict_str or predict_str.strip() == "":
        return ""

    # Clean up the response
    response = predict_str.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)

    # Add spaces to avoid partial matches
    response = " " + response + " "

    # All possible choice letters (extend if needed)
    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]

    candidates = []

    # Pattern 1: Look for choices with parentheses e.g., (A), (B), (C), (D)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append((choice, response.rfind(f"({choice})"), "parentheses"))

    # Pattern 2: Look for choices with periods e.g., A., B., C., D.
    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append((choice, response.rfind(f"{choice}."), "period"))

    # Pattern 3: Look for choices with colons e.g., A:, B:, C:, D:
    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append((choice, response.rfind(f"{choice}:"), "colon"))

    # Pattern 4: Look for choices with right parentheses e.g., A), B), C), D)
    for choice in all_choices:
        if f"{choice})" in response:
            candidates.append((choice, response.rfind(f"{choice})"), "right_paren"))

    # Pattern 5: Look for choices with spaces after e.g., A B C D
    for choice in all_choices:
        if f"{choice} " in response:
            candidates.append((choice, response.rfind(f"{choice} "), "space"))

    # Pattern 6: Look for choices with dashes e.g., A- B- C- D-
    for choice in all_choices:
        if f"{choice}-" in response:
            candidates.append((choice, response.rfind(f"{choice}-"), "dash"))

    # Pattern 7: Look for choices with underscores e.g., A_ B_ C_ D_
    for choice in all_choices:
        if f"{choice}_" in response:
            candidates.append((choice, response.rfind(f"{choice}_"), "underscore"))

    # Pattern 8: Look for choices with equal signs e.g., A= B= C= D=
    for choice in all_choices:
        if f"{choice}=" in response:
            candidates.append((choice, response.rfind(f"{choice}="), "equals"))

    # Pattern 9: Look for common answer phrases followed by choices
    answer_phrases = [
        "the answer is", "answer is", "the correct answer is", "correct answer is",
        "the answer", "answer", "correct answer", "the correct answer",
        "the best answer is", "best answer is", "the best answer", "best answer",
        "the option is", "option is", "the correct option is", "correct option is",
        "the choice is", "choice is", "the correct choice is", "correct choice is",
        "i choose", "i select", "i pick", "my answer is", "my choice is"
    ]

    for phrase in answer_phrases:
        if phrase in response.lower():
            phrase_start = response.lower().find(phrase)
            # Look for choices after the phrase
            for choice in all_choices:
                choice_pos = response.find(choice, phrase_start)
                if choice_pos != -1:
                    candidates.append((choice, choice_pos, "phrase"))

    # Pattern 10: Look for choices at the very beginning of the response
    for choice in all_choices:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, "start"))

    # Pattern 11: Look for choices at the very end of the response
    for choice in all_choices:
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))

    # Pattern 12: Look for choices with numbers (e.g., "1. A", "2. B")
    for i, choice in enumerate(all_choices):
        if f"{i+1}. {choice}" in response:
            candidates.append((choice, response.rfind(f"{i+1}. {choice}"), "numbered"))

    # If no candidates found, try to extract from the entire response
    if not candidates:
        # Look for any choice letter in the response
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))

    # Return the best candidate
    if candidates:
        # Sort by position (later in text) and priority of format
        format_priority = {
            "start": 10, "end": 9, "numbered": 8, "phrase": 7,
            "parentheses": 6, "period": 5, "colon": 4, "right_paren": 3,
            "space": 2, "dash": 1, "underscore": 1, "equals": 1, "fallback": 0
        }

        # Sort by format priority first, then by position
        candidates.sort(key=lambda x: (format_priority[x[2]], -x[1]), reverse=True)
        return candidates[0][0]

    return ""


# --- Sequence answers --------------------------------------------------------
# Ground truth is a comma-separated 1-indexed clip order, e.g. "3,1,4,2".
# score_sequence_answer reports binary exact-match accuracy and position-wise
# step accuracy.

_SEQ_NUMBER_RE = re.compile(r"\d+")
_SEQ_SEPARATOR_RE = re.compile(r",|;|\||/|->|-->|=>|→|\n")
_SEQ_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Phrase cascade tried in order when the answer carries no boxed value.
_SEQ_PHRASES = [
    "",
    "correct order is:",
    "correct order:",
    "correct order",
    "**Correct order:**",
    "*Correct order:*",
    "follow these steps in order",
    "the correct order is",
    "the final output should be",
]


def _parse_order_official(text: str) -> list:
    """Split on commas and take the first integer of each part."""
    order = []
    for part in str(text).split(","):
        number_match = _SEQ_NUMBER_RE.search(part.strip())
        if number_match:
            order.append(int(number_match.group()))
    return order


def parse_sequence_answer(predict_str: str) -> list:
    """Parse a clip order such as "[Clip 3 -> Clip 1, 2]" into [3, 1, 2].

    Tolerates spaces, "Clip N" prefixes, brackets, LaTeX wrappers and either
    "," or "->" separators.

    Args:
        predict_str (str): The extracted answer text.

    Returns:
        list: The parsed 1-indexed clip numbers, or an empty list.
    """
    if not predict_str:
        return []

    text = str(predict_str)
    for token in ("\\rightarrow", "\\longrightarrow", "\\to"):
        text = text.replace(token, ",")
    for token in ("\\text", "\\mathrm", "\\left", "\\right"):
        text = text.replace(token, " ")
    for char in "[]{}()$":
        text = text.replace(char, " ")

    parts = [part.strip() for part in _SEQ_SEPARATOR_RE.split(text) if part.strip()]
    if len(parts) > 1:
        # Separated form: keep the first integer of each part so "Clip 3" -> 3.
        order = []
        for part in parts:
            number_match = _SEQ_NUMBER_RE.search(part)
            if number_match:
                order.append(int(number_match.group()))
        return order

    # Single part: the numbers are space-separated ("3 1 4 2") or absent.
    return [int(val) for val in _SEQ_NUMBER_RE.findall(text)]


def _extract_following_text(text: str, phrase: str) -> list:
    """Return the text that follows `phrase` in each sentence."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    phrase_regex = re.compile(re.escape(phrase), re.IGNORECASE)

    results = []
    for sentence in sentences:
        match_result = phrase_regex.search(sentence)
        if match_result:
            results.append(sentence[match_result.end():].strip())

    return results


def _is_consecutive_in_range(seq: list, a: int, b: int) -> bool:
    """Return True when seq is a permutation of the integers a..b."""
    return set(seq) == set(range(a, b + 1))


def fetch_sequence_order(predict_str: str, expected_len: int) -> list:
    """Run the phrase cascade over an unformatted response.

    Fallback used when the response carries no \boxed{} answer. Accepts the
    first candidate whose numbers are a permutation of 1..expected_len.

    Args:
        predict_str (str): The response text to search.
        expected_len (int): Number of clips in the ground-truth order.

    Returns:
        list: The recovered order, or an empty list if no candidate matched.
    """
    text = str(predict_str).replace("*", "")  # strip markdown bold
    for phrase in _SEQ_PHRASES:
        candidates = _extract_following_text(text, phrase)
        if not candidates:
            continue
        for candidate in candidates:
            order = _parse_order_official(candidate)
            if len(order) == expected_len and _is_consecutive_in_range(order, 1, expected_len):
                return order

    return []


def score_sequence_answer(predict_str: str, ground_truth: str) -> tuple:
    """Score a permutation answer against a comma-separated ground-truth order.

    Args:
        predict_str (str): The full model response.
        ground_truth (str): 1-indexed comma-separated order, e.g. "3,1,4,2".

    Returns:
        tuple: (accuracy, step_accuracy). accuracy is 1.0 on an exact match and
            0.0 otherwise; step_accuracy is the position-wise match rate, and
            0.0 when the predicted and ground-truth lengths differ.
    """
    gt_order = _parse_order_official(ground_truth)
    if not gt_order:
        return 0.0, 0.0

    boxed = extract_boxed_answer(predict_str)
    if boxed:
        pred_order = parse_sequence_answer(boxed)
    else:
        # No boxed answer: search the <answer> block if there is one, else the
        # whole response, with the phrase cascade.
        answer_match = _SEQ_ANSWER_RE.search(predict_str or "")
        haystack = answer_match.group(1) if answer_match else (predict_str or "")
        pred_order = fetch_sequence_order(haystack, len(gt_order))

    if pred_order == gt_order:
        return 1.0, 1.0
    if len(pred_order) != len(gt_order):
        return 0.0, 0.0

    matches = sum(1 for gold, pred in zip(gt_order, pred_order) if gold == pred)
    return 0.0, matches / len(gt_order)
