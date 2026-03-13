"""Scoring functions for benchmark evaluation."""

import json
import re
import string
import subprocess


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison: lowercase, strip articles/punctuation/whitespace."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def token_f1(prediction: str, references: list[str] | str) -> float:
    """Compute token-level F1 between prediction and best-matching reference.

    Standard metric for extractive QA (SQuAD, LongBench, etc.).
    """
    if isinstance(references, str):
        references = [references]

    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue

        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            continue

        # Count occurrences for proper F1 (not just set intersection)
        from collections import Counter
        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)
        common_count = sum(min(pred_counts[t], ref_counts[t]) for t in common)

        precision = common_count / len(pred_tokens)
        recall = common_count / len(ref_tokens)
        if precision + recall == 0:
            continue

        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def exact_match(prediction: str, references: list[str] | str) -> float:
    """Check if normalized prediction matches any reference. Returns 0.0 or 1.0."""
    if isinstance(references, str):
        references = [references]

    pred_norm = normalize_answer(prediction)
    for ref in references:
        if normalize_answer(ref) == pred_norm:
            return 1.0
    return 0.0


def contains_match(prediction: str, references: list[str] | str) -> float:
    """Check if prediction contains the key content of any reference. Returns 0.0 or 1.0."""
    if isinstance(references, str):
        references = [references]

    pred_norm = normalize_answer(prediction)
    for ref in references:
        ref_norm = normalize_answer(ref)
        # Check if all key tokens from reference appear in prediction
        ref_tokens = set(ref_norm.split())
        pred_tokens = set(pred_norm.split())
        if ref_tokens and ref_tokens.issubset(pred_tokens):
            return 1.0
    return 0.0


def llm_judge(question: str, prediction: str, references: list[str] | str) -> float:
    """Use claude as a judge to score whether the prediction correctly answers the question.

    Returns a score from 0.0 to 1.0:
      1.0 = correct and complete
      0.5 = partially correct
      0.0 = incorrect or irrelevant
    """
    if isinstance(references, str):
        references = [references]

    refs_str = " | ".join(references)
    prompt = (
        "You are evaluating whether a model's answer correctly addresses a question.\n\n"
        f"Question: {question}\n\n"
        f"Reference answer(s): {refs_str}\n\n"
        f"Model's answer: {prediction}\n\n"
        "Does the model's answer contain the correct information from the reference?\n"
        "The model's answer may be more detailed or differently worded — that's fine.\n"
        "What matters is whether it conveys the same key facts.\n\n"
        'Respond with ONLY a JSON object: {"score": <0.0|0.5|1.0>, "reason": "<brief explanation>"}\n'
        "- 1.0 = correct (key facts from reference are present)\n"
        "- 0.5 = partially correct (some key facts present, some missing or wrong)\n"
        "- 0.0 = incorrect (key facts missing or contradicted)"
    )

    try:
        result = subprocess.run(
            ["claude", "-p"],
            input=prompt,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return -1.0  # signal failure

        text = result.stdout.strip()
        # Extract JSON from response (may have markdown fences)
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            parsed = json.loads(json_match.group())
            score = float(parsed.get("score", -1))
            if score in (0.0, 0.5, 1.0):
                return score
        return -1.0
    except Exception:
        return -1.0  # signal failure, don't crash the benchmark
