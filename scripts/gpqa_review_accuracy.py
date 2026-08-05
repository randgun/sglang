#!/usr/bin/env python3
"""Compute interim GPQA accuracy from EvalScope review JSONL files.

The review schema has changed across EvalScope releases.  This script first
uses an explicit per-sample score when one is present.  Otherwise it extracts
the predicted choice from common review fields, or from the final assistant
message, and compares it with ``target``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CHOICES = frozenset("ABCD")
PREDICTION_KEYS = (
    "filtered_resps",
    "filtered_responses",
    "predicted_answer",
    "prediction",
    "pred",
    "model_answer",
    "response",
    "model_output",
    "output",
    "completion",
    "resps",
)
SCORE_KEYS = ("sample_score", "score", "scores", "metric_score", "metrics", "value")
ACCURACY_KEYS = (
    "acc",
    "accuracy",
    "average_accuracy",
    "exact_match",
    "correct",
    "is_correct",
)


def normalize_choice(value: Any) -> str | None:
    """Return A/B/C/D only when *value* looks like a final answer."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        # EvalScope filters commonly store one response in a single-item list.
        for item in reversed(value):
            choice = normalize_choice(item)
            if choice is not None:
                return choice
        return None
    if isinstance(value, dict):
        for key in ("answer", "content", "text", "prediction", "output"):
            if key in value:
                choice = normalize_choice(value[key])
                if choice is not None:
                    return choice
        return None

    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()

    # Exact forms are safest and cover EvalScope's filtered GPQA prediction.
    exact = re.fullmatch(r"\s*(?:OPTION|CHOICE)?\s*[\(\[\{]?([A-D])[\)\]\}]?[\s.。:：]*", upper)
    if exact:
        return exact.group(1)

    patterns = (
        r"\\BOXED\s*\{\s*([A-D])\s*\}",
        r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|=|:|：)?\s*[\(\[]?([A-D])[\)\]]?",
        r"(?:最终答案|答案|选择|选项)\s*(?:是|为|=|:|：)?\s*[（(\[]?([A-D])[）)\]]?",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, upper, flags=re.IGNORECASE))
    if matches:
        return matches[-1].upper()

    # Last resort: accept a standalone choice only near the end of the reply.
    tail = upper[-160:]
    trailing = re.search(r"(?:^|[\s（(\[])\s*([A-D])\s*[）)\].。]*\s*$", tail)
    return trailing.group(1) if trailing else None


def normalize_target(value: Any) -> str | None:
    choice = normalize_choice(value)
    if choice in CHOICES:
        return choice
    return None


def score_to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)) and float(value) in (0.0, 1.0):
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "correct", "pass", "passed", "1", "1.0"}:
            return True
        if normalized in {"false", "incorrect", "wrong", "fail", "failed", "0", "0.0"}:
            return False
    return None


def _score_from_container(value: Any, seen: set[int]) -> bool | None:
    """Search nested score containers without walking unrelated record data."""
    direct = score_to_bool(value)
    if direct is not None:
        return direct
    if not isinstance(value, dict) or id(value) in seen:
        return None
    seen.add(id(value))

    for key in ACCURACY_KEYS:
        if key in value:
            result = score_to_bool(value[key])
            if result is not None:
                return result

    for key in SCORE_KEYS:
        if key in value:
            result = _score_from_container(value[key], seen)
            if result is not None:
                return result

    # A single numeric entry inside a score container is normally its metric.
    if len(value) == 1:
        result = _score_from_container(next(iter(value.values())), seen)
        if result is not None:
            return result
    return None


def explicit_score(record: dict[str, Any]) -> bool | None:
    # Retain compatibility with older EvalScope reviews that put accuracy at
    # the record root.
    for key in ACCURACY_KEYS:
        if key in record:
            result = score_to_bool(record[key])
            if result is not None:
                return result

    seen: set[int] = set()
    for score_key in SCORE_KEYS:
        if score_key in record:
            result = _score_from_container(record[score_key], seen)
            if result is not None:
                return result
    return None


def assistant_message(record: dict[str, Any]) -> Any:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() == "assistant":
            return message.get("content")
    return None


def predicted_choice(record: dict[str, Any]) -> tuple[str | None, str | None]:
    for key in PREDICTION_KEYS:
        if key not in record:
            continue
        choice = normalize_choice(record[key])
        if choice is not None:
            return choice, key
    choice = normalize_choice(assistant_message(record))
    return (choice, "messages[-assistant]") if choice is not None else (None, None)


def find_jsonl(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            review_dir = path / "reviews"
            root = review_dir if review_dir.is_dir() else path
            files.extend(sorted(root.rglob("*.jsonl")))
        else:
            print(f"warning: not found: {path}", file=sys.stderr)
    # Preserve order while removing duplicate paths.
    return list(dict.fromkeys(path.resolve() for path in files))


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute the current GPQA accuracy from EvalScope review JSONL files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="A review JSONL file, reviews directory, or EvalScope run directory.",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Show the first N malformed/unresolved records (default: 10; "
            "0 disables). All incorrect records are always shown."
        ),
    )
    args = parser.parse_args()

    files = find_jsonl(args.paths)
    if not files:
        parser.error("no JSONL files found")

    stats: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    wrong_details: list[str] = []
    diagnostic_details: list[str] = []

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                stats["lines"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats["malformed"] += 1
                    if len(diagnostic_details) < args.show_errors:
                        diagnostic_details.append(
                            f"MALFORMED {path}:{line_no}: {exc}"
                        )
                    continue
                if not isinstance(record, dict):
                    stats["malformed"] += 1
                    continue

                target = normalize_target(record.get("target"))
                judged = explicit_score(record)
                prediction, source = predicted_choice(record)
                index = record.get("index", "?")

                if target is None:
                    stats["unresolved_target"] += 1
                    if len(diagnostic_details) < args.show_errors:
                        diagnostic_details.append(
                            f"UNRESOLVED target file={path.name} line={line_no} index={index} "
                            f"raw_target={record.get('target')!r}"
                        )
                    continue

                if judged is not None:
                    stats["explicit_score"] += 1
                    sources["explicit score"] += 1
                    correct = judged
                elif prediction is not None:
                    stats["extracted_score"] += 1
                    sources[source or "unknown"] += 1
                    correct = prediction == target
                else:
                    stats["unresolved_prediction"] += 1
                    if len(diagnostic_details) < args.show_errors:
                        diagnostic_details.append(
                            f"UNRESOLVED prediction file={path.name} line={line_no} "
                            f"index={index} target={target}"
                        )
                    continue

                stats["scored"] += 1
                stats["correct" if correct else "incorrect"] += 1
                if not correct:
                    wrong_details.append(
                        f"WRONG file={path.name} line={line_no} index={index} "
                        f"target={target} prediction={prediction or '<explicit-score=0>'}"
                    )

    scored = stats["scored"]
    accuracy = stats["correct"] / scored if scored else 0.0
    low, high = wilson_interval(stats["correct"], scored)

    print("GPQA interim review accuracy")
    print(f"files:                 {len(files)}")
    print(f"JSONL records:         {stats['lines']}")
    print(f"scored:                {scored}")
    print(f"correct:               {stats['correct']}")
    print(f"incorrect:             {stats['incorrect']}")
    print(f"accuracy:              {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"95% Wilson interval:   [{low * 100:.2f}%, {high * 100:.2f}%]")
    print(f"explicitly judged:     {stats['explicit_score']}")
    print(f"locally extracted:     {stats['extracted_score']}")
    print(f"unresolved prediction: {stats['unresolved_prediction']}")
    print(f"unresolved target:     {stats['unresolved_target']}")
    print(f"malformed JSON lines:  {stats['malformed']}")
    if sources:
        print("answer sources:")
        for source, count in sources.most_common():
            print(f"  {source}: {count}")
    if wrong_details:
        print("wrong details:")
        for detail in wrong_details:
            print(f"  {detail}")
    if diagnostic_details:
        print("parse diagnostics:")
        for detail in diagnostic_details:
            print(f"  {detail}")

    return 0 if scored else 2


if __name__ == "__main__":
    raise SystemExit(main())
