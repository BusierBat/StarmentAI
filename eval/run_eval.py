"""Run StarmentAI eval items against a local Ollama model.

Usage:
    python eval/run_eval.py --model starmentai --split dev
    python eval/run_eval.py --model starmentai-candidate --split test \
        --output reports/candidate_test.json
"""

import argparse
import json
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ollama_client import (  # noqa: E402
    OllamaClient,
    OllamaError,
    question_needs_thinking,
)
from eval.score import score_item, summarize  # noqa: E402

DEFAULT_EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
DEFAULT_REPORT_DIR = ROOT / "reports"


def load_items(path: Path, split: str) -> list[dict[str, Any]]:
    """Load eval items, optionally filtered by split."""
    items: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if split != "all" and item.get("split") != split:
                continue

            items.append(item)

    return items


def resolve_think(mode: str, user_input: str) -> bool:
    """Resolve the think mode for one item."""
    if mode == "true":
        return True

    if mode == "false":
        return False

    return question_needs_thinking(user_input)


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content

    return ""


def _tokens_per_second(usage: dict[str, Any]) -> float | None:
    eval_tokens = usage.get("eval_count")
    eval_duration = usage.get("eval_duration")

    if not eval_tokens or not eval_duration:
        return None

    seconds = float(eval_duration) / 1_000_000_000

    if seconds <= 0:
        return None

    return round(float(eval_tokens) / seconds, 2)


def select_per_category(
    items: list[dict[str, Any]],
    per_category: int,
) -> list[dict[str, Any]]:
    """Pick up to N items per category, preserving file order."""
    buckets: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        buckets.setdefault(str(item.get("category")), []).append(item)

    selected: list[dict[str, Any]] = []

    for bucket in buckets.values():
        selected.extend(bucket[:per_category])

    return selected


def run_eval(
    model: str,
    split: str,
    think_mode: str,
    eval_set: Path,
    limit: int | None = None,
    per_category: int | None = None,
    sleep: float = 0.0,
) -> dict[str, Any]:
    """Run the eval set and return a JSON-serializable report."""
    items = load_items(eval_set, split)

    if per_category is not None:
        items = select_per_category(items, per_category)

    if limit is not None:
        items = items[:limit]

    if not items:
        raise ValueError(f"No eval items found for split {split!r}.")

    client = OllamaClient(model=model)

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    errors = 0

    try:
        for item in items:
            messages = item.get("messages") or []
            think = resolve_think(think_mode, _last_user_content(messages))

            started = time.perf_counter()

            try:
                response = client.chat_with_usage(messages, think=think)
                content = response["content"]
                usage = response["usage"]
                error = None

            except OllamaError as exc:
                content = ""
                usage = {}
                error = str(exc)
                errors += 1

            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)

            scored = score_item(item, content)

            if error:
                scored["score"] = 0.0
                scored["all_passed"] = False

            scored.update({
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": usage.get("prompt_eval_count"),
                "eval_tokens": usage.get("eval_count"),
                "tokens_per_second": _tokens_per_second(usage),
                "think": think,
                "usage": usage,
                "error": error,
            })

            results.append(scored)

            if sleep:
                time.sleep(sleep)
    finally:
        client.close()

    report = {
        "model": model,
        "split": split,
        "think": think_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_set": str(eval_set),
        "errors": errors,
        "items": results,
        "summary": summarize(results, latencies),
    }

    return report


def default_output_path(model: str, split: str) -> Path:
    """Build a default report path for a model and split."""
    safe_model = model.replace(":", "_").replace("/", "_")
    return DEFAULT_REPORT_DIR / f"eval_{safe_model}_{split}.json"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a Markdown summary for one eval report."""
    summary = report["summary"]

    lines = [
        f"# StarmentAI eval — {report['model']}",
        "",
        f"- Split: {report['split']}",
        f"- Think mode: {report['think']}",
        f"- Timestamp: {report['timestamp']}",
        f"- Items: {summary['items']}",
        f"- Errors: {report['errors']}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean score | {summary['mean_score']} |",
        f"| All-passed rate | {summary['all_passed_rate']} |",
        f"| Abstention rate | {summary['abstention_rate']} |",
        f"| Safety items | {summary['safety_items']} |",
        f"| Safety violations | {summary['safety_violations']} |",
    ]

    latency = summary.get("latency_ms")

    if latency:
        lines.append(
            f"| Latency p50 / p95 (ms) | {latency['p50']} / {latency['p95']} |"
        )

    lines.extend([
        "",
        "## Categories",
        "",
        "| Category | Items | Mean score | All-passed |",
        "|---|---|---|---|",
    ])

    for name, values in summary["categories"].items():
        lines.append(
            f"| {name} | {values['items']} | {values['mean_score']} "
            f"| {values['all_passed_rate']} |"
        )

    failed = [item for item in report["items"] if not item["all_passed"]]

    lines.extend(["", f"## Failed items ({len(failed)})", ""])

    if not failed:
        lines.append("None.")
    else:
        for item in failed:
            excerpt = item["response"][:240].replace("\n", " ")
            lines.append(
                f"- **{item['id']}** ({item['category']}, "
                f"score {item['score']}): {excerpt}"
            )

    lines.append("")

    return "\n".join(lines)


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact summary to stdout."""
    summary = report["summary"]

    print(f"Model: {report['model']}")
    print(f"Split: {report['split']}  Think: {report['think']}")
    print(f"Items: {summary['items']}  Errors: {report['errors']}")
    print(f"Mean score: {summary['mean_score']}")
    print(f"All-passed rate: {summary['all_passed_rate']}")
    print(f"Abstention rate: {summary['abstention_rate']}")
    print(f"Safety violations: {summary['safety_violations']}")

    latency = summary.get("latency_ms")

    if latency:
        print(
            "Latency ms: "
            f"mean={latency['mean']} "
            f"p50={latency['p50']} "
            f"p95={latency['p95']} "
            f"max={latency['max']}"
        )

    print("\nCategory scores:")

    for name, values in summary["categories"].items():
        print(
            f"  {name:<12} items={values['items']:<3} "
            f"score={values['mean_score']} "
            f"pass={values['all_passed_rate']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run StarmentAI eval items against Ollama.",
    )

    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--split",
        default="dev",
        choices=("dev", "test", "all"),
    )
    parser.add_argument(
        "--think",
        default="auto",
        choices=("auto", "true", "false"),
    )
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--per-category",
        type=int,
        default=None,
        help="Pick at most N items per category (stratified subset).",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    eval_set = Path(args.eval_set)

    if args.dry_run:
        items = load_items(eval_set, args.split)

        if args.per_category is not None:
            items = select_per_category(items, args.per_category)

        print(f"Loaded {len(items)} items from {eval_set}")
        return 0

    from config import CONFIG

    model = args.model or CONFIG.model_name

    report = run_eval(
        model=model,
        split=args.split,
        think_mode=args.think,
        eval_set=eval_set,
        limit=args.limit,
        per_category=args.per_category,
        sleep=args.sleep,
    )

    output = (
        Path(args.output)
        if args.output
        else default_output_path(model, args.split)
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print_summary(report)
    print(f"\nReport written to {output}")
    print(f"Markdown written to {markdown_path}")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
    
print("Hello, World!")         

