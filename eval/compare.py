"""Compare two StarmentAI eval reports and render a Markdown report.

Usage:
    python eval/compare.py reports/eval_starmentai_dev.json \
        reports/eval_starmentai-candidate_dev.json \
        --output reports/compare.md
"""

import argparse
import json

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_report(path: Path) -> dict[str, Any]:
    """Load one eval report JSON."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _format_delta(before: float, after: float) -> str:
    return f"{after - before:+.4f}"


def _format_latency(before: Any, after: Any) -> str:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return "n/a"

    before_p95 = float(before.get("p95", 0.0))
    after_p95 = float(after.get("p95", 0.0))

    if before_p95 == 0:
        return "n/a"

    change = (after_p95 - before_p95) / before_p95 * 100

    return f"{change:+.1f}%"


def compare(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Render a Markdown comparison for two reports."""
    before_summary = before.get("summary", {})
    after_summary = after.get("summary", {})

    lines = [
        "# StarmentAI eval comparison",
        "",
        f"- Baseline: `{before.get('model')}` ({before.get('split')}, "
        f"think={before.get('think')})",
        f"- Candidate: `{after.get('model')}` ({after.get('split')}, "
        f"think={after.get('think')})",
        f"- Baseline items: {before_summary.get('items')}",
        f"- Candidate items: {after_summary.get('items')}",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---|---|---|",
        (
            f"| Mean score | {before_summary.get('mean_score')} "
            f"| {after_summary.get('mean_score')} "
            f"| {_format_delta(before_summary.get('mean_score', 0), after_summary.get('mean_score', 0))} |"
        ),
        (
            f"| All-passed rate | {before_summary.get('all_passed_rate')} "
            f"| {after_summary.get('all_passed_rate')} "
            f"| {_format_delta(before_summary.get('all_passed_rate', 0), after_summary.get('all_passed_rate', 0))} |"
        ),
        (
            f"| Abstention rate | {before_summary.get('abstention_rate')} "
            f"| {after_summary.get('abstention_rate')} "
            f"| {_format_delta(before_summary.get('abstention_rate', 0), after_summary.get('abstention_rate', 0))} |"
        ),
        (
            f"| Safety violations | {before_summary.get('safety_violations')} "
            f"| {after_summary.get('safety_violations')} "
            f"| {after_summary.get('safety_violations', 0) - before_summary.get('safety_violations', 0):+d} |"
        ),
        (
            "| p95 latency | "
            f"{before_summary.get('latency_ms', {}).get('p95') if before_summary.get('latency_ms') else 'n/a'} | "
            f"{after_summary.get('latency_ms', {}).get('p95') if after_summary.get('latency_ms') else 'n/a'} | "
            f"{_format_latency(before_summary.get('latency_ms'), after_summary.get('latency_ms'))} |"
        ),
        "",
        "## Categories",
        "",
        "| Category | Items | Baseline | Candidate | Delta |",
        "|---|---|---|---|---|",
    ]

    before_categories = before_summary.get("categories", {})
    after_categories = after_summary.get("categories", {})

    for name in sorted(set(before_categories) | set(after_categories)):
        before_values = before_categories.get(name, {})
        after_values = after_categories.get(name, {})

        before_score = before_values.get("mean_score", 0.0)
        after_score = after_values.get("mean_score", 0.0)

        lines.append(
            f"| {name} | {after_values.get('items', before_values.get('items', 0))} "
            f"| {before_score} | {after_score} "
            f"| {_format_delta(before_score, after_score)} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Deltas are candidate minus baseline; positive is better for scores.",
        "- p95 latency delta is relative percent change.",
        "- Subjective style quality still requires manual rubric review.",
        "",
    ])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two StarmentAI eval reports.",
    )

    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "compare.md"),
    )

    args = parser.parse_args()

    before = load_report(Path(args.before))
    after = load_report(Path(args.after))

    markdown = compare(before, after)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"Comparison written to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())