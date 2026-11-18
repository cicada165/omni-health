#!/usr/bin/env python3
"""
scripts/run_evals.py — CLI Evaluation Runner & Benchmarking Suite (Milestone 8).

Executes the 42 conversational scheduling reliability scenarios across 9 categories,
computes quantitative benchmarks, asserts zero unsafe bookings, and outputs
a formatted Markdown summary and benchmark table.

Usage:
    python scripts/run_evals.py
    python scripts/run_evals.py --category "Normal Booking"
    python scripts/run_evals.py --id NORM-01
    python scripts/run_evals.py --output M8_REPORT.md
    python scripts/run_evals.py --json
    python scripts/run_evals.py --verbose
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from evals.metrics import MetricsCalculator
from evals.runner import EvalRunner
from evals.scenarios import (
    ScenarioCategory,
    get_all_scenarios,
    get_scenario_by_id,
    get_scenarios_by_category,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Appointment Scheduler Conversational Scheduling Evaluation Framework."
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default=None,
        help="Filter evaluation scenarios by category (e.g. 'Normal Booking', 'Corrections', 'Ambiguous').",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Run a specific scenario by ID (e.g. NORM-01, AMBIG-02).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run scenarios against live Gemini API instead of deterministic mocks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for live evaluation (defaults to GEMINI_MODEL).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save formatted Markdown benchmark report to file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw evaluation results in JSON format.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed conversational turn logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine scenarios to run
    if args.id:
        scenario = get_scenario_by_id(args.id)
        if not scenario:
            print(f"Error: Scenario with ID '{args.id}' not found.", file=sys.stderr)
            return 1
        scenarios = [scenario]
    elif args.category:
        scenarios = get_scenarios_by_category(args.category)
        if not scenarios:
            # Try fuzzy match
            all_s = get_all_scenarios()
            scenarios = [s for s in all_s if args.category.lower() in s.category.value.lower()]
        if not scenarios:
            print(f"Error: No scenarios match category '{args.category}'.", file=sys.stderr)
            return 1
    else:
        scenarios = get_all_scenarios()

    if not args.json:
        print("=" * 80)
        print("  APPOINTMENT SCHEDULER — RELIABILITY & EVALUATION FRAMEWORK (MILESTONE 8)")
        print(f"  Mode: {'LIVE API' if args.live else 'Deterministic Mock'} | Scenarios: {len(scenarios)}")
        print("=" * 80)
        print()

    runner = EvalRunner(live=args.live, model=args.model)
    t_start = time.perf_counter()

    results = []
    for idx, s in enumerate(scenarios, start=1):
        if not args.json:
            print(f"[{idx:02d}/{len(scenarios):02d}] Running `{s.id}`: {s.name} ... ", end="", flush=True)
        res = runner.run_scenario(s)
        results.append(res)
        if not args.json:
            status = "✔ PASS" if res.success else "✖ FAIL"
            print(f"{status} ({res.total_duration_ms:.1f}ms)")
            if args.verbose or not res.success:
                for t in res.turn_results:
                    print(f"       Turn {t.turn_index}: Patient: \"{t.user_input}\"")
                    first_line = t.agent_reply.splitlines()[0] if t.agent_reply else ""
                    print(f"               Agent  : \"{first_line[:90]}\" ({t.duration_ms:.1f}ms)")
                    if t.error:
                        print(f"               Error  : {t.error}")
                if res.error_message:
                    print(f"       Failure Reason: {res.error_message}")

    total_dur_ms = (time.perf_counter() - t_start) * 1000
    metrics = MetricsCalculator.compute(results)

    from evals.runner import EvalSuiteResult
    suite = EvalSuiteResult(
        scenario_results=results,
        metrics=metrics,
        total_duration_ms=total_dur_ms,
    )

    if args.json:
        payload = {
            "total_scenarios": metrics.total_scenarios,
            "passed_scenarios": metrics.passed_scenarios,
            "failed_scenarios": metrics.failed_scenarios,
            "task_completion_rate": metrics.task_completion_rate,
            "slot_selection_accuracy": metrics.slot_selection_accuracy,
            "unsafe_booking_rate": metrics.unsafe_booking_rate,
            "false_verbal_confirmation_rate": metrics.false_verbal_confirmation_rate,
            "failure_recovery_rate": metrics.failure_recovery_rate,
            "turn_latency_p50_ms": metrics.turn_latency_p50_ms,
            "turn_latency_p90_ms": metrics.turn_latency_p90_ms,
            "turn_latency_p99_ms": metrics.turn_latency_p99_ms,
            "scenarios": [
                {
                    "id": r.scenario_id,
                    "name": r.scenario_name,
                    "category": r.category,
                    "success": r.success,
                    "unsafe_booking": r.unsafe_booking,
                    "false_verbal_confirm": r.false_verbal_confirm,
                    "duration_ms": r.total_duration_ms,
                    "error": r.error_message,
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0 if metrics.failed_scenarios == 0 else 1

    # Print Summary Markdown Table
    print()
    print("=" * 80)
    print("  EVALUATION BENCHMARK METRICS SUMMARY")
    print("=" * 80)
    print()
    print(MetricsCalculator.format_markdown_table(metrics))
    print()
    print("=" * 80)
    print("  PERFORMANCE BY SCENARIO CATEGORY")
    print("=" * 80)
    print()
    print(MetricsCalculator.format_category_breakdown(metrics))
    print()

    # Safety Invariant Assertions
    safety_ok = True
    if metrics.unsafe_booking_rate != 0.0:
        print(f"✖ CRITICAL SAFETY VIOLATION: Unsafe Booking Rate is {metrics.unsafe_booking_rate:.1f}% (Must be 0.0%)", file=sys.stderr)
        safety_ok = False
    else:
        print("✔ INVARIANT VERIFIED: Unsafe Booking Rate is strictly 0.0%.")

    if metrics.false_verbal_confirmation_rate != 0.0:
        print(f"✖ CRITICAL SAFETY VIOLATION: False Confirmation Rate is {metrics.false_verbal_confirmation_rate:.1f}% (Must be 0.0%)", file=sys.stderr)
        safety_ok = False
    else:
        print("✔ INVARIANT VERIFIED: False Verbal Confirmation Rate is strictly 0.0%.")

    if metrics.failed_scenarios > 0:
        print(f"\n✖ Benchmark failed: {metrics.failed_scenarios} scenario(s) failed.", file=sys.stderr)
        safety_ok = False
    else:
        print(f"\n✔ All {metrics.passed_scenarios} scenarios passed successfully in {total_dur_ms:.1f}ms.")

    # Write Report if requested
    if args.output:
        report_content = MetricsCalculator.format_full_report(suite)
        out_path = Path(args.output)
        out_path.write_text(report_content, encoding="utf-8")
        print(f"✔ Full evaluation report saved to: {out_path.resolve()}")

    return 0 if safety_ok else 1


if __name__ == "__main__":
    sys.exit(main())
