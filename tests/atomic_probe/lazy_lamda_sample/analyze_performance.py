#!/usr/bin/env python3
"""Validate and summarize the final independent-run A/B/C measurements."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "A": ("A_original", "original", "A original"),
    "B": ("B_compete_first", "compete-first", "B compete-first eager"),
    "C": ("C_compete_first_lazy", "compete-first-lazy", "C compete-first + lazy lambda"),
}
FILE_RE = re.compile(
    r"block_(?P<block>\d{2})_(?P<order>[ABC]{3})_pos(?P<position>[123])_"
    r"(?P<variant>original|compete-first|compete-first-lazy)\.txt$"
)
CONFIG_RE = re.compile(
    r"^device=0 batches=256 tasks=1280 workers=96 runs=1 .*\bswimlane=off\b",
    re.MULTILINE,
)
SUMMARY_RE = re.compile(
    r"^\[SUMMARY\] runs=1 completed_runs=1 median_submit_span_us=([0-9.]+) "
    r"execution_status=PASS semantic_status=PASS postprocess_status=PASS$",
    re.MULTILINE,
)
FIELD_PATTERNS = {
    "submit_span_us": re.compile(r"^\[METRIC\].*submit_span_us=([0-9.]+)", re.MULTILINE),
    "host_launch_us": re.compile(r"^\[METRIC\].*host_launch_us=([0-9.]+)", re.MULTILINE),
    "fanin_loads": re.compile(r"^\[METRIC\].*fanin_loads=(\d+)", re.MULTILINE),
    "fanin_not_ready": re.compile(r"^\[ATOMIC\].*fanin_not_ready=(\d+)", re.MULTILINE),
    "frontier_flag": re.compile(r"^\[ATOMIC\].*frontier_flag=(\d+)", re.MULTILINE),
    "frontier_ready_fetch_max": re.compile(
        r"^\[ATOMIC\].*frontier_ready_fetch_max=(\d+)", re.MULTILINE
    ),
    "ring_bp": re.compile(r"^\[PLACEMENT\].*RingBp=(\d+)", re.MULTILINE),
    "sf_max_us": re.compile(r"^\[KERNEL\] SF .*max_us=([0-9.]+)", re.MULTILINE),
}
FLOAT_FIELDS = {"submit_span_us", "host_launch_us", "sf_max_us"}
HAMPEL_SCALE = 1.4826
HAMPEL_SIGMAS = 3.0


def fail(message: str) -> None:
    raise SystemExit(message)


def one_match(pattern: re.Pattern[str], text: str, path: Path, field: str) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        fail(f"{path}: expected one {field}, found {len(matches)}")
    return matches[0]


def median(values: list[float | int]) -> float:
    if not values:
        fail("internal error: median of empty data")
    return float(statistics.median(values))


def pearson(rows: list[dict[str, Any]], x_name: str, y_name: str) -> float | None:
    xs = [float(row[x_name]) for row in rows]
    ys = [float(row[y_name]) for row in rows]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = math.sqrt(
        sum((value - x_mean) ** 2 for value in xs)
        * sum((value - y_mean) ** 2 for value in ys)
    )
    if denominator == 0.0:
        return None
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(xs, ys)
    ) / denominator


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for letter, (package, variant, _label) in VARIANTS.items():
        raw_dir = ROOT / package / "output" / "performance" / "raw"
        paths = sorted(raw_dir.glob("*.txt"))
        if len(paths) != 24:
            fail(f"{variant}: expected 24 final raw files, found {len(paths)}")
        for path in paths:
            name_match = FILE_RE.fullmatch(path.name)
            if name_match is None or name_match.group("variant") != variant:
                fail(f"{variant}: unexpected raw filename {path.name}")
            text = path.read_text(encoding="utf-8")
            if CONFIG_RE.search(text) is None:
                fail(f"{path}: fixed device0/b256/runs=1/no-swimlane config missing")
            if "[WINNER-WORKLOAD] mode=real-compute " not in text:
                fail(f"{path}: real-compute workload missing")
            if "[PMU-CONFIG] window=off " not in text:
                fail(f"{path}: PMU is not explicitly off")
            summary_span = float(one_match(SUMMARY_RE, text, path, "PASS summary"))
            values: dict[str, float | int] = {}
            for field, pattern in FIELD_PATTERNS.items():
                value = one_match(pattern, text, path, field)
                values[field] = float(value) if field in FLOAT_FIELDS else int(value)
            if summary_span != values["submit_span_us"]:
                fail(f"{path}: one-run SUMMARY and METRIC spans differ")
            block = int(name_match.group("block"))
            if not 1 <= block <= 24:
                fail(f"{path}: block must be in 01..24")
            rows.append(
                {
                    "letter": letter,
                    "variant": variant,
                    "block": block,
                    "order": name_match.group("order"),
                    "position": int(name_match.group("position")),
                    "source": str(path.relative_to(ROOT)),
                    **values,
                }
            )

    for block in range(1, 25):
        block_rows = [row for row in rows if row["block"] == block]
        if {row["letter"] for row in block_rows} != set(VARIANTS):
            fail(f"block {block:02d}: missing A/B/C member")
        orders = {row["order"] for row in block_rows}
        if len(orders) != 1:
            fail(f"block {block:02d}: variants disagree on launch order")
        order = next(iter(orders))
        if sorted(order) != ["A", "B", "C"]:
            fail(f"block {block:02d}: invalid launch order {order}")
        for row in block_rows:
            expected_position = order.index(str(row["letter"])) + 1
            if row["position"] != expected_position:
                fail(f"block {block:02d}: filename position mismatch for {row['letter']}")

    order_counts = {
        order: sum(row["letter"] == "A" and row["order"] == order for row in rows)
        for order in ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")
    }
    if set(order_counts.values()) != {4}:
        fail(f"six launch orders are not each repeated four times: {order_counts}")
    for letter in VARIANTS:
        position_counts = {
            position: sum(
                row["letter"] == letter and row["position"] == position for row in rows
            )
            for position in (1, 2, 3)
        }
        if set(position_counts.values()) != {8}:
            fail(f"{letter}: launch positions are not 8/8/8: {position_counts}")
    return rows


def flag_outliers(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for letter in VARIANTS:
        variant_rows = [row for row in rows if row["letter"] == letter]
        values = [float(row["submit_span_us"]) for row in variant_rows]
        center = median(values)
        mad = median([abs(value - center) for value in values])
        if mad == 0.0:
            fail(f"{letter}: zero MAD cannot define an outlier threshold")
        scaled_mad = HAMPEL_SCALE * mad
        lower = center - HAMPEL_SIGMAS * scaled_mad
        upper = center + HAMPEL_SIGMAS * scaled_mad
        thresholds[letter] = {
            "median_us": center,
            "mad_us": mad,
            "scaled_mad_us": scaled_mad,
            "lower_us": lower,
            "upper_us": upper,
        }
        for row in variant_rows:
            value = float(row["submit_span_us"])
            row["included"] = lower <= value <= upper
            row["exclusion_reason"] = "" if row["included"] else "hampel_3_scaled_mad"
    return thresholds


def describe(rows: list[dict[str, Any]], included_only: bool) -> dict[str, Any]:
    selected = [row for row in rows if not included_only or row["included"]]
    values = [float(row["submit_span_us"]) for row in selected]
    return {
        "count": len(values),
        "median_submit_span_us": median(values),
        "mean_submit_span_us": statistics.fmean(values),
        "min_submit_span_us": min(values),
        "max_submit_span_us": max(values),
        "population_stdev_us": statistics.pstdev(values),
    }


def signal_medians(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    return {
        field: median([row[field] for row in rows])
        for field in (
            "submit_span_us",
            "fanin_loads",
            "fanin_not_ready",
            "frontier_flag",
            "frontier_ready_fetch_max",
            "ring_bp",
            "sf_max_us",
        )
    }


def variant_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for letter, (_package, variant, label) in VARIANTS.items():
        variant_rows = [row for row in rows if row["letter"] == letter]
        kept = [row for row in variant_rows if row["included"]]
        excluded = [row for row in variant_rows if not row["included"]]
        result[variant] = {
            "letter": letter,
            "label": label,
            "raw": describe(variant_rows, included_only=False),
            "outlier_filtered": describe(variant_rows, included_only=True),
            "position_medians_us_raw": {
                str(position): median(
                    [
                        row["submit_span_us"]
                        for row in variant_rows
                        if row["position"] == position
                    ]
                )
                for position in (1, 2, 3)
            },
            "collection_half_medians_us_raw": {
                "blocks_01_12": median(
                    [row["submit_span_us"] for row in variant_rows if row["block"] <= 12]
                ),
                "blocks_13_24": median(
                    [row["submit_span_us"] for row in variant_rows if row["block"] >= 13]
                ),
            },
            "kept_signal_medians": signal_medians(kept),
            "excluded_signal_medians": signal_medians(excluded),
            "excluded_samples": [
                {
                    "block": row["block"],
                    "position": row["position"],
                    "submit_span_us": row["submit_span_us"],
                    "source": row["source"],
                }
                for row in excluded
            ],
        }
    return result


def comparison(
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    candidate_letter: str,
    baseline_letter: str,
) -> dict[str, Any]:
    candidate_variant = VARIANTS[candidate_letter][1]
    baseline_variant = VARIANTS[baseline_letter][1]

    def aggregate(filtered: bool) -> dict[str, float]:
        candidate = float(
            stats[candidate_variant]["outlier_filtered" if filtered else "raw"]
            ["median_submit_span_us"]
        )
        baseline = float(
            stats[baseline_variant]["outlier_filtered" if filtered else "raw"]
            ["median_submit_span_us"]
        )
        delta = candidate - baseline
        return {"delta_us": delta, "delta_percent": delta / baseline * 100.0}

    raw_deltas: list[float] = []
    clean_deltas: list[float] = []
    excluded_blocks: list[int] = []
    for block in range(1, 25):
        candidate = next(
            row
            for row in rows
            if row["letter"] == candidate_letter and row["block"] == block
        )
        baseline = next(
            row
            for row in rows
            if row["letter"] == baseline_letter and row["block"] == block
        )
        delta = float(candidate["submit_span_us"]) - float(baseline["submit_span_us"])
        raw_deltas.append(delta)
        if candidate["included"] and baseline["included"]:
            clean_deltas.append(delta)
        else:
            excluded_blocks.append(block)

    clean_baseline_median = float(
        stats[baseline_variant]["outlier_filtered"]["median_submit_span_us"]
    )
    return {
        "candidate": candidate_variant,
        "baseline": baseline_variant,
        "aggregate_raw": aggregate(False),
        "aggregate_outlier_filtered": aggregate(True),
        "paired_raw": {
            "count": len(raw_deltas),
            "median_delta_us": median(raw_deltas),
            "candidate_faster_blocks": sum(delta < 0.0 for delta in raw_deltas),
        },
        "paired_outlier_filtered": {
            "count": len(clean_deltas),
            "median_delta_us": median(clean_deltas),
            "median_delta_percent_of_filtered_baseline": (
                median(clean_deltas) / clean_baseline_median * 100.0
            ),
            "mean_delta_us": statistics.fmean(clean_deltas),
            "candidate_faster_blocks": sum(delta < 0.0 for delta in clean_deltas),
            "excluded_blocks": excluded_blocks,
        },
        "paired_raw_collection_halves": {
            "blocks_01_12_median_delta_us": median(raw_deltas[:12]),
            "blocks_13_24_median_delta_us": median(raw_deltas[12:]),
        },
    }


def write_samples(rows: list[dict[str, Any]]) -> None:
    columns = (
        "letter",
        "variant",
        "block",
        "order",
        "position",
        "submit_span_us",
        "included",
        "exclusion_reason",
        "host_launch_us",
        "fanin_loads",
        "fanin_not_ready",
        "frontier_flag",
        "frontier_ready_fetch_max",
        "ring_bp",
        "sf_max_us",
        "source",
    )
    with (ROOT / "performance_samples.tsv").open("w", encoding="utf-8") as output:
        output.write("\t".join(columns) + "\n")
        for row in sorted(rows, key=lambda item: (item["block"], item["position"])):
            output.write("\t".join(str(row[column]) for column in columns) + "\n")


def main() -> None:
    rows = load_rows()
    thresholds = flag_outliers(rows)
    stats = variant_statistics(rows)
    correlations = {
        letter: {
            field: pearson(
                [row for row in rows if letter == "all" or row["letter"] == letter],
                "submit_span_us",
                field,
            )
            for field in (
                "fanin_loads",
                "fanin_not_ready",
                "frontier_flag",
                "frontier_ready_fetch_max",
                "ring_bp",
                "sf_max_us",
                "host_launch_us",
            )
        }
        for letter in ("A", "B", "C", "all")
    }
    document = {
        "schema": "pa_lazy_lambda_independent_performance/v1",
        "configuration": {
            "device": 0,
            "workers": 96,
            "batches": 256,
            "samples_per_variant": 24,
            "host_launches": 72,
            "runs_per_host_launch": 1,
            "launch_orders": "all six permutations, each repeated four times",
            "launch_positions_per_variant": {"1": 8, "2": 8, "3": 8},
            "pmu_window": "off",
            "swimlane": "off-runtime",
            "winner_workload": "real-compute",
        },
        "outlier_rule": {
            "name": "Hampel",
            "center": "per-variant median",
            "scale": "1.4826 * per-variant MAD",
            "threshold_scaled_mad": 3.0,
            "raw_logs_retained": True,
            "paired_policy": "exclude a block if either member is an outlier",
            "thresholds": thresholds,
        },
        "variants": stats,
        "comparisons": {
            "B_minus_A_compete_first": comparison(rows, stats, "B", "A"),
            "C_minus_B_lazy_lambda": comparison(rows, stats, "C", "B"),
            "C_minus_A_total": comparison(rows, stats, "C", "A"),
        },
        "pearson_correlations": correlations,
    }
    (ROOT / "performance_summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_samples(rows)
    print(json.dumps(document["comparisons"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
