#!/usr/bin/env python3
"""Validate this standalone package's 24 independent b256 measurements."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


FILE_RE = re.compile(
    r"block_(?P<block>\d{2})_(?P<order>[ABC]{3})_pos(?P<position>[123])_"
    r"(?P<variant>original|compete-first|compete-first-lazy)\.txt$"
)
CONFIG_RE = re.compile(
    r"^device=0 batches=256 tasks=1280 workers=96 runs=1 .*\bswimlane=off\b",
    re.MULTILINE,
)
METRIC_RE = re.compile(r"^\[METRIC\] run=1 submit_span_us=([0-9.]+)\b", re.MULTILINE)
SUMMARY_RE = re.compile(
    r"^\[SUMMARY\] runs=1 completed_runs=1 median_submit_span_us=([0-9.]+) "
    r"execution_status=PASS semantic_status=PASS postprocess_status=PASS$",
    re.MULTILINE,
)
HAMPEL_SCALE = 1.4826
HAMPEL_SIGMAS = 3.0


def fail(message: str) -> None:
    raise SystemExit(message)


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median_submit_span_us": statistics.median(values),
        "mean_submit_span_us": statistics.fmean(values),
        "min_submit_span_us": min(values),
        "max_submit_span_us": max(values),
        "population_stdev_us": statistics.pstdev(values),
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    variant = (script_dir / "VARIANT").read_text(encoding="utf-8").strip()
    result_dir = script_dir / "output" / "performance"
    paths = sorted((result_dir / "raw").glob("*.txt"))
    if len(paths) != 24:
        fail(f"{variant}: expected 24 independent raw files, found {len(paths)}")

    rows: list[dict[str, object]] = []
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
        metrics = METRIC_RE.findall(text)
        summaries = SUMMARY_RE.findall(text)
        if len(metrics) != 1 or len(summaries) != 1 or metrics[0] != summaries[0]:
            fail(f"{path}: expected one matching METRIC and PASS SUMMARY")
        rows.append(
            {
                "block": int(name_match.group("block")),
                "order": name_match.group("order"),
                "position": int(name_match.group("position")),
                "submit_span_us": float(metrics[0]),
                "source": path.name,
            }
        )

    if {int(row["block"]) for row in rows} != set(range(1, 25)):
        fail(f"{variant}: blocks must be exactly 01..24")
    order_counts = {
        order: sum(row["order"] == order for row in rows)
        for order in ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")
    }
    if set(order_counts.values()) != {4}:
        fail(f"{variant}: six orders are not each repeated four times: {order_counts}")
    position_counts = {
        position: sum(row["position"] == position for row in rows)
        for position in (1, 2, 3)
    }
    if set(position_counts.values()) != {8}:
        fail(f"{variant}: launch positions are not 8/8/8: {position_counts}")

    raw_values = [float(row["submit_span_us"]) for row in rows]
    center = statistics.median(raw_values)
    mad = statistics.median(abs(value - center) for value in raw_values)
    if mad == 0.0:
        fail(f"{variant}: zero MAD cannot define an outlier threshold")
    lower = center - HAMPEL_SIGMAS * HAMPEL_SCALE * mad
    upper = center + HAMPEL_SIGMAS * HAMPEL_SCALE * mad
    for row in rows:
        value = float(row["submit_span_us"])
        row["included"] = lower <= value <= upper
        row["exclusion_reason"] = "" if row["included"] else "hampel_3_scaled_mad"
    clean_values = [
        float(row["submit_span_us"]) for row in rows if bool(row["included"])
    ]

    document = {
        "schema": "pa_lazy_lambda_variant_independent_performance/v1",
        "variant": variant,
        "configuration": {
            "device": 0,
            "workers": 96,
            "batches": 256,
            "samples": 24,
            "runs_per_host_launch": 1,
            "launch_orders": "all six permutations, each repeated four times",
            "launch_positions": {"1": 8, "2": 8, "3": 8},
            "pmu_window": "off",
            "swimlane": "off-runtime",
            "winner_workload": "real-compute",
        },
        "outlier_rule": {
            "name": "Hampel",
            "median_us": center,
            "mad_us": mad,
            "lower_us": lower,
            "upper_us": upper,
            "raw_logs_retained": True,
        },
        "raw": describe(raw_values),
        "outlier_filtered": describe(clean_values),
        "excluded_samples": [row for row in rows if not bool(row["included"])],
    }
    (result_dir / "summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (result_dir / "samples.tsv").open("w", encoding="utf-8") as output:
        output.write(
            "variant\tblock\torder\tposition\tsubmit_span_us\tincluded\t"
            "exclusion_reason\tsource\n"
        )
        for row in sorted(rows, key=lambda item: int(item["block"])):
            output.write(
                f"{variant}\t{row['block']}\t{row['order']}\t{row['position']}\t"
                f"{float(row['submit_span_us']):.3f}\t{row['included']}\t"
                f"{row['exclusion_reason']}\t{row['source']}\n"
            )
    raw = document["raw"]
    clean = document["outlier_filtered"]
    lines = [
        f"# {variant} device0 b256 result",
        "",
        "24 independent host launches, one device run per launch; PMU off, runtime swimlane off, real-compute.",
        "Raw logs are retained. The primary row excludes samples outside median ± 3×1.4826×MAD.",
        "",
        "| View | N | Median (us) | Mean (us) | Min (us) | Max (us) | Stddev (us) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Raw | {raw['count']} | {raw['median_submit_span_us']:.3f} | "
        f"{raw['mean_submit_span_us']:.3f} | {raw['min_submit_span_us']:.3f} | "
        f"{raw['max_submit_span_us']:.3f} | {raw['population_stdev_us']:.3f} |",
        f"| Outlier-filtered | {clean['count']} | {clean['median_submit_span_us']:.3f} | "
        f"{clean['mean_submit_span_us']:.3f} | {clean['min_submit_span_us']:.3f} | "
        f"{clean['max_submit_span_us']:.3f} | {clean['population_stdev_us']:.3f} |",
        "",
        f"Excluded samples: {raw['count'] - clean['count']}.",
    ]
    (result_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
