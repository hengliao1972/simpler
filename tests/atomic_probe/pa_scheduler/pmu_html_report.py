#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""将 standalone submit-pmu raw JSON 转成可离线浏览的自包含 HTML 报告。"""

from __future__ import annotations

import argparse
import hashlib
import html
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

try:
    from .pmu_sidecar_analyzer import analyze, load_capture
except ImportError:
    from pmu_sidecar_analyzer import analyze, load_capture


REPORT_VERSION = 5
AIC_COLOR = "#2563eb"
AIV_COLOR = "#ea580c"
GRID_COLOR = "#cbd5e1"
TEXT_COLOR = "#334155"

# 本机 A5 受控 cold/warm 校准：PMU cycle 与 1 ns SYS_CNT 同窗读取。
# ALL 使用整组实测比值；AIC/AIV 使用各自分组的实测比值，避免用名义频率替代证据。
PMU_CALIBRATION_CYCLE_DELTA = 1_817_457
PMU_CALIBRATION_SYS_TICK_NS = 1_101_593
DEFAULT_PMU_CYCLES_PER_NS = 1.649844
DEFAULT_AIC_PMU_CYCLES_PER_NS = 1.650062
DEFAULT_AIV_PMU_CYCLES_PER_NS = 1.649731


def default_output_path(input_path: Path) -> Path:
    """由描述性 raw 名称稳定推导报告名称。"""

    name = input_path.name
    if name.endswith("_raw.json"):
        return input_path.with_name(f"{name[:-len('_raw.json')]}_report.html")
    if name.endswith(".json"):
        return input_path.with_name(f"{name[:-len('.json')]}_report.html")
    return input_path.with_name(f"{name}_report.html")


def _format_count(value: int | float) -> str:
    return f"{value:,.0f}"


def _format_per_core(value: int | float) -> str:
    return f"{value:,.2f}"


def _format_rate(value: int | float) -> str:
    return f"{value * 100:.4f}%"


def _cycles_to_us(cycles: int | float, cycles_per_ns: float) -> float:
    """按受控校准频率把 PMU cycle 换算为单核等效微秒。"""

    return float(cycles) / cycles_per_ns / 1000.0


def _cycle_time_cell(cycles: int | float, cycles_per_ns: float, decimals: int = 0) -> str:
    """同时保留原始 cycle 与校准后的等效时间，避免丢失原始证据。"""

    cycle_text = f"{float(cycles):,.{decimals}f}"
    return (
        f'<span class="cycle-value">{cycle_text} cycle</span>'
        f'<span class="cycle-time">≈{_cycles_to_us(cycles, cycles_per_ns):,.3f} µs</span>'
    )


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _raw_href(input_path: Path, output_path: Path) -> str:
    relative = os.path.relpath(input_path, output_path.parent)
    return quote(relative.replace(os.sep, "/"), safe="/")


def _group_metrics(
    analysis_group: dict[str, Any],
    group_records: Sequence[dict[str, Any]],
    cycles_per_ns: float,
) -> dict[str, int | float]:
    cores = int(analysis_group["cores"])
    total_sum = int(analysis_group["total_cycles_sum"])
    scalar_sum = int(analysis_group["scalar_busy_sum"])
    if len(group_records) != cores:
        raise ValueError("PMU report group record count does not match analyzer core count")
    total_values = [int(record["total_cycles"]) for record in group_records]
    scalar_values = [int(record["scalar_busy"]) for record in group_records]
    request_values = [int(record["icache_requests"]) for record in group_records]
    miss_values = [int(record["icache_misses"]) for record in group_records]
    return {
        "cores": cores,
        "total_sum": total_sum,
        "total_min": min(total_values),
        "total_per_core": total_sum / cores,
        "total_max": max(total_values),
        "total_per_core_us": _cycles_to_us(total_sum / cores, cycles_per_ns),
        "scalar_sum": scalar_sum,
        "scalar_min": min(scalar_values),
        "scalar_per_core": scalar_sum / cores,
        "scalar_max": max(scalar_values),
        "scalar_per_core_us": _cycles_to_us(scalar_sum / cores, cycles_per_ns),
        "scalar_share": 0.0 if total_sum == 0 else scalar_sum / total_sum,
        "non_scalar_busy_per_core": (total_sum - scalar_sum) / cores,
        "cycles_per_ns": cycles_per_ns,
        "requests_min": min(request_values),
        "requests_per_core": analysis_group["icache_requests_per_core"],
        "requests_max": max(request_values),
        "misses_min": min(miss_values),
        "misses_per_core": analysis_group["icache_misses_per_core"],
        "misses_max": max(miss_values),
        "miss_rate": analysis_group["icache_miss_rate"],
        "serial_equivalent_us": analysis_group["first_order_miss_per_core_us"],
    }


def _plot_value(record: dict[str, Any], metric: str) -> float:
    if metric == "icache_miss_rate":
        requests = int(record["icache_requests"])
        return 0.0 if requests == 0 else int(record["icache_misses"]) / requests
    return float(record[metric])


def _plot_label(metric: str, value: float) -> str:
    if metric == "icache_miss_rate":
        return f"{value * 100:.2f}%"
    return f"{value:,.0f}"


def _distribution_svg(
    records: Sequence[dict[str, Any]],
    metric: str,
    title: str,
    description: str,
    cycles_per_ns_by_role: dict[str, float] | None = None,
) -> str:
    """按 physical_core_id 绘制离散点；核编号只是位置，不连接成时间线。"""

    width = 1080
    height = 310
    left, right, top, bottom = 78, 28, 34, 54
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = [(record, _plot_value(record, metric)) for record in records]
    maximum = max(value for _, value in points)
    maximum = maximum * 1.08 if maximum > 0 else 1.0

    def x_position(physical_id: int) -> float:
        return left + plot_width * physical_id / 107.0

    def y_position(value: float) -> float:
        return top + plot_height * (1.0 - value / maximum)

    title_id = f"plot-{metric}-title"
    desc_id = f"plot-{metric}-desc"
    fragments = [
        f'<svg class="distribution" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="{title_id} {desc_id}">',
        f'<title id="{title_id}">{_escape(title)}</title>',
        f'<desc id="{desc_id}">{_escape(description)}</desc>',
    ]
    for index in range(5):
        ratio = index / 4
        value = maximum * (1.0 - ratio)
        y = top + plot_height * ratio
        fragments.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" />'
        )
        fragments.append(
            f'<text class="axis-label" x="{left-10}" y="{y+4:.2f}" text-anchor="end">'
            f'{_escape(_plot_label(metric, value))}</text>'
        )

    for tick in (0, 18, 36, 54, 72, 90, 107):
        x = x_position(tick)
        fragments.append(
            f'<line class="tick" x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" '
            f'y2="{height-bottom+5}" />'
        )
        fragments.append(
            f'<text class="axis-label" x="{x:.2f}" y="{height-bottom+22}" '
            f'text-anchor="middle">{tick}</text>'
        )
    fragments.append(
        f'<text class="axis-title" x="{left + plot_width/2:.2f}" y="{height-8}" '
        'text-anchor="middle">physical_core_id（0–107，未连接）</text>'
    )

    for record, value in sorted(points, key=lambda item: int(item[0]["physical_core_id"])):
        role = str(record["role"])
        physical_id = int(record["physical_core_id"])
        x = x_position(physical_id)
        y = y_position(value)
        requests = int(record["icache_requests"])
        misses = int(record["icache_misses"])
        rate = 0.0 if requests == 0 else misses / requests
        tooltip = (
            f"{role.upper()} worker={record['worker_id']} physical={physical_id} "
            f"block={record['block_id']} lane={record['lane']} "
            f"{metric}={_plot_label(metric, value)} whole_request={requests:,} "
            f"whole_miss={misses:,} whole_rate={rate * 100:.4f}%"
        )
        if metric in ("total_cycles", "scalar_busy") and cycles_per_ns_by_role is not None:
            tooltip += (
                f" calibrated_time={_cycles_to_us(value, cycles_per_ns_by_role[role]):,.3f}us"
            )
        if role == "aic":
            fragments.append(
                f'<circle class="point aic-point" tabindex="0" cx="{x:.2f}" cy="{y:.2f}" r="4.5">'
                f'<title>{_escape(tooltip)}</title></circle>'
            )
        else:
            fragments.append(
                f'<rect class="point aiv-point" tabindex="0" x="{x-4:.2f}" y="{y-4:.2f}" '
                f'width="8" height="8"><title>{_escape(tooltip)}</title></rect>'
            )
    fragments.append("</svg>")
    return "".join(fragments)


def _per_core_rows(
    records: Sequence[dict[str, Any]], cycles_per_ns_by_role: dict[str, float]
) -> str:
    rows: list[str] = []
    for record in sorted(records, key=lambda item: int(item["physical_core_id"])):
        requests = int(record["icache_requests"])
        misses = int(record["icache_misses"])
        total = int(record["total_cycles"])
        scalar = int(record["scalar_busy"])
        cycles_per_ns = cycles_per_ns_by_role[str(record["role"])]
        rate = 0.0 if requests == 0 else misses / requests
        scalar_share = 0.0 if total == 0 else scalar / total
        exact = bool(record.get("shadow_matches_primary"))
        rows.append(
            f'<tr data-worker-id="{int(record["worker_id"])}" data-role="{_escape(record["role"])}">'
            f"<td>{int(record['worker_id'])}</td>"
            f"<td>{int(record['physical_core_id'])}</td>"
            f"<td><span class=\"role role-{_escape(record['role'])}\">{_escape(str(record['role']).upper())}</span></td>"
            f"<td>{int(record['block_id'])}</td>"
            f"<td>{int(record['lane'])}</td>"
            f"<td>{_cycle_time_cell(total, cycles_per_ns)}</td>"
            f"<td>{_cycle_time_cell(scalar, cycles_per_ns)}</td>"
            f"<td>{scalar_share * 100:.4f}%</td>"
            f"<td>{requests:,}</td>"
            f"<td>{misses:,}</td>"
            f"<td>{rate * 100:.4f}%</td>"
            f"<td>{'是' if exact else '否'}</td>"
            f"<td>{'PASS' if record.get('trusted') is True else 'FAIL'}</td>"
            "</tr>"
        )
    return "".join(rows)


def _phase_group_row(label: str, group: dict[str, Any]) -> str:
    ratio = group["phase_observed_read_clear_ratio"]
    ratio_text = "—" if ratio is None else f"{ratio * 100:.4f}%"
    request_share = _format_share_bounds(
        group["phase_icache_request_lower_bound_share_of_submit"],
        group["phase_icache_request_upper_bound_share_of_submit"],
    )
    miss_share = _format_share_bounds(
        group["phase_icache_miss_lower_bound_share_of_submit"],
        group["phase_icache_miss_upper_bound_share_of_submit"],
    )
    time_share = group.get("phase_time_share_of_submit")
    time_share_text = "—" if time_share is None else f"{time_share * 100:.4f}%"
    phase_time_text = (
        "—"
        if group.get("phase_elapsed_per_core_us") is None
        else f"{group['phase_elapsed_per_core_us']:,.3f} µs"
    )
    per_call_text = (
        "—"
        if group.get("phase_elapsed_per_call_ns") is None
        else f"{group['phase_elapsed_per_call_ns']:,.3f} ns"
    )
    return (
        "<tr>"
        f"<td>{_escape(label)}</td>"
        f"<td>{int(group['phase_calls_sum']):,}</td>"
        f"<td>{phase_time_text}</td>"
        f"<td>{per_call_text}</td>"
        f"<td>{time_share_text}</td>"
        f"<td>{group['phase_icache_requests_lower_bound_sum']:,}..{group['phase_icache_requests_upper_bound_sum']:,}</td>"
        f"<td>{group['phase_icache_requests_lower_bound_per_core']:,.3f}..{group['phase_icache_requests_upper_bound_per_core']:,.3f}</td>"
        f"<td>{request_share}</td>"
        f"<td>{group['phase_icache_misses_lower_bound_sum']:,}..{group['phase_icache_misses_upper_bound_sum']:,}</td>"
        f"<td>{group['phase_icache_misses_lower_bound_per_core']:,.3f}..{group['phase_icache_misses_upper_bound_per_core']:,.3f}</td>"
        f"<td>{miss_share}</td>"
        f"<td>{int(group['shadow_request_loss_sum']):,} / {int(group['shadow_miss_loss_sum']):,}</td>"
        f"<td>{ratio_text}</td>"
        "</tr>"
    )


def _format_share_bounds(lower: float | None, upper: float | None) -> str:
    """将局部事件占同组完整 Submit primary 的比例格式化为区间。"""

    if lower is None or upper is None:
        return "—"
    return f"{lower * 100:.4f}%..{upper * 100:.4f}%"


def _phase_share_metric(
    group_label: str,
    label: str,
    metric_class: str,
    lower_share: float | None,
    upper_share: float | None,
    lower_count: int,
    upper_count: int,
) -> str:
    """生成一个以完整 Submit primary 为 100% 的局部占比区间条。"""

    if lower_share is None or upper_share is None:
        return (
            f'<div class="phase-share-metric" data-phase-group="{_escape(group_label.lower())}" '
            f'data-metric="{_escape(metric_class)}" data-lower-share="" data-upper-share="">'
            f'<div class="phase-share-head"><strong>{_escape(label)}</strong><span>—</span></div>'
            '<div class="phase-share-unavailable">完整窗口分母为 0，比例不可计算</div>'
            "</div>"
        )
    lower_percent = lower_share * 100.0
    upper_percent = upper_share * 100.0
    bounds_text = f"{lower_percent:.4f}%..{upper_percent:.4f}%"
    aria_label = (
        f"{label} 局部占完整 Submit primary，"
        f"下界 {lower_percent:.4f}%，上界 {upper_percent:.4f}%"
    )
    return f"""
      <div class="phase-share-metric" data-phase-group="{_escape(group_label.lower())}" data-metric="{_escape(metric_class)}" data-lower-share="{lower_share:.12f}" data-upper-share="{upper_share:.12f}">
        <div class="phase-share-head"><strong>{_escape(label)}</strong><span>{bounds_text}</span></div>
        <div class="phase-share-track share-{_escape(metric_class)}" role="img" aria-label="{_escape(aria_label)}">
          <span class="phase-share-upper" style="width:{upper_percent:.6f}%"></span>
          <span class="phase-share-lower" style="width:{lower_percent:.6f}%"></span>
          <span class="phase-share-upper-marker" style="left:{upper_percent:.6f}%"></span>
        </div>
        <div class="phase-share-axis"><span>0%</span><span>50%</span><span>100%</span></div>
        <div class="phase-share-count">事件数 {lower_count:,}..{upper_count:,}</div>
      </div>
"""


def _phase_time_metric(group_label: str, group: dict[str, Any]) -> str:
    """生成所选阶段的逐核累计时间占比；该值是单点观察，不伪造上下界。"""

    share = group.get("phase_time_share_of_submit")
    if share is None:
        return (
            f'<div class="phase-share-metric" data-phase-group="{_escape(group_label.lower())}" '
            'data-metric="time" data-time-share="">'
            '<div class="phase-share-head"><strong>阶段时间</strong><span>—</span></div>'
            '<div class="phase-share-unavailable">本次 raw 未采集阶段 SYS_CNT，不能由 I-cache 或 PMU total 反推</div>'
            "</div>"
        )
    percent = float(share) * 100.0
    phase_per_core_us = float(group["phase_elapsed_per_core_us"])
    per_call_ns = group.get("phase_elapsed_per_call_ns")
    per_call_text = "—" if per_call_ns is None else f"{float(per_call_ns):,.3f} ns/call"
    aria_label = f"{group_label} 阶段逐核累计时间占同核完整 Submit {percent:.4f}%"
    return f"""
      <div class="phase-share-metric" data-phase-group="{_escape(group_label.lower())}" data-metric="time" data-time-share="{float(share):.12f}">
        <div class="phase-share-head"><strong>阶段时间</strong><span>{percent:.4f}%</span></div>
        <div class="phase-share-track share-time" role="img" aria-label="{_escape(aria_label)}">
          <span class="phase-time-value" style="width:{percent:.6f}%"></span>
        </div>
        <div class="phase-share-axis"><span>0%</span><span>50%</span><span>100%</span></div>
        <div class="phase-share-count">平均 {phase_per_core_us:,.3f} µs/核 · {per_call_text}</div>
      </div>
"""


def _phase_share_card(label: str, group: dict[str, Any]) -> str:
    """把一个角色组的时间/request/miss 局部占比放在同一张响应式卡片中。"""

    time_metric = _phase_time_metric(label, group)
    request_metric = _phase_share_metric(
        label,
        "I-cache request",
        "request",
        group["phase_icache_request_lower_bound_share_of_submit"],
        group["phase_icache_request_upper_bound_share_of_submit"],
        int(group["phase_icache_requests_lower_bound_sum"]),
        int(group["phase_icache_requests_upper_bound_sum"]),
    )
    miss_metric = _phase_share_metric(
        label,
        "I-cache miss",
        "miss",
        group["phase_icache_miss_lower_bound_share_of_submit"],
        group["phase_icache_miss_upper_bound_share_of_submit"],
        int(group["phase_icache_misses_lower_bound_sum"]),
        int(group["phase_icache_misses_upper_bound_sum"]),
    )
    role_class = "" if label == "ALL" else f" role-{label.lower()}"
    return f"""
    <article class="phase-share-card" data-phase-group="{_escape(label.lower())}">
      <div class="phase-share-card-title"><span class="phase-share-role{role_class}">{_escape(label)}</span><span>{int(group['phase_calls_sum']):,} calls · {group['phase_calls_per_core']:,.0f}/核</span></div>
      {time_metric}
      {request_metric}
      {miss_metric}
    </article>
"""


def _pmu_stat_row(
    label: str,
    stat: str,
    total_cycles: int | float,
    scalar_cycles: int | float,
    cycles_per_ns: float,
    decimals: int = 0,
) -> str:
    """生成一行紧凑的 PMU 分布统计，同时保留 raw cycle 与校准时间。"""

    return (
        f'<tr data-stat="{_escape(stat)}">'
        f"<th scope=\"row\">{_escape(label)}</th>"
        f"<td>{_cycle_time_cell(total_cycles, cycles_per_ns, decimals)}</td>"
        f"<td>{_cycle_time_cell(scalar_cycles, cycles_per_ns, decimals)}</td>"
        "</tr>"
    )


def _pmu_role_card(label: str, group: dict[str, Any]) -> str:
    """按角色生成纵向分布卡片，避免横向堆叠十余列。"""

    role_name = label.lower()
    role_class = "" if label == "ALL" else f" role-{role_name}"
    cycles_per_ns = float(group["cycles_per_ns"])
    rows = "".join(
        (
            _pmu_stat_row(
                "最小值", "min", group["total_min"], group["scalar_min"], cycles_per_ns
            ),
            _pmu_stat_row(
                "平均值", "mean", group["total_per_core"], group["scalar_per_core"],
                cycles_per_ns, 2
            ),
            _pmu_stat_row(
                "最大值", "max", group["total_max"], group["scalar_max"], cycles_per_ns
            ),
        )
    )
    return f"""
    <article class="pmu-role-card" data-pmu-group="{_escape(role_name)}">
      <div class="pmu-role-title"><span class="pmu-role-label{role_class}">{_escape(label)}</span><span>{_format_count(group['cores'])} 核 · {cycles_per_ns:.6f} cycles/ns</span></div>
      <table class="pmu-compact-table">
        <thead><tr><th>统计</th><th>PMU total</th><th>scalar busy</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div class="pmu-role-foot"><span>Σscalar/Σtotal：<strong>{_format_rate(group['scalar_share'])}</strong></span><span>非 Scalar-busy 残余/核：{_cycle_time_cell(group['non_scalar_busy_per_core'], cycles_per_ns, 2)}</span></div>
    </article>
"""


def render_report(
    input_path: Path,
    output_path: Path | None = None,
    miss_penalty_ns: float = 90.0,
    pmu_cycles_per_ns: float = DEFAULT_PMU_CYCLES_PER_NS,
    aic_pmu_cycles_per_ns: float = DEFAULT_AIC_PMU_CYCLES_PER_NS,
    aiv_pmu_cycles_per_ns: float = DEFAULT_AIV_PMU_CYCLES_PER_NS,
) -> str:
    """先通过独立 analyzer 门禁，再生成一个无需网络的完整 HTML 字符串。"""

    for name, value in (
        ("pmu_cycles_per_ns", pmu_cycles_per_ns),
        ("aic_pmu_cycles_per_ns", aic_pmu_cycles_per_ns),
        ("aiv_pmu_cycles_per_ns", aiv_pmu_cycles_per_ns),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    input_path = Path(input_path)
    output_path = default_output_path(input_path) if output_path is None else Path(output_path)
    # 原始件只读取一次；analyzer 与页面元数据都基于同一份私有快照，避免生成期间
    # raw 被替换后，统计值、元数据和页面 SHA256 分属不同版本。
    raw_bytes = input_path.read_bytes()
    raw_digest = hashlib.sha256(raw_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="pa-submit-pmu-report-") as snapshot_directory:
        snapshot_path = Path(snapshot_directory) / input_path.name
        snapshot_path.write_bytes(raw_bytes)
        analysis = analyze([snapshot_path], miss_penalty_ns)
        capture = load_capture(snapshot_path)
    if capture.schema_version not in (4, 5):
        raise ValueError("HTML report requires submit-pmu schema-v4/v5 input")

    data = capture.data
    configuration = data["configuration"]
    validation = data["validation"]
    records = data["records"]
    aic_records = [record for record in records if record["role"] == "aic"]
    aiv_records = [record for record in records if record["role"] == "aiv"]
    per_run = analysis["per_run"][0]
    all_metrics = _group_metrics(per_run["groups"]["all"], records, pmu_cycles_per_ns)
    aic = _group_metrics(per_run["groups"]["aic"], aic_records, aic_pmu_cycles_per_ns)
    aiv = _group_metrics(per_run["groups"]["aiv"], aiv_records, aiv_pmu_cycles_per_ns)
    cycles_per_ns_by_role = {
        "aic": aic_pmu_cycles_per_ns,
        "aiv": aiv_pmu_cycles_per_ns,
    }
    submit_us = float(per_run["submit_span_us"])
    aic_request_per_core = float(aic["requests_per_core"])
    aiv_request_per_core = float(aiv["requests_per_core"])
    request_delta_percent = (
        None
        if aic_request_per_core == 0.0
        else (aiv_request_per_core / aic_request_per_core - 1.0) * 100.0
    )
    aic_miss_per_core = float(aic["misses_per_core"])
    aiv_miss_per_core = float(aiv["misses_per_core"])
    delta_misses = aiv_miss_per_core - aic_miss_per_core
    delta_percent = (
        None if aic_miss_per_core == 0.0 else delta_misses / aic_miss_per_core * 100.0
    )
    delta_rate_pp = (float(aiv["miss_rate"]) - float(aic["miss_rate"])) * 100.0
    request_comparison = (
        "AIC request/core 为 0，AIV 相对变化不可计算"
        if request_delta_percent is None
        else f"AIV 的 request/core 相比 AIC {'高' if request_delta_percent >= 0 else '低'} "
        f"{abs(request_delta_percent):.2f}%"
    )
    miss_comparison = (
        "AIC miss/core 为 0，AIV 相对变化不可计算"
        if delta_percent is None
        else f"miss/core {'高' if delta_percent >= 0 else '低'} {abs(delta_percent):.2f}%"
        f"（{delta_misses:+,.2f}/core）"
    )
    rate_comparison = (
        f"miss rate {'高' if delta_rate_pp >= 0 else '低'} {abs(delta_rate_pp):.3f} 个百分点"
    )
    total_requests = int(per_run["groups"]["all"]["icache_requests_sum"])
    total_misses = int(per_run["groups"]["all"]["icache_misses_sum"])
    phase = str(configuration["compiled_phase"])
    workload = configuration.get("winner_workload") or {}
    workload_counts = workload.get("counts") or {}
    exact_records = int(validation["shadow_primary_match_records"])
    bounded_records = int(validation["shadow_primary_bounded_records"])
    workers = int(configuration["workers"])
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    raw_link = _raw_href(input_path, output_path)

    total_plot = _distribution_svg(
        records,
        "total_cycles",
        "逐物理核 PMU total cycle",
        "每核 Submit gate 内的 PMU raw total；96 核求和是 core-work，不是墙钟时间。",
        cycles_per_ns_by_role=cycles_per_ns_by_role,
    )
    scalar_plot = _distribution_svg(
        records,
        "scalar_busy",
        "逐物理核 scalar busy cycle",
        "CNT2 scalar_instr_busy(0x001) 的每核累计；它不包含全部等待周期。",
        cycles_per_ns_by_role=cycles_per_ns_by_role,
    )
    request_plot = _distribution_svg(
        records,
        "icache_requests",
        "逐物理核 I-cache request",
        "96 个实测物理子核的 request 离散分布；AIC 为圆点，AIV 为方点。",
    )
    miss_plot = _distribution_svg(
        records,
        "icache_misses",
        "逐物理核 I-cache miss",
        "96 个实测物理子核的 miss 离散分布；AIC 为圆点，AIV 为方点。",
    )
    rate_plot = _distribution_svg(
        records,
        "icache_miss_rate",
        "逐物理核 I-cache miss rate",
        "每核 miss/request，仅用于观察离散分布；汇总 rate 仍按总 miss 除以总 request。",
    )
    per_core_rows = _per_core_rows(records, cycles_per_ns_by_role)
    pmu_role_cards = "".join(
        (
            _pmu_role_card("ALL", all_metrics),
            _pmu_role_card("AIC", aic),
            _pmu_role_card("AIV", aiv),
        )
    )
    if phase == "none":
        shadow_badge = f"PRIMARY ↔ SHADOW EXACT {exact_records}/{workers}"
        phase_front_section = """
  <section class="panel phase-overview-front phase-disabled" aria-label="局部阶段总览">
    <h2>局部阶段总览：none</h2>
    <p><strong>不适用：</strong>该 ELF 未编译局部阶段。request、miss 和阶段时间都不能作为某个局部阶段的 0% 结果；本报告只提供完整 Submit 基准。</p>
  </section>
"""
        phase_section = """
  <section class="panel phase-disabled">
    <h2>局部 phase：none</h2>
    <p>未执行 running read-clear 或阶段 SYS_CNT 边界；raw 中局部字段按契约为 0，但语义是“未选择阶段”，不是某个阶段实测为 0。</p>
  </section>
"""
    else:
        shadow_badge = f"SHADOW BOUNDED {bounded_records}/{workers}（exact {exact_records}/{workers}）"
        phase_groups = per_run["groups"]
        phase_share_cards = "".join(
            (
                _phase_share_card("ALL", phase_groups["all"]),
                _phase_share_card("AIC", phase_groups["aic"]),
                _phase_share_card("AIV", phase_groups["aiv"]),
            )
        )
        phase_front_section = f"""
  <section class="panel phase-overview-front" aria-label="局部阶段总览">
    <h2>局部阶段总览：{_escape(phase)}</h2>
    <p class="phase-share-note">时间占比是 <code>Σ阶段 SYS_CNT / Σ同核首个 submit_begin 计时点到末个 submit_end 计时点 SYS_CNT</code>，分别在 ALL/AIC/AIV 内先求和再相除。首个 submit_begin 位于 BeginSubmit 上下文初始化之后，末个 submit_end 位于返回之前。它是逐核累计 core-time 构成，不是该阶段占全局约 5 ms 墙钟的切片。request/miss 仍显示同一 ELF primary-shadow 形成的下界..上界；阶段时间是单点观察值。</p>
    <div class="phase-share-grid">{phase_share_cards}</div>
  </section>
"""
        phase_plot = _distribution_svg(
            records,
            "phase_icache_misses",
            f"{phase} 局部 I-cache miss 观测下界",
            "running read-clear 的逐核观测下界；上界还需加该核 primary-shadow residual。",
        )
        phase_rows = "".join(
            (
                _phase_group_row("ALL", phase_groups["all"]),
                _phase_group_row("AIC", phase_groups["aic"]),
                _phase_group_row("AIV", phase_groups["aiv"]),
            )
        )
        phase_section = f"""
  <h2>局部阶段详细数据：{_escape(phase)}</h2>
  <section class="panel phase-panel">
    <div class="phase-warning"><strong>带边界扰动的诊断区间。</strong>每次调用新增两次 SYS_CNT；时间起点位于 begin shadow read-clear 之后，终点位于 end shadow read-clear 之前，因此不包含两侧 ld_dev，但包含时间戳边界本身的扰动。request/miss 下界是直接观测值，上界是下界加本核 primary-shadow residual；不同 phase ELF 的局部值不能相加，也不能从 none 相减得到无扰动净值。</div>
    <details class="phase-table-details">
      <summary>展开 ALL / AIC / AIV 完整数字表</summary>
      <div class="phase-table-scroll">
        <table class="phase-table">
          <thead><tr><th>分组</th><th>calls</th><th>阶段时间/core</th><th>阶段时间/call</th><th>阶段 core-time / Submit</th><th>request sum 下界..上界</th><th>request/core 下界..上界</th><th>局部 request / Submit primary</th><th>miss sum 下界..上界</th><th>miss/core 下界..上界</th><th>局部 miss / Submit primary</th><th>shadow loss req/miss</th><th>observed miss/request</th></tr></thead>
          <tbody>{phase_rows}</tbody>
        </table>
      </div>
    </details>
    <h3>逐核 phase miss 下界</h3>
    <div class="phase-plot-scroll">{phase_plot}</div>
  </section>
"""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Standalone PA Submit I-cache 报告</title>
  <style>
    :root {{ color-scheme: light; --aic:{AIC_COLOR}; --aiv:{AIV_COLOR}; --ink:#0f172a; --muted:#64748b; --line:#cbd5e1; --panel:#f8fafc; --pass:#166534; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; color:var(--ink); background:#eef2f7; line-height:1.5; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:24px auto 64px; }}
    h1,h2,h3 {{ line-height:1.2; }}
    h1 {{ margin:0 0 6px; font-size:clamp(1.65rem,4vw,2.4rem); }}
    h2 {{ margin:34px 0 14px; font-size:1.35rem; }}
    .subtitle,.muted {{ color:var(--muted); }}
    .badges {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0; }}
    .badge {{ border:1px solid #86efac; background:#f0fdf4; color:var(--pass); border-radius:999px; padding:4px 10px; font-weight:700; font-size:.86rem; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
    .card,.panel {{ min-width:0; background:white; border:1px solid var(--line); border-radius:12px; box-shadow:0 4px 18px #0f172a0d; }}
    .card {{ padding:16px; }}
    .card .label {{ color:var(--muted); font-size:.86rem; }}
    .card .value {{ margin-top:4px; font-size:1.55rem; font-weight:750; font-variant-numeric:tabular-nums; }}
    .cycle-value,.cycle-time {{ display:block; }}
    .cycle-time {{ color:var(--muted); font-size:.82rem; }}
    .panel {{ padding:18px; overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:9px 11px; text-align:right; border-bottom:1px solid #e2e8f0; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    thead th {{ background:#f8fafc; color:#475569; font-size:.82rem; text-transform:none; }}
    .role {{ display:inline-block; min-width:42px; text-align:center; border-radius:999px; color:white; padding:2px 7px; font-size:.78rem; font-weight:700; }}
    .role-aic {{ background:var(--aic); }} .role-aiv {{ background:var(--aiv); }}
    .pmu-role-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(330px,100%),1fr)); gap:12px; width:100%; }}
    .pmu-role-card {{ min-width:0; padding:14px; border:1px solid #dbe3ee; border-radius:10px; background:#f8fafc; }}
    .pmu-role-title {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; margin-bottom:10px; color:var(--muted); font-size:.82rem; font-variant-numeric:tabular-nums; }}
    .pmu-role-label {{ display:inline-block; min-width:44px; padding:3px 9px; border-radius:999px; background:#334155; color:white; text-align:center; font-weight:800; }}
    .pmu-role-label.role-aic {{ background:var(--aic); }} .pmu-role-label.role-aiv {{ background:var(--aiv); }}
    .pmu-compact-table {{ table-layout:fixed; font-size:.86rem; }}
    .pmu-compact-table th,.pmu-compact-table td {{ padding:7px 6px; white-space:normal; vertical-align:top; }}
    .pmu-compact-table th:first-child {{ width:23%; }}
    .pmu-compact-table .cycle-value,.pmu-compact-table .cycle-time {{ white-space:nowrap; }}
    .pmu-role-foot {{ display:grid; gap:5px; margin-top:10px; color:#475569; font-size:.82rem; }}
    .pmu-role-foot .cycle-value,.pmu-role-foot .cycle-time {{ display:inline; white-space:nowrap; }}
    .pmu-role-foot .cycle-time {{ margin-left:5px; }}
    .table-scroll {{ width:100%; max-width:100%; overflow-x:auto; overscroll-behavior-x:contain; }}
    .icache-table {{ table-layout:fixed; min-width:760px; }}
    .icache-table th,.icache-table td {{ padding:7px 6px; white-space:normal; overflow-wrap:anywhere; }}
    .insight {{ margin-top:12px; padding:12px 14px; border-left:4px solid var(--aiv); background:#fff7ed; }}
    .phase-warning {{ margin-bottom:14px; padding:12px 14px; border-left:4px solid #7c3aed; background:#f5f3ff; }}
    .phase-panel {{ overflow:hidden; }}
    .phase-overview-front {{ margin:0 0 20px; overflow:hidden; }}
    .phase-overview-front h2 {{ margin-top:0; }}
    .phase-share-title {{ margin:20px 0 5px; }}
    .phase-share-note {{ max-width:900px; margin:0 0 14px; color:var(--muted); }}
    .phase-share-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr)); gap:12px; width:100%; }}
    .phase-share-card {{ min-width:0; padding:14px; border:1px solid #dbe3ee; border-radius:10px; background:#f8fafc; }}
    .phase-share-card-title {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; margin-bottom:13px; color:var(--muted); font-size:.86rem; font-variant-numeric:tabular-nums; }}
    .phase-share-role {{ display:inline-block; min-width:44px; padding:3px 9px; border-radius:999px; background:#334155; color:white; text-align:center; font-weight:800; }}
    .phase-share-role.role-aic {{ background:var(--aic); }} .phase-share-role.role-aiv {{ background:var(--aiv); }}
    .phase-share-metric + .phase-share-metric {{ margin-top:16px; }}
    .phase-share-head {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:6px 12px; margin-bottom:6px; font-size:.9rem; font-variant-numeric:tabular-nums; }}
    .phase-share-head span {{ color:#334155; font-weight:750; }}
    .phase-share-track {{ position:relative; width:100%; height:19px; overflow:hidden; border:1px solid #cbd5e1; border-radius:999px; background:#e2e8f0; }}
    .phase-share-upper,.phase-share-lower {{ position:absolute; inset:0 auto 0 0; }}
    .share-request .phase-share-upper {{ background:#93c5fd; }} .share-request .phase-share-lower {{ background:#2563eb; }}
    .share-miss .phase-share-upper {{ background:#c4b5fd; }} .share-miss .phase-share-lower {{ background:#7c3aed; }}
    .phase-time-value {{ position:absolute; inset:0 auto 0 0; background:#0f766e; }}
    .phase-share-upper-marker {{ position:absolute; top:-1px; bottom:-1px; width:2px; transform:translateX(-1px); background:#0f172a; opacity:.75; }}
    .phase-share-axis {{ display:grid; grid-template-columns:repeat(3,1fr); margin-top:2px; color:#64748b; font-size:.68rem; }}
    .phase-share-axis span:nth-child(2) {{ text-align:center; }} .phase-share-axis span:last-child {{ text-align:right; }}
    .phase-share-count {{ margin-top:2px; color:var(--muted); font-size:.76rem; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
    .phase-share-unavailable {{ padding:8px; border-radius:8px; background:#e2e8f0; color:var(--muted); }}
    .phase-table-details {{ margin-top:18px; }}
    .phase-table-details summary {{ padding-bottom:8px; }}
    .phase-table-scroll {{ width:100%; max-width:100%; overflow-x:auto; border:1px solid #dbe3ee; border-radius:10px; overscroll-behavior-x:contain; }}
    .phase-table {{ min-width:1540px; }}
    .phase-table th,.phase-table td {{ padding:8px 10px; }}
    .phase-plot-scroll {{ width:100%; max-width:100%; overflow-x:auto; overscroll-behavior-x:contain; }}
    .phase-disabled {{ margin-top:34px; }} .phase-overview-front.phase-disabled {{ margin-top:0; }} .phase-disabled h2 {{ margin-top:0; }}
    .legend {{ display:flex; gap:18px; align-items:center; color:#475569; font-size:.88rem; margin-bottom:8px; }}
    .circle-key {{ width:10px; height:10px; border-radius:50%; background:var(--aic); }}
    .square-key {{ width:10px; height:10px; background:var(--aiv); }}
    .distribution {{ width:100%; min-width:760px; height:auto; display:block; }}
    .grid {{ stroke:{GRID_COLOR}; stroke-width:1; }} .tick {{ stroke:#64748b; }}
    .axis-label,.axis-title {{ fill:{TEXT_COLOR}; font-size:12px; }}
    .axis-title {{ font-size:13px; font-weight:650; }}
    .aic-point {{ fill:var(--aic); }} .aiv-point {{ fill:var(--aiv); }}
    .point {{ opacity:.82; outline:none; }} .point:hover,.point:focus {{ opacity:1; stroke:#020617; stroke-width:2; }}
    .warning {{ border:2px dashed #94a3b8; background:#f8fafc; border-radius:12px; padding:18px; }}
    .warning h2 {{ margin-top:0; }}
    details summary {{ cursor:pointer; font-weight:700; padding:4px 0 14px; }}
    a {{ color:#1d4ed8; }} code {{ overflow-wrap:anywhere; }}
    footer {{ margin-top:32px; color:#475569; font-size:.86rem; }}
    :focus-visible {{ outline:3px solid #0ea5e9; outline-offset:2px; }}
    @media (max-width:640px) {{ main {{ width:min(100% - 16px,1180px); margin-top:12px; }} .panel {{ padding:13px; }} .phase-share-grid,.pmu-role-grid {{ grid-template-columns:1fr; }} }}
    @media print {{ body {{ background:white; }} main {{ width:100%; margin:0; }} .card,.panel {{ box-shadow:none; break-inside:avoid; }} details {{ display:block; }} .phase-share-upper,.phase-share-lower,.phase-share-upper-marker,.phase-time-value {{ print-color-adjust:exact; -webkit-print-color-adjust:exact; }} }}
    @media (prefers-reduced-motion:reduce) {{ * {{ scroll-behavior:auto !important; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Standalone PA Submit I-cache 报告</h1>
    <div class="subtitle">submit-pmu / {_escape(phase)} / batch {_escape(configuration['batches'])} / {_escape(workload.get('mode'))} / QK,SF,PV,UP={_escape(workload_counts.get('qk'))},{_escape(workload_counts.get('sf'))},{_escape(workload_counts.get('pv'))},{_escape(workload_counts.get('up'))}</div>
    <div class="badges">
      <span class="badge">RAW → SUMMARY PASS</span>
      <span class="badge">语义与 PMU PASS</span>
      <span class="badge">OWNER RESTORE PASS</span>
      <span class="badge">{_escape(shadow_badge)}</span>
    </div>
  </header>

  {phase_front_section}

  <section class="cards" aria-label="关键指标">
    <div class="card"><div class="label">完整 Submit（最早开始 → 最晚结束）</div><div class="value">{submit_us / 1000:.6f} ms</div><div class="muted">96 核整体 Submit 耗时</div></div>
    <div class="card"><div class="label">96 核逐核 PMU total 平均值（校准）</div><div class="value">{float(all_metrics['total_per_core_us']):,.3f} µs</div><div class="muted">最小 {_cycles_to_us(all_metrics['total_min'], pmu_cycles_per_ns):,.3f} µs · 最大 {_cycles_to_us(all_metrics['total_max'], pmu_cycles_per_ns):,.3f} µs</div><div class="muted">{_format_per_core(all_metrics['total_per_core'])} raw cycle/核</div></div>
    <div class="card"><div class="label">96 核逐核 scalar busy 平均值（校准）</div><div class="value">{float(all_metrics['scalar_per_core_us']):,.3f} µs</div><div class="muted">最小 {_cycles_to_us(all_metrics['scalar_min'], pmu_cycles_per_ns):,.3f} µs · 最大 {_cycles_to_us(all_metrics['scalar_max'], pmu_cycles_per_ns):,.3f} µs</div><div class="muted">Σscalar/Σtotal={_format_rate(all_metrics['scalar_share'])}</div></div>
    <div class="card"><div class="label">完整 Submit primary I-cache request（96 核总和）</div><div class="value">{total_requests:,}</div><div class="muted">逐核平均 {_format_per_core(all_metrics['requests_per_core'])}</div><div class="muted">逐核最小 {_format_count(all_metrics['requests_min'])} · 逐核最大 {_format_count(all_metrics['requests_max'])}</div></div>
    <div class="card"><div class="label">完整 Submit primary I-cache miss（96 核总和）</div><div class="value">{total_misses:,}</div><div class="muted">逐核平均 {_format_per_core(all_metrics['misses_per_core'])}</div><div class="muted">逐核最小 {_format_count(all_metrics['misses_min'])} · 逐核最大 {_format_count(all_metrics['misses_max'])}</div></div>
    <div class="card"><div class="label">聚合 miss rate</div><div class="value">{_format_rate(float(all_metrics['miss_rate']))}</div></div>
    <div class="card"><div class="label">实测物理子核</div><div class="value">{workers}（32 AIC + 64 AIV）</div></div>
    <div class="card"><div class="label">Primary/Shadow</div><div class="value">exact {exact_records}/{workers}</div><div class="muted">bounded {bounded_records}/{workers}</div></div>
  </section>

  <h2>PMU cycle 频率校准</h2>
  <section class="panel">
    <p><strong>本机受控 cold/warm 同窗证据：</strong>PMU cycle_delta = {PMU_CALIBRATION_CYCLE_DELTA:,}，1 ns SYS_CNT tick_delta = {PMU_CALIBRATION_SYS_TICK_NS:,} ns；二者相除为 {PMU_CALIBRATION_CYCLE_DELTA / PMU_CALIBRATION_SYS_TICK_NS:.6f} cycles/ns（约 1.65 GHz）。</p>
    <p>当前报告换算频率：ALL = <strong>{pmu_cycles_per_ns:.6f}</strong>、AIC = <strong>{aic_pmu_cycles_per_ns:.6f}</strong>、AIV = <strong>{aiv_pmu_cycles_per_ns:.6f}</strong> cycles/ns。公式：<code>等效 µs = PMU cycles / (cycles/ns) / 1000</code>。</p>
    <p class="muted">SYS_CNT 仍按 1 ns/tick 解释；表中的 PMU 时间是每个物理子核 gate 内 cycle 的校准等效时间。它不是 96 核 cycle 求和后的墙钟时间，也不替代独立记录的完整 Submit 墙钟。</p>
  </section>

  <h2>PMU total 与 scalar busy</h2>
  <section class="panel">
    <div class="pmu-role-grid">{pmu_role_cards}</div>
    <div class="phase-warning"><strong>口径：</strong>每张卡只保留最小值、平均值和最大值。total 与 scalar busy 的最小/最大值分别从各自逐核分布独立取得，不保证来自同一个物理核，也不能相减成配对差值。total 是每个物理子核在 Submit gate 内的 PMU raw total cycle；求和代表 96 核 core-work。页面始终保留 raw cycle，旁边的 µs 只按本机实测频率换算。逐核最大值用于观察慢核，但仍不等同于“最早开始至最晚结束”的完整 Submit。scalar busy 是 CNT2 scalar_instr_busy(0x001)。卡片中的“非 Scalar-busy 残余”严格等于 total−scalar busy，<strong>它不是空闲时间，也不是 I-cache stall</strong>，其中还混有同步等待、engine 等待及其他未归因周期。受控微基准已观察到依赖返回的 atomic 等待大部分进入 scalar busy，而 I-cache refill 的额外周期大部分只进入 total。当前 A5/DAV3510 正式事件表和 CANN 9.1 上板输出均不提供 scalar_wait_ib_time/scalar_wait_time，报告不会用其他产品的 selector 猜测这两项。</div>
    <h3>PMU total cycle</h3>{total_plot}
    <h3>Scalar busy cycle</h3>{scalar_plot}
  </section>

  <h2>AIC 与 AIV 的 I-cache 对比</h2>
  <section class="panel">
    <div class="table-scroll">
    <table class="icache-table">
      <thead><tr><th>角色</th><th>核数</th><th>request min/core</th><th>request mean/core</th><th>request max/core</th><th>miss min/core</th><th>miss mean/core</th><th>miss max/core</th><th>Σmiss/Σrequest</th></tr></thead>
      <tbody>
        <tr><td><span class="role role-aic">AIC</span></td><td>{_format_count(aic['cores'])}</td><td>{_format_count(aic['requests_min'])}</td><td>{_format_per_core(aic['requests_per_core'])}</td><td>{_format_count(aic['requests_max'])}</td><td>{_format_count(aic['misses_min'])}</td><td>{_format_per_core(aic['misses_per_core'])}</td><td>{_format_count(aic['misses_max'])}</td><td>{_format_rate(aic['miss_rate'])}</td></tr>
        <tr><td><span class="role role-aiv">AIV</span></td><td>{_format_count(aiv['cores'])}</td><td>{_format_count(aiv['requests_min'])}</td><td>{_format_per_core(aiv['requests_per_core'])}</td><td>{_format_count(aiv['requests_max'])}</td><td>{_format_count(aiv['misses_min'])}</td><td>{_format_per_core(aiv['misses_per_core'])}</td><td>{_format_count(aiv['misses_max'])}</td><td>{_format_rate(aiv['miss_rate'])}</td></tr>
      </tbody>
    </table>
    </div>
    <div class="insight">{_escape(request_comparison)}；{_escape(miss_comparison)}；聚合 {_escape(rate_comparison)}。</div>
  </section>

  <h2>逐物理核分布</h2>
  <section class="panel">
    <div class="legend"><span class="circle-key" aria-hidden="true"></span>AIC 圆点 <span class="square-key" aria-hidden="true"></span>AIV 方点</div>
    <h3>I-cache request</h3>{request_plot}
    <h3>I-cache miss</h3>{miss_plot}
    <h3>每核 miss rate</h3>{rate_plot}
  </section>

  {phase_section}

  <section class="warning">
    <h2>假设性 core-equivalent，不是 Submit 墙钟损失</h2>
    <p>AIC：{_format_per_core(aic['misses_per_core'])} miss/core × {miss_penalty_ns:.3f} ns = <strong>{float(aic['serial_equivalent_us']):,.3f} µs/core-equivalent</strong></p>
    <p>AIV：{_format_per_core(aiv['misses_per_core'])} miss/core × {miss_penalty_ns:.3f} ns = <strong>{float(aiv['serial_equivalent_us']):,.3f} µs/core-equivalent</strong></p>
    <p>该标尺不可跨 96 核相加，也不可直接从 {submit_us / 1000:.6f} ms Submit 中扣除。各核并行，miss 可能重叠或被流水/等待隐藏；实际收益必须由相同语义优化前后的 ΔSubmit 与 Δmiss/core 成对实验确认。</p>
  </section>

  <h2>96 核精确数据</h2>
  <section class="panel">
    <details>
      <summary>展开逐核表格</summary>
      <div class="table-scroll">
      <table>
        <thead><tr><th>worker</th><th>physical</th><th>role</th><th>block</th><th>lane</th><th>PMU total<br>cycle / 等效 µs</th><th>scalar busy<br>cycle / 等效 µs</th><th>scalar/total</th><th>requests</th><th>misses</th><th>per-core rate</th><th>primary=shadow</th><th>trusted</th></tr></thead>
        <tbody>{per_core_rows}</tbody>
      </table>
      </div>
    </details>
    <p class="muted">聚合 miss rate 使用 Σmiss/Σrequest，不平均逐核百分比。request/miss 的 min、mean、max 都从同组逐核 raw 计算；request 与 miss 的极值不保证来自同一个物理核。</p>
  </section>

  <footer>
    <div>原始件：<a href="{_escape(raw_link)}"><code>{_escape(input_path.name)}</code></a></div>
    <div>capture_id：<code>{_escape(data['capture'].get('capture_id'))}</code></div>
    <div>SHA-256：<code>{raw_digest}</code></div>
    <div>生成时间：{generated_at}；生成器：pmu_html_report schema v{REPORT_VERSION}</div>
  </footer>
</main>
</body>
</html>
"""


def write_report(
    input_path: Path,
    output_path: Path | None = None,
    miss_penalty_ns: float = 90.0,
    pmu_cycles_per_ns: float = DEFAULT_PMU_CYCLES_PER_NS,
    aic_pmu_cycles_per_ns: float = DEFAULT_AIC_PMU_CYCLES_PER_NS,
    aiv_pmu_cycles_per_ns: float = DEFAULT_AIV_PMU_CYCLES_PER_NS,
) -> Path:
    """在完整 HTML 构造成功后原子发布，失败时不留下半截报告。"""

    input_path = Path(input_path)
    output_path = default_output_path(input_path) if output_path is None else Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("HTML output path must differ from raw JSON input")
    document = render_report(
        input_path,
        output_path,
        miss_penalty_ns,
        pmu_cycles_per_ns,
        aic_pmu_cycles_per_ns,
        aiv_pmu_cycles_per_ns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary.write(document)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.fchmod(temporary.fileno(), 0o644)
            temporary_name = temporary.name
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="一份已完成的 submit-pmu raw JSON")
    parser.add_argument("-o", "--output", type=Path, help="HTML 输出路径；默认由 *_raw.json 推导")
    parser.add_argument(
        "--icache-miss-ns",
        type=float,
        default=90.0,
        help="受控 cold/warm 串行标尺，仅用于 core-equivalent（默认 90）",
    )
    parser.add_argument(
        "--pmu-cycles-per-ns",
        type=float,
        default=DEFAULT_PMU_CYCLES_PER_NS,
        help=f"ALL PMU cycle 校准频率（默认 {DEFAULT_PMU_CYCLES_PER_NS:.6f} cycles/ns）",
    )
    parser.add_argument(
        "--aic-pmu-cycles-per-ns",
        type=float,
        default=DEFAULT_AIC_PMU_CYCLES_PER_NS,
        help=f"AIC PMU cycle 校准频率（默认 {DEFAULT_AIC_PMU_CYCLES_PER_NS:.6f} cycles/ns）",
    )
    parser.add_argument(
        "--aiv-pmu-cycles-per-ns",
        type=float,
        default=DEFAULT_AIV_PMU_CYCLES_PER_NS,
        help=f"AIV PMU cycle 校准频率（默认 {DEFAULT_AIV_PMU_CYCLES_PER_NS:.6f} cycles/ns）",
    )
    arguments = parser.parse_args(argv)
    if not math.isfinite(arguments.icache_miss_ns) or arguments.icache_miss_ns <= 0:
        parser.error("--icache-miss-ns must be finite and positive")
    for option, value in (
        ("--pmu-cycles-per-ns", arguments.pmu_cycles_per_ns),
        ("--aic-pmu-cycles-per-ns", arguments.aic_pmu_cycles_per_ns),
        ("--aiv-pmu-cycles-per-ns", arguments.aiv_pmu_cycles_per_ns),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{option} must be finite and positive")
    try:
        output = write_report(
            arguments.input,
            arguments.output,
            arguments.icache_miss_ns,
            arguments.pmu_cycles_per_ns,
            arguments.aic_pmu_cycles_per_ns,
            arguments.aiv_pmu_cycles_per_ns,
        )
    except (OSError, ValueError) as error:
        print(f"PMU HTML report failed: {error}", file=sys.stderr)
        return 1
    print(f"[PMU-HTML] raw={arguments.input} report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
