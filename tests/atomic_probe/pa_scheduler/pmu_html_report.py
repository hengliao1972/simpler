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


REPORT_VERSION = 2
AIC_COLOR = "#2563eb"
AIV_COLOR = "#ea580c"
GRID_COLOR = "#cbd5e1"
TEXT_COLOR = "#334155"


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


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _raw_href(input_path: Path, output_path: Path) -> str:
    relative = os.path.relpath(input_path, output_path.parent)
    return quote(relative.replace(os.sep, "/"), safe="/")


def _group_metrics(
    analysis_group: dict[str, Any], raw_group: dict[str, Any]
) -> dict[str, int | float]:
    cores = int(analysis_group["cores"])
    total_sum = int(analysis_group["total_cycles_sum"])
    scalar_sum = int(analysis_group["scalar_busy_sum"])
    return {
        "cores": cores,
        "total_sum": total_sum,
        "total_per_core": total_sum / cores,
        "total_median": raw_group["total_cycles"]["median"],
        "total_p95": raw_group["total_cycles"]["p95"],
        "scalar_sum": scalar_sum,
        "scalar_per_core": scalar_sum / cores,
        "scalar_median": raw_group["scalar_busy"]["median"],
        "scalar_p95": raw_group["scalar_busy"]["p95"],
        "scalar_share": 0.0 if total_sum == 0 else scalar_sum / total_sum,
        "non_scalar_busy_per_core": (total_sum - scalar_sum) / cores,
        "requests_per_core": analysis_group["icache_requests_per_core"],
        "requests_p95": raw_group["icache_requests"]["p95"],
        "misses_per_core": analysis_group["icache_misses_per_core"],
        "misses_median": analysis_group["icache_misses_per_core_median"],
        "misses_p95": analysis_group["icache_misses_per_core_p95"],
        "misses_max": raw_group["icache_misses"]["max"],
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
    p95_by_role: dict[str, float] | None = None,
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

    if p95_by_role is not None:
        for role, color in (("aic", AIC_COLOR), ("aiv", AIV_COLOR)):
            value = p95_by_role[role]
            y = y_position(value)
            fragments.append(
                f'<line class="p95-line" style="stroke:{color}" x1="{left}" y1="{y:.2f}" '
                f'x2="{width-right}" y2="{y:.2f}" />'
            )
            fragments.append(
                f'<text class="p95-label" style="fill:{color}" x="{width-right-4}" '
                f'y="{y-5:.2f}" text-anchor="end">{role.upper()} p95 '
                f'{_escape(_plot_label(metric, value))}</text>'
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


def _per_core_rows(records: Sequence[dict[str, Any]]) -> str:
    rows: list[str] = []
    for record in sorted(records, key=lambda item: int(item["physical_core_id"])):
        requests = int(record["icache_requests"])
        misses = int(record["icache_misses"])
        total = int(record["total_cycles"])
        scalar = int(record["scalar_busy"])
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
            f"<td>{total:,}</td>"
            f"<td>{scalar:,}</td>"
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
    return (
        "<tr>"
        f"<td>{_escape(label)}</td>"
        f"<td>{int(group['phase_calls_sum']):,}</td>"
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


def _phase_share_card(label: str, group: dict[str, Any]) -> str:
    """把一个角色组的 request/miss 局部占比放在同一张响应式卡片中。"""

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
      <div class="phase-share-card-title"><span class="phase-share-role{role_class}">{_escape(label)}</span><span>{int(group['phase_calls_sum']):,} calls</span></div>
      {request_metric}
      {miss_metric}
    </article>
"""


def render_report(
    input_path: Path,
    output_path: Path | None = None,
    miss_penalty_ns: float = 90.0,
) -> str:
    """先通过独立 analyzer 门禁，再生成一个无需网络的完整 HTML 字符串。"""

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
    if capture.schema_version != 4:
        raise ValueError("HTML report requires submit-pmu schema-v4 input")

    data = capture.data
    configuration = data["configuration"]
    validation = data["validation"]
    summary = data["summary"]
    records = data["records"]
    per_run = analysis["per_run"][0]
    all_metrics = _group_metrics(per_run["groups"]["all"], summary["all"])
    aic = _group_metrics(per_run["groups"]["aic"], summary["aic"])
    aiv = _group_metrics(per_run["groups"]["aiv"], summary["aiv"])
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
        {
            "aic": float(summary["aic"]["total_cycles"]["p95"]),
            "aiv": float(summary["aiv"]["total_cycles"]["p95"]),
        },
    )
    scalar_plot = _distribution_svg(
        records,
        "scalar_busy",
        "逐物理核 scalar busy cycle",
        "CNT2 scalar_instr_busy(0x001) 的每核累计；它不包含全部等待周期。",
        {
            "aic": float(summary["aic"]["scalar_busy"]["p95"]),
            "aiv": float(summary["aiv"]["scalar_busy"]["p95"]),
        },
    )
    request_plot = _distribution_svg(
        records,
        "icache_requests",
        "逐物理核 I-cache request",
        "96 个实测物理子核的 request 离散分布；AIC 为圆点，AIV 为方点。",
        {
            "aic": float(summary["aic"]["icache_requests"]["p95"]),
            "aiv": float(summary["aiv"]["icache_requests"]["p95"]),
        },
    )
    miss_plot = _distribution_svg(
        records,
        "icache_misses",
        "逐物理核 I-cache miss",
        "96 个实测物理子核的 miss 离散分布；虚线是各角色 nearest-rank p95。",
        {"aic": float(aic["misses_p95"]), "aiv": float(aiv["misses_p95"])},
    )
    rate_plot = _distribution_svg(
        records,
        "icache_miss_rate",
        "逐物理核 I-cache miss rate",
        "每核 miss/request，仅用于观察离散分布；汇总 rate 仍按总 miss 除以总 request。",
    )
    per_core_rows = _per_core_rows(records)
    if phase == "none":
        shadow_badge = f"PRIMARY ↔ SHADOW EXACT {exact_records}/{workers}"
        phase_section = """
  <section class="panel phase-disabled">
    <h2>局部 phase：none</h2>
    <p>完整 Submit 中未执行任何 running read-clear 边界；局部 calls、request 和 miss 全部为 0。此构建是完整 Submit I-cache 的主观察口径。</p>
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
        phase_plot = _distribution_svg(
            records,
            "phase_icache_misses",
            f"{phase} 局部 I-cache miss 观测下界",
            "running read-clear 的逐核观测下界；上界还需加该核 primary-shadow residual。",
            {
                "aic": float(summary["aic"]["phase_icache_misses"]["p95"]),
                "aiv": float(summary["aiv"]["phase_icache_misses"]["p95"]),
            },
        )
        phase_rows = "".join(
            (
                _phase_group_row("ALL", phase_groups["all"]),
                _phase_group_row("AIC", phase_groups["aic"]),
                _phase_group_row("AIV", phase_groups["aiv"]),
            )
        )
        phase_section = f"""
  <h2>局部 phase：{_escape(phase)}</h2>
  <section class="panel phase-panel">
    <div class="phase-warning"><strong>带边界扰动的 running read-clear 区间。</strong>下界是直接观测值，上界是下界加本核 primary-shadow residual；不同 phase ELF 的局部值不能相加，也不能从 none 相减得到无扰动净值。</div>
    <h3 class="phase-share-title">局部占同一 ELF 完整 Submit primary</h3>
    <p class="phase-share-note">每条灰色底轨代表该组完整 Submit 的 100%；深色为直接观测下界，浅色延伸到保守上界，空白部分属于局部窗口之外。request 与 miss 使用各自的完整窗口事件数作分母。</p>
    <div class="phase-share-grid">{phase_share_cards}</div>
    <details class="phase-table-details">
      <summary>展开 ALL / AIC / AIV 完整数字表</summary>
      <div class="phase-table-scroll">
        <table class="phase-table">
          <thead><tr><th>分组</th><th>calls</th><th>request sum 下界..上界</th><th>request/core 下界..上界</th><th>局部 request / Submit primary</th><th>miss sum 下界..上界</th><th>miss/core 下界..上界</th><th>局部 miss / Submit primary</th><th>shadow loss req/miss</th><th>observed miss/request</th></tr></thead>
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
    .card,.panel {{ background:white; border:1px solid var(--line); border-radius:12px; box-shadow:0 4px 18px #0f172a0d; }}
    .card {{ padding:16px; }}
    .card .label {{ color:var(--muted); font-size:.86rem; }}
    .card .value {{ margin-top:4px; font-size:1.55rem; font-weight:750; font-variant-numeric:tabular-nums; }}
    .panel {{ padding:18px; overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:9px 11px; text-align:right; border-bottom:1px solid #e2e8f0; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    thead th {{ background:#f8fafc; color:#475569; font-size:.82rem; text-transform:none; }}
    .role {{ display:inline-block; min-width:42px; text-align:center; border-radius:999px; color:white; padding:2px 7px; font-size:.78rem; font-weight:700; }}
    .role-aic {{ background:var(--aic); }} .role-aiv {{ background:var(--aiv); }}
    .insight {{ margin-top:12px; padding:12px 14px; border-left:4px solid var(--aiv); background:#fff7ed; }}
    .phase-warning {{ margin-bottom:14px; padding:12px 14px; border-left:4px solid #7c3aed; background:#f5f3ff; }}
    .phase-panel {{ overflow:hidden; }}
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
    .phase-share-upper-marker {{ position:absolute; top:-1px; bottom:-1px; width:2px; transform:translateX(-1px); background:#0f172a; opacity:.75; }}
    .phase-share-axis {{ display:grid; grid-template-columns:repeat(3,1fr); margin-top:2px; color:#64748b; font-size:.68rem; }}
    .phase-share-axis span:nth-child(2) {{ text-align:center; }} .phase-share-axis span:last-child {{ text-align:right; }}
    .phase-share-count {{ margin-top:2px; color:var(--muted); font-size:.76rem; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
    .phase-share-unavailable {{ padding:8px; border-radius:8px; background:#e2e8f0; color:var(--muted); }}
    .phase-table-details {{ margin-top:18px; }}
    .phase-table-details summary {{ padding-bottom:8px; }}
    .phase-table-scroll {{ width:100%; max-width:100%; overflow-x:auto; border:1px solid #dbe3ee; border-radius:10px; overscroll-behavior-x:contain; }}
    .phase-table {{ min-width:1240px; }}
    .phase-table th,.phase-table td {{ padding:8px 10px; }}
    .phase-plot-scroll {{ width:100%; max-width:100%; overflow-x:auto; overscroll-behavior-x:contain; }}
    .phase-disabled {{ margin-top:34px; }} .phase-disabled h2 {{ margin-top:0; }}
    .legend {{ display:flex; gap:18px; align-items:center; color:#475569; font-size:.88rem; margin-bottom:8px; }}
    .circle-key {{ width:10px; height:10px; border-radius:50%; background:var(--aic); }}
    .square-key {{ width:10px; height:10px; background:var(--aiv); }}
    .distribution {{ width:100%; min-width:760px; height:auto; display:block; }}
    .grid {{ stroke:{GRID_COLOR}; stroke-width:1; }} .tick {{ stroke:#64748b; }}
    .axis-label,.axis-title,.p95-label {{ fill:{TEXT_COLOR}; font-size:12px; }}
    .axis-title {{ font-size:13px; font-weight:650; }} .p95-label {{ font-weight:700; }}
    .p95-line {{ stroke-width:1.5; stroke-dasharray:7 5; opacity:.75; }}
    .aic-point {{ fill:var(--aic); }} .aiv-point {{ fill:var(--aiv); }}
    .point {{ opacity:.82; outline:none; }} .point:hover,.point:focus {{ opacity:1; stroke:#020617; stroke-width:2; }}
    .warning {{ border:2px dashed #94a3b8; background:#f8fafc; border-radius:12px; padding:18px; }}
    .warning h2 {{ margin-top:0; }}
    details summary {{ cursor:pointer; font-weight:700; padding:4px 0 14px; }}
    a {{ color:#1d4ed8; }} code {{ overflow-wrap:anywhere; }}
    footer {{ margin-top:32px; color:#475569; font-size:.86rem; }}
    :focus-visible {{ outline:3px solid #0ea5e9; outline-offset:2px; }}
    @media (max-width:640px) {{ main {{ width:min(100% - 16px,1180px); margin-top:12px; }} .panel {{ padding:13px; }} .phase-share-grid {{ grid-template-columns:1fr; }} }}
    @media print {{ body {{ background:white; }} main {{ width:100%; margin:0; }} .card,.panel {{ box-shadow:none; break-inside:avoid; }} details {{ display:block; }} .phase-share-upper,.phase-share-lower,.phase-share-upper-marker {{ print-color-adjust:exact; -webkit-print-color-adjust:exact; }} }}
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

  <section class="cards" aria-label="关键指标">
    <div class="card"><div class="label">完整 Submit</div><div class="value">{submit_us / 1000:.6f} ms</div></div>
    <div class="card"><div class="label">ALL PMU total/core mean</div><div class="value">{_format_per_core(all_metrics['total_per_core'])}</div><div class="muted">raw cycle；Σ={_format_count(all_metrics['total_sum'])}</div></div>
    <div class="card"><div class="label">ALL scalar busy/core mean</div><div class="value">{_format_per_core(all_metrics['scalar_per_core'])}</div><div class="muted">Σscalar/Σtotal={_format_rate(all_metrics['scalar_share'])}</div></div>
    <div class="card"><div class="label">完整 Submit primary I-cache request</div><div class="value">{total_requests:,}</div></div>
    <div class="card"><div class="label">完整 Submit primary I-cache miss</div><div class="value">{total_misses:,}</div></div>
    <div class="card"><div class="label">聚合 miss rate</div><div class="value">{_format_rate(float(all_metrics['miss_rate']))}</div></div>
    <div class="card"><div class="label">实测物理子核</div><div class="value">{workers}（32 AIC + 64 AIV）</div></div>
    <div class="card"><div class="label">Primary/Shadow</div><div class="value">exact {exact_records}/{workers}</div><div class="muted">bounded {bounded_records}/{workers}</div></div>
  </section>

  <h2>PMU total 与 scalar busy</h2>
  <section class="panel">
    <table>
      <thead><tr><th>角色</th><th>核数</th><th>total/core mean</th><th>total median</th><th>total p95</th><th>scalar/core mean</th><th>scalar median</th><th>scalar p95</th><th>Σscalar/Σtotal</th><th>非 Scalar-busy 残余/core</th></tr></thead>
      <tbody>
        <tr><td>ALL</td><td>{_format_count(all_metrics['cores'])}</td><td>{_format_per_core(all_metrics['total_per_core'])}</td><td>{_format_count(all_metrics['total_median'])}</td><td>{_format_count(all_metrics['total_p95'])}</td><td>{_format_per_core(all_metrics['scalar_per_core'])}</td><td>{_format_count(all_metrics['scalar_median'])}</td><td>{_format_count(all_metrics['scalar_p95'])}</td><td>{_format_rate(all_metrics['scalar_share'])}</td><td>{_format_per_core(all_metrics['non_scalar_busy_per_core'])}</td></tr>
        <tr><td><span class="role role-aic">AIC</span></td><td>{_format_count(aic['cores'])}</td><td>{_format_per_core(aic['total_per_core'])}</td><td>{_format_count(aic['total_median'])}</td><td>{_format_count(aic['total_p95'])}</td><td>{_format_per_core(aic['scalar_per_core'])}</td><td>{_format_count(aic['scalar_median'])}</td><td>{_format_count(aic['scalar_p95'])}</td><td>{_format_rate(aic['scalar_share'])}</td><td>{_format_per_core(aic['non_scalar_busy_per_core'])}</td></tr>
        <tr><td><span class="role role-aiv">AIV</span></td><td>{_format_count(aiv['cores'])}</td><td>{_format_per_core(aiv['total_per_core'])}</td><td>{_format_count(aiv['total_median'])}</td><td>{_format_count(aiv['total_p95'])}</td><td>{_format_per_core(aiv['scalar_per_core'])}</td><td>{_format_count(aiv['scalar_median'])}</td><td>{_format_count(aiv['scalar_p95'])}</td><td>{_format_rate(aiv['scalar_share'])}</td><td>{_format_per_core(aiv['non_scalar_busy_per_core'])}</td></tr>
      </tbody>
    </table>
    <div class="phase-warning"><strong>口径：</strong>total 是每个物理子核在 Submit gate 内的 PMU raw total cycle；求和代表 96 核 core-work。scalar busy 是 CNT2 scalar_instr_busy(0x001)。表中的“非 Scalar-busy 残余”严格等于 total−scalar busy，<strong>它不是空闲时间，也不是 I-cache stall</strong>，其中还混有同步等待、engine 等待及其他未归因周期。受控微基准已观察到依赖返回的 atomic 等待大部分进入 scalar busy，而 I-cache refill 的额外周期大部分只进入 total。当前 A5/DAV3510 正式事件表和 CANN 9.1 上板输出均不提供 scalar_wait_ib_time/scalar_wait_time，报告不会用其他产品的 selector 猜测这两项。</div>
    <h3>PMU total cycle</h3>{total_plot}
    <h3>Scalar busy cycle</h3>{scalar_plot}
  </section>

  <h2>AIC 与 AIV 的 I-cache 对比</h2>
  <section class="panel">
    <table>
      <thead><tr><th>角色</th><th>核数</th><th>request/core</th><th>request p95</th><th>miss/core mean</th><th>miss/core median</th><th>miss p95</th><th>miss max</th><th>Σmiss/Σrequest</th></tr></thead>
      <tbody>
        <tr><td><span class="role role-aic">AIC</span></td><td>{_format_count(aic['cores'])}</td><td>{_format_per_core(aic['requests_per_core'])}</td><td>{_format_count(aic['requests_p95'])}</td><td>{_format_per_core(aic['misses_per_core'])}</td><td>{_format_count(aic['misses_median'])}</td><td>{_format_count(aic['misses_p95'])}</td><td>{_format_count(aic['misses_max'])}</td><td>{_format_rate(aic['miss_rate'])}</td></tr>
        <tr><td><span class="role role-aiv">AIV</span></td><td>{_format_count(aiv['cores'])}</td><td>{_format_per_core(aiv['requests_per_core'])}</td><td>{_format_count(aiv['requests_p95'])}</td><td>{_format_per_core(aiv['misses_per_core'])}</td><td>{_format_count(aiv['misses_median'])}</td><td>{_format_count(aiv['misses_p95'])}</td><td>{_format_count(aiv['misses_max'])}</td><td>{_format_rate(aiv['miss_rate'])}</td></tr>
      </tbody>
    </table>
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
      <table>
        <thead><tr><th>worker</th><th>physical</th><th>role</th><th>block</th><th>lane</th><th>PMU total</th><th>scalar busy</th><th>scalar/total</th><th>requests</th><th>misses</th><th>per-core rate</th><th>primary=shadow</th><th>trusted</th></tr></thead>
        <tbody>{per_core_rows}</tbody>
      </table>
    </details>
    <p class="muted">聚合 miss rate 使用 Σmiss/Σrequest，不平均逐核百分比。p95 使用 nearest-rank：ceil(0.95 × N)。</p>
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
) -> Path:
    """在完整 HTML 构造成功后原子发布，失败时不留下半截报告。"""

    input_path = Path(input_path)
    output_path = default_output_path(input_path) if output_path is None else Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("HTML output path must differ from raw JSON input")
    document = render_report(input_path, output_path, miss_penalty_ns)
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
    arguments = parser.parse_args(argv)
    if not math.isfinite(arguments.icache_miss_ns) or arguments.icache_miss_ns <= 0:
        parser.error("--icache-miss-ns must be finite and positive")
    try:
        output = write_report(arguments.input, arguments.output, arguments.icache_miss_ns)
    except (OSError, ValueError) as error:
        print(f"PMU HTML report failed: {error}", file=sys.stderr)
        return 1
    print(f"[PMU-HTML] raw={arguments.input} report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
