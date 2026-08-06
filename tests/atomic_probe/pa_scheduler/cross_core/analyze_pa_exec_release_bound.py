#!/usr/bin/env python3
"""从 PA full-swimlane 推导 Build 发布与 Execute 调度的性能上界。

该工具只读取已经生成的 merged_swimlane.json，不修改设备代码，也不新增
泳道记录。当前 standalone PA 的 task-id 合同固定为每组
Alloc/QK/SF/PV/UP，因此可以从 task id 精确重建 QK->SF->PV 以及
QK/SF/PV->UP 的依赖。输出用于回答两个不同问题：

1. 保留真实 BUILT 发布时间，假设真正 ready 后可以零调度开销执行，最多还能
   缩短多少；
2. 再把 BUILT 发布时间理想化为 0，只保留真实 kernel/completion 工作量时，
   当前负载离计算下界还有多远。

第二项是理想化的可实现列表调度结果，不应写成硬件频率意义上的严格下界；
严格的算术下界会另外按各 engine 的总工作量/核数给出。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


TASK_SUFFIX = re.compile(r"#(\d+)$")
KERNEL_NAME = re.compile(r"^task\.execute\.(QK|SF|PV|UP)#(\d+)$")


@dataclass(frozen=True)
class TaskTiming:
    task_id: int
    kind: str
    built_end_us: float
    claim_end_us: float
    kernel_begin_us: float
    kernel_duration_us: float
    completion_flag_end_us: float
    done_end_us: float

    @property
    def service_us(self) -> float:
        """kernel 加 vend/flag/DONE 发布的实际占用时间。"""
        return max(0.0, self.done_end_us - self.kernel_begin_us)

    @property
    def dependency_publish_us(self) -> float:
        """从 kernel 开始到 completion flag 对后继可见的时间。"""
        return max(
            0.0, self.completion_flag_end_us - self.kernel_begin_us
        )


def task_kind(task_id: int) -> str:
    remainder = task_id % 5
    kinds = {1: "QK", 2: "SF", 3: "PV", 4: "UP"}
    if remainder not in kinds:
        raise ValueError(f"task {task_id} is not an executable PA task")
    return kinds[remainder]


def task_dependencies(task_id: int) -> Tuple[int, ...]:
    remainder = task_id % 5
    if remainder == 1:
        return ()
    if remainder in (2, 3):
        return (task_id - 1,)
    if remainder == 4:
        return (task_id - 3, task_id - 2, task_id - 1)
    raise ValueError(f"task {task_id} is not an executable PA task")


def task_role(task_id: int) -> str:
    return "AIC" if task_id % 5 in (1, 3) else "AIV"


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return (
        ordered[low] * (high - position)
        + ordered[high] * (position - low)
    )


def distribution(values: Sequence[float]) -> Mapping[str, float]:
    return {
        "count": len(values),
        "sum_us": sum(values),
        "mean_us": statistics.mean(values),
        "median_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
        "max_us": max(values),
    }


def event_task_id(name: str) -> int | None:
    match = TASK_SUFFIX.search(name)
    return int(match.group(1)) if match else None


def load_timings(path: Path) -> Dict[int, TaskTiming]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    built: Dict[int, float] = {}
    claim: Dict[int, float] = {}
    kernel_begin: Dict[int, float] = {}
    kernel_duration: Dict[int, float] = {}
    kernel_kind: Dict[int, str] = {}
    completion_flag: Dict[int, float] = {}
    done: Dict[int, float] = {}

    for event in document.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        task_id = event_task_id(name)
        if task_id is None:
            continue
        begin = float(event["ts"])
        duration = float(event.get("dur", 0.0))
        end = begin + duration
        if "shared_exec_built_publish.compare_exchange" in name:
            built[task_id] = end
        elif "shared_exec_claim.compare_exchange" in name:
            claim[task_id] = end
        elif "shared_exec_completion_flag_publish.exchange" in name:
            completion_flag[task_id] = end
        elif "shared_exec_done_publish.compare_exchange" in name:
            done[task_id] = end
        else:
            match = KERNEL_NAME.match(name)
            if match:
                kernel_kind[task_id] = match.group(1)
                kernel_begin[task_id] = begin
                kernel_duration[task_id] = duration

    identities = set(built)
    sources = {
        "claim": set(claim),
        "kernel": set(kernel_begin),
        "completion_flag": set(completion_flag),
        "done": set(done),
    }
    for source_name, task_ids in sources.items():
        if task_ids != identities:
            missing = sorted(identities - task_ids)[:8]
            extra = sorted(task_ids - identities)[:8]
            raise ValueError(
                f"{source_name} task set differs from BUILT: "
                f"missing={missing} extra={extra}"
            )

    timings: Dict[int, TaskTiming] = {}
    for task_id in sorted(identities):
        expected_kind = task_kind(task_id)
        if kernel_kind[task_id] != expected_kind:
            raise ValueError(
                f"task {task_id} kind mismatch: "
                f"trace={kernel_kind[task_id]} expected={expected_kind}"
            )
        dependencies = task_dependencies(task_id)
        if any(producer not in identities for producer in dependencies):
            raise ValueError(
                f"task {task_id} has a missing producer {dependencies}"
            )
        timings[task_id] = TaskTiming(
            task_id=task_id,
            kind=expected_kind,
            built_end_us=built[task_id],
            claim_end_us=claim[task_id],
            kernel_begin_us=kernel_begin[task_id],
            kernel_duration_us=kernel_duration[task_id],
            completion_flag_end_us=completion_flag[task_id],
            done_end_us=done[task_id],
        )
    return timings


def simulate_work_conserving(
    timings: Mapping[int, TaskTiming],
    aic_workers: int,
    aiv_workers: int,
    preserve_build_release: bool,
) -> Mapping[str, object]:
    """在依赖 ready 时才占 engine，构造一个乐观的 work-conserving 调度。"""
    engine_available: MutableMapping[str, List[float]] = {
        "AIC": [0.0] * aic_workers,
        "AIV": [0.0] * aiv_workers,
    }
    finish: Dict[int, float] = {}
    dependency_publish: Dict[int, float] = {}
    starts: Dict[int, float] = {}
    ready_times: Dict[int, float] = {}
    pending = set(timings)

    while pending:
        best: Tuple[float, float, int, str, int] | None = None
        for task_id in pending:
            dependencies = task_dependencies(task_id)
            if any(
                producer not in dependency_publish
                for producer in dependencies
            ):
                continue
            role = task_role(task_id)
            engine_index = min(
                range(len(engine_available[role])),
                key=engine_available[role].__getitem__,
            )
            release = (
                timings[task_id].built_end_us
                if preserve_build_release
                else 0.0
            )
            ready = max(
                [release]
                + [
                    dependency_publish[producer]
                    for producer in dependencies
                ]
            )
            start = max(ready, engine_available[role][engine_index])
            candidate = (start, ready, task_id, role, engine_index)
            if best is None or candidate < best:
                best = candidate

        if best is None:
            raise ValueError("dependency graph contains a cycle or missing task")
        start, ready, task_id, role, engine_index = best
        end = start + timings[task_id].service_us
        starts[task_id] = start
        ready_times[task_id] = ready
        finish[task_id] = end
        dependency_publish[task_id] = (
            start + timings[task_id].dependency_publish_us
        )
        engine_available[role][engine_index] = end
        pending.remove(task_id)

    return {
        "first_kernel_begin_us": min(starts.values()),
        "last_completion_end_us": max(finish.values()),
        "engine_queue_delay": distribution(
            [starts[task_id] - ready_times[task_id] for task_id in starts]
        ),
        "aic_last_completion_end_us": max(
            finish[task_id]
            for task_id in finish
            if task_role(task_id) == "AIC"
        ),
        "aiv_last_completion_end_us": max(
            finish[task_id]
            for task_id in finish
            if task_role(task_id) == "AIV"
        ),
    }


def analyze(
    path: Path, aic_workers: int, aiv_workers: int
) -> Mapping[str, object]:
    timings = load_timings(path)
    completion_flag_times = {
        task_id: timing.completion_flag_end_us
        for task_id, timing in timings.items()
    }
    by_kind: Dict[str, Mapping[str, Mapping[str, float]]] = {}
    for kind in ("QK", "SF", "PV", "UP"):
        selected = [timing for timing in timings.values() if timing.kind == kind]
        built_to_claim: List[float] = []
        dependency_wait_after_claim: List[float] = []
        runnable_to_kernel: List[float] = []
        for timing in selected:
            dependency_ready = max(
                [timing.built_end_us]
                + [
                    completion_flag_times[producer]
                    for producer in task_dependencies(timing.task_id)
                ]
            )
            built_to_claim.append(
                timing.claim_end_us - timing.built_end_us
            )
            dependency_wait_after_claim.append(
                max(0.0, dependency_ready - timing.claim_end_us)
            )
            runnable_to_kernel.append(
                timing.kernel_begin_us
                - max(timing.claim_end_us, dependency_ready)
            )
        by_kind[kind] = {
            "built_to_claim": distribution(built_to_claim),
            "dependency_wait_after_claim": distribution(
                dependency_wait_after_claim
            ),
            "claimed_and_ready_to_kernel": distribution(runnable_to_kernel),
        }

    role_work = {
        role: sum(
            timing.service_us
            for timing in timings.values()
            if task_role(timing.task_id) == role
        )
        for role in ("AIC", "AIV")
    }
    return {
        "trace": str(path),
        "task_count": len(timings),
        "actual": {
            "first_kernel_begin_us": min(
                timing.kernel_begin_us for timing in timings.values()
            ),
            "last_built_end_us": max(
                timing.built_end_us for timing in timings.values()
            ),
            "last_completion_end_us": max(
                timing.done_end_us for timing in timings.values()
            ),
        },
        "release_preserving_ideal": simulate_work_conserving(
            timings, aic_workers, aiv_workers, True
        ),
        "zero_build_release_ideal": simulate_work_conserving(
            timings, aic_workers, aiv_workers, False
        ),
        "arithmetic_work_lower_bound": {
            "aic_total_service_us": role_work["AIC"],
            "aic_service_per_core_us": role_work["AIC"] / aic_workers,
            "aiv_total_service_us": role_work["AIV"],
            "aiv_service_per_core_us": role_work["AIV"] / aiv_workers,
        },
        "latency_by_kind": by_kind,
    }


def print_distribution(label: str, values: Mapping[str, float]) -> None:
    print(
        f"  {label}: n={int(values['count'])} "
        f"mean={values['mean_us']:.3f} us "
        f"median={values['median_us']:.3f} us "
        f"p95={values['p95_us']:.3f} us "
        f"max={values['max_us']:.3f} us"
    )


def print_human(result: Mapping[str, object]) -> None:
    actual = result["actual"]
    release_ideal = result["release_preserving_ideal"]
    zero_ideal = result["zero_build_release_ideal"]
    lower = result["arithmetic_work_lower_bound"]
    print(f"trace: {result['trace']}")
    print(f"executable tasks: {result['task_count']}")
    print(
        "actual: "
        f"first_kernel={actual['first_kernel_begin_us']:.3f} us "
        f"last_built={actual['last_built_end_us']:.3f} us "
        f"last_completion={actual['last_completion_end_us']:.3f} us"
    )
    print(
        "release-preserving ideal: "
        f"first_kernel={release_ideal['first_kernel_begin_us']:.3f} us "
        f"last_completion={release_ideal['last_completion_end_us']:.3f} us"
    )
    print_distribution(
        "ideal engine queue delay", release_ideal["engine_queue_delay"]
    )
    print(
        "zero-Build-release ideal: "
        f"last_completion={zero_ideal['last_completion_end_us']:.3f} us"
    )
    print(
        "arithmetic service lower bound: "
        f"AIC={lower['aic_service_per_core_us']:.3f} us/core "
        f"AIV={lower['aiv_service_per_core_us']:.3f} us/core"
    )
    for kind, metrics in result["latency_by_kind"].items():
        print(kind)
        print_distribution("BUILT -> CLAIMED", metrics["built_to_claim"])
        print_distribution(
            "post-claim dependency wait",
            metrics["dependency_wait_after_claim"],
        )
        print_distribution(
            "claimed+ready -> kernel",
            metrics["claimed_and_ready_to_kernel"],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("swimlane", type=Path)
    parser.add_argument("--aic-workers", type=int, default=32)
    parser.add_argument("--aiv-workers", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.aic_workers <= 0 or args.aiv_workers <= 0:
        parser.error("worker counts must be positive")
    result = analyze(args.swimlane, args.aic_workers, args.aiv_workers)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
