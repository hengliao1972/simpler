# PTO2 Device Log Profiling Guide

## How to Find Device Logs

AICPU logs (via `LOG_INFO_V9`) are written by CANN's **dlog** subsystem and do **not** appear in the `python test_*.py` / pytest terminal output. They are written to CANN's device log directory:

```text
$HOME/ascend/log/debug/device-<device_id>/device-<pid>_<timestamp>.log
```

Each run produces a new log file (or appends to an existing one). Find the most recent file by modification time:

```bash
ls -lt $HOME/ascend/log/debug/device-<device_id>/ | head -5
```

## Log Structure Overview

A direct distributed AICore run produces one AICPU timing line after the
AICore replay window completes. Older central-scheduler builds also produced
per-scheduler summaries; those are not emitted by the slim AICPU executor.

| Block | Emitted by | Function | Content |
| ----- | ---------- | -------- | ------- |
| **AICore Replay Timing** | Thread 3 (AICPU setup thread) | `AicpuExecutor::run` | Wall time from releasing AICores to all workers reporting done |

All timing values are in microseconds (us), converted from AICPU cycle counters.

---

## Block 1: AICore Replay Timing

fdwic links the per-example orchestration source into `libaicore_kernel.so`.
Thread 3 initializes the shared distributed-engine state, releases the AICore
workers, waits for every worker to increment `Runtime::dist.done_count`, and
then prints the replay-window timing. It does not `dlopen` a standalone
orchestration `.so`.

### Example

```text
Thread 3: orch_start=1783051497915574420 orch_end=1783051498123092400 orch_cost=207517.980us
```

### Field Reference

| Field | Source | Description |
| ----- | ------ | ----------- |
| **orch_start** | `AicpuExecutor::run` | Timestamp just before AICPU publishes `Runtime::dist.go = 1` |
| **orch_end** | `AicpuExecutor::run` | Timestamp after `Runtime::dist.done_count == num_workers` |
| **orch_cost** | `AicpuExecutor::run` | Full distributed replay window, including AICore orchestration and kernel execution |

### Interpreting the Numbers

- `orch_cost` is end-to-end for the AICore replay window, not just host-side
  graph construction.
- With `--use-example-exec-time`, kernels with configured
  `example_exec_time_ns` busy-wait instead of executing their real body; golden
  comparison is skipped in that mode.
- Without `--use-example-exec-time`, real kernels execute and scene tests run
  the normal golden comparison unless `--skip-golden` is passed.

---

## Historical Block: PTO2 Scheduler Summary

The following section applies only to the legacy central-scheduler path. The
direct distributed AICore path does not link `SchedulerContext`; AICPU performs
only worker handshake, register setup, release, completion polling, and
shutdown.

Each of the 3 scheduler threads (Thread 0, 1, 2) prints its own summary after completing all tasks. The output has two sub-sections: **summary** and **phase breakdown**.

### Example (Thread 0, from a different run: batch=1, 1044 tasks)

```text
Thread 0: completed=352 tasks in 3477.420us (147 loops, 2.4 tasks/loop)
Thread 0: --- Phase Breakdown ---
Thread 0:   complete:    1485.020us (42.7%)
Thread 0:   scan:        14.400us (0.4%)
Thread 0:   dispatch:    1973.060us (56.7%)
Thread 0:   idle:        4.940us (0.1%)
```

### Summary Line

```text
Thread N: completed=X tasks in Yus (Z loops, W tasks/loop)
```

| Field | Description |
| ----- | ----------- |
| **completed** | Number of tasks this thread processed to completion |
| **Y us** | Total scheduler loop time (sum of all phase cycles) |
| **Z loops** | Number of scheduler loop iterations |
| **W tasks/loop** | Average tasks completed per loop iteration; higher = better throughput |

### Phase Breakdown

The scheduler loop runs four phases each iteration. Each phase's time is accumulated across all loop iterations.

| Phase | What it does | Inline stats |
| ----- | ------------ | ------------ |
| **complete** | Polls handshake on each managed core; when a core completes, calls `on_subtask_complete(task_id, subslot)` to increment the completion counter; when `completed_subtasks == total_required_subtasks`, triggers `on_task_complete` which traverses fanout list (notify consumers) and fanin list (release producers) | `fanout`: edges/max_degree/avg for consumer notification; `fanin`: edges/max_degree/avg for producer release |
| **scan** | Updates the perf profiling header with latest scheduler state | — |
| **dispatch** | For each idle core, pops a task from the shape-based ready queue via `get_ready_task(shape)`, builds the dispatch payload, and writes the task to the core's handshake register | `pop`: `hit` = successful pops (task dispatched), `miss` = empty queue pops, `hit_rate` = hit/(hit+miss) |
| **idle** | Scheduler loop iteration where no progress was made (no completions, no dispatches) | — |

**Interpreting phase percentages:**

- **dispatch** is typically the largest (~55-60%) because it includes ready-queue pops (with spinlock), payload construction, and cache flush (`dc cvac` + `dsb sy`).
- **complete** is the second largest (~40-45%) because it traverses both fanout (CAS-based fanin decrement, conditional ready-queue push) and fanin (release_producer, check_consumed, ring pointer advancement).
- **scan** is small (<1%) — only updates the perf header.
- **idle** is negligible when tasks are flowing; high idle% indicates the scheduler is starved.

**Interpreting pop hit_rate:**

- **High hit_rate (>50%)**: Ready queue is well-supplied; dispatch is efficient.
- **Low hit_rate (<10%)**: Ready queue is mostly empty when cores become idle. The bottleneck is upstream (orchestrator submission speed or fanout resolution latency), not dispatch itself.

### Per-Task Averages

Divide each thread's phase times by its `completed` count to get per-task scheduling cost:

| Metric | Formula | Typical value |
| ------ | ------- | ------------- |
| Scheduling overhead per task | total_time / completed | ~5-10 us/task |
| Dispatch per task | dispatch_time / completed | ~3-6 us/task |
| Complete per task | complete_time / completed | ~2-4 us/task |

---

## Cross-Referencing with Host Profiling

When `--enable-l2-swimlane` is used, the host terminal prints a **Task Statistics by Function** table with `Total_Exec` (total AICore kernel execution time). Combined with device log data:

| Metric | Source | Description |
| ------ | ------ | ----------- |
| Avg kernel exec time | `Total_Exec / total_tasks` (host) | Time AICore spends executing each kernel |
| Avg scheduling overhead | `sum(thread_total) / total_tasks` (device log) | Time AICPU spends scheduling each task |
| Sched/Exec ratio | scheduling / execution | Scheduling overhead relative to kernel execution |

A high sched/exec ratio (e.g., >3x) indicates that scheduling overhead dominates, and optimizations should target the scheduler's dispatch hot path (cache flush, payload construction) or upstream task flow.

---

## Quick Reference: Extracting Profiling Data

```bash
# Find the latest device log for device 2
ls -t $HOME/ascend/log/debug/device-2/device-*.log | head -1

# Extract orchestrator profiling (Thread 3)
grep "Thread 3:" <logfile>

# Extract scheduler profiling (Threads 0/1/2)
grep -E "Thread [012]:" <logfile>
```
