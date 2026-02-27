#!/usr/bin/env python3
"""Scheduler overhead analysis for PTO2.

Analyzes scheduling overhead from two sources:
  1. Per-task perf profiling data (perf_swimlane_*.json)
  2. AICPU scheduler loop breakdown (device log)

Usage:
    python sched_overhead_analysis.py                          # auto-select latest files
    python sched_overhead_analysis.py --perf-json <path>       # specify perf data
    python sched_overhead_analysis.py --device-log <path>      # specify device log
"""
import argparse
import json
import re
import sys
from pathlib import Path


def auto_select_perf_json():
    """Find the latest perf_swimlane_*.json in outputs/ directory."""
    outputs_dir = Path(__file__).parent.parent / 'outputs'
    files = sorted(outputs_dir.glob('perf_swimlane_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(f"Error: No perf_swimlane_*.json files found in {outputs_dir}", file=sys.stderr)
        sys.exit(1)
    return files[0]


def auto_select_device_log():
    """Find the latest .log in ~/ascend/log/debug/device-0/."""
    log_dir = Path.home() / 'ascend' / 'log' / 'debug' / 'device-0'
    if not log_dir.exists():
        print(f"Error: Device log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)
    files = sorted(log_dir.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(f"Error: No .log files found in {log_dir}", file=sys.stderr)
        sys.exit(1)
    return files[0]


def parse_scheduler_threads(log_path):
    """Parse device log for PTO2 scheduler stats per thread.

    Expected log format (per thread):
        Thread N: completed=X tasks in Yus (Z loops, W tasks/loop)
        Thread N: --- Phase Breakdown (execution order) ---
        Thread N:   scan:            Xus (Y%)
        Thread N:   early_ready:     Xus (Y%)  (deps already met at submit time)
        Thread N:   complete:        Xus (Y%)  [fanout: edges=A, max_degree=B, avg=C]
        Thread N:   dispatch:        Xus (Y%)  [steal: own=A, steal=B, pct=C%]
        Thread N: --- Lock Contention (ready_q) ---
        Thread N:   total:         wait= Xus hold= Yus
        Thread N:   scan:          wait= Xus hold= Yus
        Thread N:   early_ready:   wait= Xus hold= Yus
        Thread N:   complete:      wait= Xus hold= Yus
        Thread N:   dispatch:      wait= Xus hold= Yus
        Thread N:     hit:         wait= Xus hold= Yus (dequeued task)
        Thread N:     miss:        wait= Xus hold= Yus (empty queue)
    """
    threads = {}
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            # Summary: Thread N: completed=X tasks in Yus (Z loops, W tasks/loop)
            m = re.search(r'Thread (\d+): completed=(\d+) tasks in ([\d.]+)us \((\d+) loops, ([\d.]+) tasks/loop\)', line)
            if m:
                tid = int(m.group(1))
                threads[tid] = {
                    'completed': int(m.group(2)),
                    'total_us': float(m.group(3)),
                    'loops': int(m.group(4)),
                    'tasks_per_loop': float(m.group(5)),
                }

            # Phase: scan (distinguished from lock scan by absence of "wait=")
            m = re.search(r'Thread (\d+):\s+scan:\s+([\d.]+)us \(\s*([\d.]+)%\)', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['scan_us'] = float(m.group(2))
                    threads[tid]['scan_pct'] = float(m.group(3))

            # Phase: early_ready
            m = re.search(r'Thread (\d+):\s+early_ready:\s+([\d.]+)us \(\s*([\d.]+)%\)', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['early_ready_us'] = float(m.group(2))
                    threads[tid]['early_ready_pct'] = float(m.group(3))

            # Phase: complete [fanout: edges=X, max_degree=Y, avg=Z]
            m = re.search(r'Thread (\d+):\s+complete:\s+([\d.]+)us \(\s*([\d.]+)%\)\s+\[fanout: edges=(\d+), max_degree=(\d+), avg=([\d.]+)\]', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['complete_us'] = float(m.group(2))
                    threads[tid]['complete_pct'] = float(m.group(3))
                    threads[tid]['fanout_edges'] = int(m.group(4))
                    threads[tid]['fanout_max_degree'] = int(m.group(5))
                    threads[tid]['fanout_avg'] = float(m.group(6))

            # Phase: dispatch [steal: own=X, steal=Y, pct=Z%]
            m = re.search(r'Thread (\d+):\s+dispatch:\s+([\d.]+)us \(\s*([\d.]+)%\)\s+\[steal: own=(\d+), steal=(\d+), pct=([\d.]+)%\]', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['dispatch_us'] = float(m.group(2))
                    threads[tid]['dispatch_pct'] = float(m.group(3))
                    threads[tid]['steal_own'] = int(m.group(4))
                    threads[tid]['steal_steal'] = int(m.group(5))
                    threads[tid]['steal_pct'] = float(m.group(6))

            # Lock: total
            m = re.search(r'Thread (\d+):\s+total:\s+wait=\s*(\d+)us hold=\s*(\d+)us', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['lock_wait_us'] = int(m.group(2))
                    threads[tid]['lock_hold_us'] = int(m.group(3))

            # Lock: scan
            m = re.search(r'Thread (\d+):\s+scan:\s+wait=\s*(\d+)us hold=\s*(\d+)us', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['lock_scan_wait'] = int(m.group(2))
                    threads[tid]['lock_scan_hold'] = int(m.group(3))

            # Lock: early_ready
            m = re.search(r'Thread (\d+):\s+early_ready:\s+wait=\s*(\d+)us hold=\s*(\d+)us', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['lock_early_ready_wait'] = int(m.group(2))
                    threads[tid]['lock_early_ready_hold'] = int(m.group(3))

            # Lock: complete
            m = re.search(r'Thread (\d+):\s+complete:\s+wait=\s*(\d+)us hold=\s*(\d+)us', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['lock_complete_wait'] = int(m.group(2))
                    threads[tid]['lock_complete_hold'] = int(m.group(3))

            # Lock: dispatch
            m = re.search(r'Thread (\d+):\s+dispatch:\s+wait=\s*(\d+)us hold=\s*(\d+)us', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['lock_dispatch_wait'] = int(m.group(2))
                    threads[tid]['lock_dispatch_hold'] = int(m.group(3))

            # Lock: dispatch hit
            m = re.search(r'Thread (\d+):\s+hit:\s+wait=\s*(\d+)us hold=\s*(\d+)us', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['lock_dispatch_hit_wait'] = int(m.group(2))
                    threads[tid]['lock_dispatch_hit_hold'] = int(m.group(3))

            # Lock: dispatch miss
            m = re.search(r'Thread (\d+):\s+miss:\s+wait=\s*(\d+)us hold=\s*(\d+)us', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['lock_dispatch_miss_wait'] = int(m.group(2))
                    threads[tid]['lock_dispatch_miss_hold'] = int(m.group(3))

    return threads


def main():
    parser = argparse.ArgumentParser(
        description='Scheduler overhead analysis for PTO2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # auto-select latest files
  %(prog)s --perf-json outputs/perf_swimlane_*.json
  %(prog)s --device-log ~/ascend/log/debug/device-0/device-*.log
        """
    )
    parser.add_argument('--perf-json', help='Path to perf_swimlane_*.json file. If not specified, uses the latest in outputs/')
    parser.add_argument('--device-log', help='Path to device log file. If not specified, uses the latest in ~/ascend/log/debug/device-0/')
    args = parser.parse_args()

    # Resolve input paths
    perf_path = Path(args.perf_json) if args.perf_json else auto_select_perf_json()
    log_path = Path(args.device_log) if args.device_log else auto_select_device_log()

    if not perf_path.exists():
        print(f"Error: Perf JSON not found: {perf_path}", file=sys.stderr)
        return 1
    if not log_path.exists():
        print(f"Error: Device log not found: {log_path}", file=sys.stderr)
        return 1

    print(f"Perf data:  {perf_path}")
    print(f"Device log: {log_path}")

    # === Part 1: Per-task time breakdown from perf data ===
    with open(perf_path) as f:
        data = json.load(f)
    tasks = data['tasks']
    n_total = len(tasks)

    all_exec = sum(t['duration_us'] for t in tasks)
    all_head = sum(t['start_time_us'] - t['dispatch_time_us'] for t in tasks)
    all_tail = sum(t['finish_time_us'] - t['end_time_us'] for t in tasks)
    min_disp = min(t['dispatch_time_us'] for t in tasks)
    max_fin = max(t['finish_time_us'] for t in tasks)
    wall = max_fin - min_disp

    all_latency = all_exec + all_head + all_tail

    print()
    print('=' * 90)
    print('Part 1: Per-task time breakdown (from perf profiling data)')
    print('=' * 90)
    print(f'Total tasks: {n_total}')
    print(f'Wall-clock:  {wall:.1f} us')
    print()
    fmt = "  {:<35} {:>12} {:>14} {:>13}"
    print(fmt.format('Component', 'Total (us)', 'Avg/task (us)', '% of Latency'))
    print('  ' + '-' * 78)
    print(fmt.format('Kernel Exec (end-start)', f'{all_exec:.1f}', f'{all_exec/n_total:.2f}', f'{all_exec/all_latency*100:.1f}%'))
    print(fmt.format('Head OH (start-dispatch)', f'{all_head:.1f}', f'{all_head/n_total:.2f}', f'{all_head/all_latency*100:.1f}%'))
    print(fmt.format('Tail OH (finish-end)', f'{all_tail:.1f}', f'{all_tail/n_total:.2f}', f'{all_tail/all_latency*100:.1f}%'))
    print()

    # === Part 2: AICPU scheduler loop breakdown from device log ===
    threads = parse_scheduler_threads(log_path)
    n_threads = len(threads)

    print('=' * 90)
    print('Part 2: AICPU scheduler loop breakdown (from device log)')
    print(f'  {n_threads} scheduler threads')
    print('=' * 90)
    print()

    fmt2 = "  {:<10} {:>7} {:>10} {:>12} {:>11}"
    print(fmt2.format('Thread', 'Loops', 'Completed', 'Tasks/loop', 'Total (us)'))
    print('  ' + '-' * 54)
    for tid in sorted(threads.keys()):
        t = threads[tid]
        print(fmt2.format('T'+str(tid), t['loops'], t['completed'], f"{t['tasks_per_loop']:.1f}", f"{t['total_us']:.1f}"))
    total_us = sum(t['total_us'] for t in threads.values())
    total_completed = sum(t['completed'] for t in threads.values())
    total_loops = sum(t['loops'] for t in threads.values())
    avg_tpl = total_completed / total_loops if total_loops > 0 else 0
    print(fmt2.format('SUM', total_loops, total_completed, f'{avg_tpl:.1f}', f'{total_us:.1f}'))
    print()

    # Phase breakdown
    phases = ['scan', 'early_ready', 'complete', 'dispatch']
    phase_labels = {
        'scan':        'Scan (discover new root tasks)',
        'early_ready': 'Early ready (deps met at submit time)',
        'complete':    'Complete (poll handshake, resolve fanout)',
        'dispatch':    'Dispatch (pop queue, build payload, flush)',
    }

    fmt3 = "  {:<50} {:>11} {:>10} {:>14}"
    print(fmt3.format('Phase', 'Total (us)', '% of total', 'Avg/task (us)'))
    print('  ' + '-' * 89)
    phase_totals = {}
    for p in phases:
        key = p + '_us'
        tot = sum(t.get(key, 0) for t in threads.values())
        phase_totals[p] = tot
        pct = tot / total_us * 100 if total_us > 0 else 0
        avg = tot / total_completed if total_completed > 0 else 0
        print(fmt3.format(phase_labels[p], f'{tot:.1f}', f'{pct:.1f}%', f'{avg:.2f}'))
    print()

    # Fanout stats (from complete phase)
    fanout_edges = sum(t.get('fanout_edges', 0) for t in threads.values())
    fanout_max = max((t.get('fanout_max_degree', 0) for t in threads.values()), default=0)
    fanout_avg_weighted = sum(t.get('fanout_avg', 0) * t.get('fanout_edges', 0) for t in threads.values())
    fanout_avg = fanout_avg_weighted / fanout_edges if fanout_edges > 0 else 0
    print(f'  Fanout: total edges={fanout_edges}, max_degree={fanout_max}, avg_degree={fanout_avg:.1f}')
    print()

    # Work stealing stats (from dispatch phase)
    steal_own = sum(t.get('steal_own', 0) for t in threads.values())
    steal_steal = sum(t.get('steal_steal', 0) for t in threads.values())
    steal_total = steal_own + steal_steal
    steal_pct = steal_steal / steal_total * 100 if steal_total > 0 else 0
    print(f'  Work stealing: own={steal_own}, stolen={steal_steal} ({steal_pct:.1f}% steal rate)')
    print()

    # Lock contention breakdown
    fmt4 = "  {:<50} {:>11} {:>10}"
    print(fmt4.format('Lock contention (ready_q)', 'Total (us)', '% of total'))
    print('  ' + '-' * 75)
    lock_wait = sum(t.get('lock_wait_us', 0) for t in threads.values())
    lock_hold = sum(t.get('lock_hold_us', 0) for t in threads.values())
    print(fmt4.format('  wait (spinning for lock)', str(lock_wait), f'{lock_wait/total_us*100:.1f}%' if total_us > 0 else '0.0%'))
    print(fmt4.format('  hold (inside critical section)', str(lock_hold), f'{lock_hold/total_us*100:.1f}%' if total_us > 0 else '0.0%'))
    print()

    # Lock wait breakdown by phase
    print('  Lock wait by phase:')
    for p in phases:
        w = sum(t.get(f'lock_{p}_wait', 0) for t in threads.values())
        h = sum(t.get(f'lock_{p}_hold', 0) for t in threads.values())
        print(f'    {p:<14}  wait={w:>6} us  hold={h:>6} us')
    # Dispatch hit/miss sub-breakdown
    hit_w = sum(t.get('lock_dispatch_hit_wait', 0) for t in threads.values())
    hit_h = sum(t.get('lock_dispatch_hit_hold', 0) for t in threads.values())
    miss_w = sum(t.get('lock_dispatch_miss_wait', 0) for t in threads.values())
    miss_h = sum(t.get('lock_dispatch_miss_hold', 0) for t in threads.values())
    print(f'      {"hit":<12}  wait={hit_w:>6} us  hold={hit_h:>6} us  (dequeued task)')
    print(f'      {"miss":<12}  wait={miss_w:>6} us  hold={miss_h:>6} us  (empty queue)')

    print()
    print('=' * 90)
    print('Part 3: Tail OH distribution & cause analysis')
    print('=' * 90)
    print()

    tails = [t['finish_time_us'] - t['end_time_us'] for t in tasks]
    tails.sort()
    n = len(tails)
    print(f'  Tail OH distribution (N={n}):')
    for pct_val in [10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(n * pct_val / 100), n - 1)
        print(f'    P{pct_val:<4}  {tails[idx]:>7.1f} us')
    print(f'    Max:   {tails[-1]:>7.1f} us')
    print(f'    Mean:  {sum(tails)/n:>7.1f} us')
    print()

    # Scheduler loop time
    avg_loop_us = total_us / total_loops if total_loops > 0 else 0
    avg_tail_oh = sum(tails) / n
    loop_ratio = avg_tail_oh / avg_loop_us if avg_loop_us > 0 else 0
    print(f'  Avg scheduler loop iteration: {avg_loop_us:.1f} us (\u2248 avg polling interval per loop)')
    if n_threads > 0:
        print(f'  With {n_threads} threads sharing {total_loops} loops over {total_us/n_threads:.0f} us wall each:')
    print()

    print('  Scheduler CPU time breakdown (per completed task):')

    # Build phase data with sub-items for sorting
    phase_details = {
        'dispatch': {
            'label': 'Dispatch phase (build payload + cache flush)',
            'total': phase_totals.get('dispatch', 0),
            'sub_items': [
                ('Lock wait (ready_q pop)', sum(t.get('lock_dispatch_wait', 0) for t in threads.values())),
                ('Lock hold + build + dc cvac/civac + dsb sy', phase_totals.get('dispatch', 0) - sum(t.get('lock_dispatch_wait', 0) for t in threads.values())),
            ]
        },
        'complete': {
            'label': 'Complete phase (poll + fanout resolve)',
            'total': phase_totals.get('complete', 0),
            'sub_items': [
                ('Lock wait (ready_q push)', sum(t.get('lock_complete_wait', 0) for t in threads.values())),
                ('Fanout traversal + atomic ops', phase_totals.get('complete', 0) - sum(t.get('lock_complete_wait', 0) for t in threads.values())),
            ]
        },
        'scan': {
            'label': 'Scan phase (new task discovery)',
            'total': phase_totals.get('scan', 0),
            'sub_items': []
        },
        'early_ready': {
            'label': 'Early ready (deps met at submit time)',
            'total': phase_totals.get('early_ready', 0),
            'sub_items': []
        },
    }

    # Sort by total descending
    for p, detail in sorted(phase_details.items(), key=lambda x: x[1]['total'], reverse=True):
        per_task = detail['total'] / total_completed if total_completed > 0 else 0
        pct = detail['total'] / total_us * 100 if total_us > 0 else 0
        print(f'    - {detail["label"]:<50} {per_task:.2f} us/task  ({pct:.1f}% of scheduler CPU)')
        for sub_label, sub_total in detail['sub_items']:
            sub_per_task = sub_total / total_completed if total_completed > 0 else 0
            print(f'        {sub_label:<48} {sub_per_task:.2f} us/task')

    print()
    print(f'  Avg Tail OH = {avg_tail_oh:.1f} us \u2248 {loop_ratio:.1f} \u00d7 avg loop iteration ({avg_loop_us:.1f} us)')
    print(f'  \u2192 on average, a completed task waits ~{loop_ratio:.1f} loop iterations before being detected')
    print()

    # Data-driven insight: find the dominant phase (excluding early_ready which is typically trivial)
    work_phases = {p: phase_totals.get(p, 0) for p in ['scan', 'complete', 'dispatch']}
    dominant_phase = max(work_phases, key=work_phases.get)
    dominant_pct = work_phases[dominant_phase] / total_us * 100 if total_us > 0 else 0
    print(f'  Key insight: {phase_labels[dominant_phase].split(" (")[0]} phase consumes ~{dominant_pct:.0f}% of scheduler CPU.')
    if dominant_phase == 'dispatch':
        dispatch_lock_pct = sum(t.get('lock_dispatch_wait', 0) for t in threads.values()) / phase_totals.get('dispatch', 1) * 100
        print(f'  Within dispatch, lock contention accounts for {dispatch_lock_pct:.0f}% of time.')
        if miss_w > hit_w:
            print(f'  Dispatch miss (empty queue) dominates lock wait: miss={miss_w}us vs hit={hit_w}us.')
        print(f'  Cache flush (dc cvac + dsb sy) is the dominant non-lock cost.')
    elif dominant_phase == 'complete':
        complete_lock_pct = sum(t.get('lock_complete_wait', 0) for t in threads.values()) / phase_totals.get('complete', 1) * 100
        print(f'  Within complete, lock contention accounts for {complete_lock_pct:.0f}% of time.')
        print(f'  Fanout traversal and atomic ops dominate the non-lock cost.')
    elif dominant_phase == 'scan':
        print(f'  Scan phase overhead indicates too many root tasks or inefficient task graph traversal.')
    print('=' * 90)

    return 0


if __name__ == '__main__':
    sys.exit(main())