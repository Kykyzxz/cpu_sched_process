from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Process:
    def __init__(self, pid, arrival, burst, priority=0):
        # storage of all inputs
        self.id = pid
        self.arrival = arrival
        self.burst = burst
        self.priority = priority
        self.remaining = burst  # mutable copy of the original data while burst stays the same for original calculations
        self.end = 0
        self.turnaround = 0
        self.waiting = 0
        self.response = None
        self.gantt = []

    def _insert_idle(self, gantt):
        #Fill gaps between consecutive gantt segments with Idle entries
        if not gantt:
            return gantt
        filled = []
        for i, seg in enumerate(gantt):
            pid, s, e = seg[0], seg[1], seg[2]
            if filled:
                prev_end = filled[-1][2]
                if s > prev_end:
                    filled.append(("Idle", prev_end, s))
            filled.append((pid, s, e))
        return filled
    
    def fcfs(self, processes):
        # sort by arrival time so earliest arrival goes first
        processes = sorted(processes, key=lambda p: p.arrival)
        time = 0
        gantt = []

        for p in processes:  # if theres an idle process jump time forward and also checks if theres a process that needs to process already
            if time < p.arrival:
                gantt.append(("Idle", time, p.arrival))
                time = p.arrival

            start = time
            time += p.burst
            end = time

            p.end = end
            p.turnaround = p.end - p.arrival
            p.waiting = p.turnaround - p.burst
            p.response = start - p.arrival
            gantt.append((p.id, start, end))

        avg_wt = sum(p.waiting for p in processes) / len(processes)
        avg_tat = sum(p.turnaround for p in processes) / len(processes)

        return processes, avg_wt, avg_tat, gantt

    def sjf(self, processes):
        time = 0
        completed = []
        remaining = processes[:]
        gantt = []

        while remaining:
            # only considers processes that have already arrived
            ready = [p for p in remaining if p.arrival <= time]
  
            if not ready:
            # jump to the next arrival if no process is ready
                next_t = min(p.arrival for p in remaining)
                gantt.append(("Idle", time, next_t))
                time = next_t
                continue

            shortest = min(ready, key=lambda p: p.burst)  # picks the one with the shortest burst time (non-preemptive)
            start = time
            time += shortest.burst
            end = time

            shortest.end = end
            shortest.turnaround = end - shortest.arrival
            shortest.waiting = shortest.turnaround - shortest.burst
            shortest.response = start - shortest.arrival

            gantt.append((shortest.id, start, end))
            completed.append(shortest)
            remaining.remove(shortest)

        gantt = self._insert_idle(gantt)
        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)
        return completed, avg_wt, avg_tat, gantt

    def srt(self, processes):
            time = 0
            completed = []
            gantt = []
            last_pid = None

            while len(completed) < len(processes):
                ready = [p for p in processes if p.arrival <= time and p.remaining > 0]

                if not ready:
                    next_t = min(p.arrival for p in processes if p.remaining > 0)
                    if last_pid != "Idle":
                        gantt.append(["Idle", time, next_t])
                    else:
                        gantt[-1][2] = next_t
                    time = next_t
                    last_pid = "Idle"
                    continue

                current = min(ready, key=lambda p: p.remaining)

                if current.response is None:
                    current.response = time - current.arrival

                # jump to next event instead of tick-by-tick
                next_arrival = min(
                    (p.arrival for p in processes if p.arrival > time and p.remaining > 0),
                    default=float('inf')
                )
                run_time = min(current.remaining, next_arrival - time)

                if last_pid != current.id:
                    gantt.append([current.id, time, time + run_time])
                else:
                    gantt[-1][2] += run_time

                current.remaining -= run_time
                time += run_time
                last_pid = current.id

                if current.remaining == 0:
                    current.end = time
                    current.turnaround = current.end - current.arrival
                    current.waiting = current.turnaround - current.burst
                    completed.append(current)

            gantt = self._insert_idle(gantt)
            avg_wt = sum(p.waiting for p in completed) / len(completed)
            avg_tat = sum(p.turnaround for p in completed) / len(completed)

            return completed, avg_wt, avg_tat, gantt

    def round_robin(self, processes, quantum):
        time = 0
        queue = deque()
        completed = []
        gantt = []

        processes = sorted(processes, key=lambda p: p.arrival)
        i = 0

        while len(completed) < len(processes):
            # enqueues all processes that have arrived by the current time
            while i < len(processes) and processes[i].arrival <= time:
                queue.append(processes[i])
                i += 1

                # jump to next arrival if queue is empty
            if not queue:
                next_t = processes[i].arrival
                gantt.append(("Idle", time, next_t))
                time = next_t
                continue
            
            current = queue.popleft()

            if current.response is None:
                current.response = time - current.arrival

            # runs whichever is smaller the quantum or what's left
            exec_time = min(quantum, current.remaining)
            start = time
            time += exec_time
            end = time

            gantt.append((current.id, start, end))
            current.remaining -= exec_time

            # checks for new arrivals after time has advanced
            while i < len(processes) and processes[i].arrival <= time:
                queue.append(processes[i])
                i += 1

            if current.remaining == 0:
                current.end = time
                current.turnaround = current.end - current.arrival
                current.waiting = current.turnaround - current.burst
                completed.append(current)
            else:
                queue.append(current)  # unfinished process goes to the back of the queue
                
        gantt = self._insert_idle(gantt)
        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)

        return completed, avg_wt, avg_tat, gantt

    def priority_non_preemptive(self, processes, higher_priority_smaller=True):
        time = 0
        completed = []
        remaining = processes[:]
        gantt = []

        while remaining:
            ready = [p for p in remaining if p.arrival <= time]

                # jump to next arrival
            if not ready:
                next_t = min(p.arrival for p in remaining)
                gantt.append(("Idle", time, next_t))
                time = next_t
                continue

            # controls whether lower number = higher priority or higher number = higher priority
            key_func = (lambda p: p.priority) if higher_priority_smaller else (lambda p: -p.priority)
            # picks the best-priority process from those already arrived, runs to completion without interruption
            current = min(ready, key=key_func)

            start = time
            time += current.burst
            end = time

            current.end = end
            current.turnaround = end - current.arrival
            current.waiting = current.turnaround - current.burst
            current.response = start - current.arrival

            gantt.append((current.id, start, end))
            completed.append(current)
            remaining.remove(current)
            
        gantt = self._insert_idle(gantt)
        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)

        return completed, avg_wt, avg_tat, gantt

    def priority_preemptive(self, processes, higher_priority_smaller=True):  # almost the same as srt
        time = 0
        completed = []
        gantt = []
        last_pid = None

        while len(completed) < len(processes):
            ready = [p for p in processes if p.arrival <= time and p.remaining > 0]

            if not ready:
                next_t = min(p.arrival for p in processes if p.remaining > 0)
                if last_pid != "Idle":
                    gantt.append(["Idle", time, next_t])
                else:
                    gantt[-1][2] = next_t
                time = next_t
                last_pid = "Idle"
                continue

            key_func = (lambda p: p.priority) if higher_priority_smaller else (lambda p: -p.priority)
            current = min(ready, key=key_func)

            if current.response is None:
                current.response = time - current.arrival

            # merges consecutive ticks of the same process into one gantt bar
            if last_pid != current.id:
                gantt.append([current.id, time, time + 1])
            else:
                gantt[-1][2] += 1

            current.remaining -= 1
            time += 1
            last_pid = current.id

            if current.remaining == 0:
                current.end = time
                current.turnaround = current.end - current.arrival
                current.waiting = current.turnaround - current.burst
                completed.append(current)
                
        gantt = self._insert_idle(gantt)
        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)

        return completed, avg_wt, avg_tat, gantt

    def priority_rr(self, processes, quantum, higher_priority_smaller=True):
        time = 0
        completed = []
        gantt = []

        # sorted by priority each iteration so newly arrived higher-priority processes jump ahead
        remaining = sorted(processes, key=lambda p: (p.arrival, p.priority if higher_priority_smaller else -p.priority))
        queue = deque()
        i = 0

        while len(completed) < len(processes):
            while i < len(remaining) and remaining[i].arrival <= time:
                queue.append(remaining[i])
                i += 1

            # re-sort queue by priority after adding new arrivals
            sorted_queue = sorted(queue, key=lambda p: p.priority if higher_priority_smaller else -p.priority)
            queue = deque(sorted_queue)

            if not queue:
                next_t = remaining[i].arrival
                gantt.append(("Idle", time, next_t))
                time = next_t
                continue
            current = queue.popleft()

            if current.response is None:
                current.response = time - current.arrival

            exec_time = min(quantum, current.remaining)
            start = time
            time += exec_time

            gantt.append((current.id, start, time))
            current.remaining -= exec_time

            while i < len(remaining) and remaining[i].arrival <= time:
                queue.append(remaining[i])
                i += 1

            if current.remaining == 0:
                current.end = time
                current.turnaround = current.end - current.arrival
                current.waiting = current.turnaround - current.burst
                completed.append(current)
            else:
                queue.append(current)
        
        gantt = self._insert_idle(gantt)
        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)

        return completed, avg_wt, avg_tat, gantt


# color pallete
PALETTE = ['#4C9BE8', '#E8834C', '#4CE8A0', '#E8D44C',
           '#A04CE8', '#4CE8E0', '#E84C6B', '#8BE84C']
HEADER_COLOR  = '#1a2744'
ROW_ODD       = '#ffffff'
ROW_EVEN      = '#f0f4ff'
AVG_ROW_BG    = '#e8f5e9'
AVG_ROW_FG    = '#2e7d32'
TEXT_DARK     = '#1a2744'
BG_COLOR      = '#f7f9fc'
IDLE_COLOR    = '#c8d0dc'
STARVATION_BG = '#fff0f0'   # light red background for starved processes
STARVATION_FG = '#c0392b'   # red text for starved processes

# starvation threshold 
STARVATION_THRESHOLD = 20


class Scheduler:
    def __init__(self, processes):
        self.processes = processes
        self.gantt = []

    def draw_gantt_chart(self, gantt, result, avg_wt, avg_tat, algo_name='', show_priority=False):

        total_time = max(p.end for p in result)

        # cpu utilization how much of total time the CPU was actually busy
        idle_time = sum(e - s for pid, s, e in gantt if pid == 'Idle')
        cpu_util  = ((total_time - idle_time) / total_time * 100) if total_time > 0 else 100.0

        # throughput processes completed per unit time
        throughput = len(result) / total_time if total_time > 0 else 0.0

        # flag starved processes waiting time above threshold
        starved = {p.id for p in result if p.waiting > STARVATION_THRESHOLD}

        n_rows  = len(result) + 2   # +1 header +1 avg row

        # dynamically size figure height based on number of process rows
        fig_h = max(7.5, 3.0 + n_rows * 0.48 + 2.2)
        n_segments = len(gantt)
        gantt_fig_w = max(14, min(120, n_segments * 2.5))
        fig = plt.figure(figsize=(gantt_fig_w, fig_h))
        fig.patch.set_facecolor(BG_COLOR)

        # gridspec
        gs = gridspec.GridSpec(
            4, 1,
            figure=fig,
            hspace=0.0,
            height_ratios=[0.10, 0.58, 0.08, 0.24],
            left=0.04, right=0.96,
            top=0.97, bottom=0.05
        )

        # title banner
        ax_title = fig.add_subplot(gs[0])
        ax_title.set_facecolor(HEADER_COLOR)
        ax_title.axis('off')
        ax_title.text(0.5, 0.65, 'CPU Scheduling Simulator',
                      ha='center', va='center', transform=ax_title.transAxes,
                      fontsize=15, fontweight='bold', color='#1a2744')
        ax_title.text(0.5, 0.20, f'Algorithm:  {algo_name}',
                      ha='center', va='center', transform=ax_title.transAxes,
                      fontsize=9, color='#1a2744')

        # results table
        ax_table = fig.add_subplot(gs[1])
        ax_table.set_facecolor(BG_COLOR)
        ax_table.axis('off')

        # response time column is always shown
        if show_priority:
            headers = ['Process', 'Arrival', 'Burst', 'Priority',
                       'Response', 'Waiting', 'Turnaround', 'End Time']
            rows = [[p.id, p.arrival, p.burst, p.priority,
                     p.response, p.waiting, p.turnaround, p.end] 
                    for p in result]
            avg_response = sum(p.response for p in result) / len(result)
            avg_row = ['— Average —', '', '', '',
                       f'{avg_response:.2f}', f'{avg_wt:.2f}', f'{avg_tat:.2f}', '']
        else:
            headers = ['Process', 'Arrival', 'Burst',
                       'Response', 'Waiting', 'Turnaround', 'End Time']
            rows = [[p.id, p.arrival, p.burst,
                     p.response, p.waiting, p.turnaround, p.end]
                    for p in result]
            avg_response = sum(p.response for p in result) / len(result)
            avg_row = ['— Average —', '', '',
                       f'{avg_response:.2f}', f'{avg_wt:.2f}', f'{avg_tat:.2f}', '']

        rows.append(avg_row)
        
        tbl = ax_table.table(
            cellText=rows,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            bbox=[0.0, 0.0, 1.0, 1.0]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)

        n_cols = len(headers)

        # header row styling
        for c in range(n_cols):
            cell = tbl[0, c]
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_edgecolor('#2e3f6e')
            cell.set_height(0.13)

        # data + avg rows styling
        for r in range(1, len(rows) + 1):
            is_avg     = (r == len(rows))
            process_id = result[r - 1].id if not is_avg else None
            is_starved = (process_id in starved) if process_id else False

            for c in range(n_cols):
                cell = tbl[r, c]
                cell.set_edgecolor('#d4dbe8')
                cell.set_height(0.10)

                if is_avg:
                    cell.set_facecolor(AVG_ROW_BG)
                    cell.set_text_props(color=AVG_ROW_FG, fontweight='bold')
                elif is_starved:
                    # highlight starved processes in red
                    cell.set_facecolor(STARVATION_BG)
                    cell.set_text_props(color=STARVATION_FG, fontweight='bold')
                elif r % 2 == 0:
                    cell.set_facecolor(ROW_EVEN)
                    cell.set_text_props(color=TEXT_DARK)
                else:
                    cell.set_facecolor(ROW_ODD)
                    cell.set_text_props(color=TEXT_DARK)

        # starvation warning note below table if any process is flagged
        if starved:
            ax_table.text(0.0, -0.01,
                          f'⚠  Starvation detected (waiting > {STARVATION_THRESHOLD}): '
                          + ', '.join(sorted(starved)),
                          transform=ax_table.transAxes,
                          fontsize=8, color=STARVATION_FG, va='top')

        # metrics strip 
        ax_metrics = fig.add_subplot(gs[2])
        ax_metrics.set_facecolor('#eef2fa')
        ax_metrics.axis('off')

        metrics_text = (
            f'CPU Utilization:  {cpu_util:.1f}%'
            f'          |          '
            f'Throughput:  {throughput:.4f} proc / time unit'
            f'          |          '
            f'Total Time:  {total_time}'
            f'          |          '
            f'Idle Time:  {idle_time}'
        )
        ax_metrics.text(0.5, 0.5, metrics_text,
                        ha='center', va='center',
                        transform=ax_metrics.transAxes,
                        fontsize=9, color=TEXT_DARK)

        # gantt chart
        ax_gantt = fig.add_subplot(gs[3])
        ax_gantt.set_facecolor('#ffffff')
        for spine in ax_gantt.spines.values():
            spine.set_edgecolor('#d4dbe8')
            spine.set_linewidth(0.8)

        unique_ids = list(dict.fromkeys(pid for pid, _, _ in gantt))
        color_map  = {pid: PALETTE[i % len(PALETTE)] for i, pid in enumerate(unique_ids)}

        MIN_BAR_WIDTH = 1.2
        MAX_BAR_WIDTH = 20.0
        bar_h = 0.52
        vis_pos = 0.0
        tick_positions = []

        for idx, (pid, start, end) in enumerate(gantt):
            dur = end - start
            vis_w = max(MIN_BAR_WIDTH, min(dur, MAX_BAR_WIDTH))
            color       = IDLE_COLOR if pid == 'Idle' else color_map[pid]
            label_color = '#666666' if pid == 'Idle' else 'white'

            ax_gantt.barh(0, vis_w, left=vis_pos,
                          color=color, edgecolor='white',
                          height=bar_h, linewidth=1.2)

            label = str(pid)
            ax_gantt.text(vis_pos + vis_w / 2, 0, label,
                          ha='center', va='center',
                          fontsize=7.5, fontweight='bold',
                          color=label_color, clip_on=True)

            tick_positions.append((vis_pos, start))
            vis_pos += vis_w

        tick_positions.append((vis_pos, gantt[-1][2]))

        seen_labels = {}
        for vx, t in tick_positions:
            seen_labels.setdefault(t, vx)

        vx_list = list(seen_labels.values())
        t_list  = list(seen_labels.keys())

        for vx in vx_list:
            ax_gantt.axvline(x=vx, color='#c8d0dc', linewidth=0.7, zorder=0)

        ax_gantt.set_yticks([])
        ax_gantt.set_xticks(vx_list)
        ax_gantt.set_xticklabels(t_list)
        rot = 45 if len(vx_list) > 12 else 0
        ax_gantt.tick_params(axis='x', colors=TEXT_DARK, labelsize=7, rotation=rot)
        ax_gantt.set_xlim(0, vis_pos)
        ax_gantt.set_ylim(-0.6, 0.6)
        ax_gantt.set_xlabel('Time Units', color=TEXT_DARK, fontsize=9, labelpad=4)
        ax_gantt.text(0.0, 0.97, 'Gantt Chart',
                      transform=ax_gantt.transAxes,
                      fontsize=9, fontweight='bold',
                      color=TEXT_DARK, va='top')
        plt.show()


# helpers

def get_int(prompt, min_val=None):  # keeps prompting until a valid integer is received
    while True:
        try:
            value = int(input(prompt))
            if min_val is not None and value < min_val:  # enforces a minimum bound if one was given
                print(f"  Please enter a value of at least {min_val}.")
                continue
            return value
        except ValueError:
            print("  Invalid input. Please enter a whole number.")


def get_yes_no(prompt):
    # keeps asking until the user types exactly y or n
    while True:
        answer = input(prompt).strip().lower()
        if answer in ('y', 'n'):
            return answer
        print("  Please enter y or n.")


# algorithms that need priority input
NEEDS_PRIORITY = {5, 6, 7}

# algorithms that need a quantum input
NEEDS_QUANTUM = {4, 7}

ALGO_NAMES = {
    1: 'FCFS — First-Come, First-Served',
    2: 'SJF — Shortest Job First (Non-Preemptive)',
    3: 'SRT — Shortest Remaining Time (Preemptive)',
    4: 'Round Robin',
    5: 'Priority Scheduling (Non-Preemptive)',
    6: 'Priority Scheduling (Preemptive)',
    7: 'Priority + Round Robin',
}

# main function
if __name__ == "__main__":
    while True:

        # choose algorithm
        print("\nChoose scheduling algorithm:")
        print("  1. FCFS              (First-Come, First-Served)")
        print("  2. SJF               (Shortest Job First — non-preemptive)")
        print("  3. SRT               (Shortest Remaining Time — preemptive)")
        print("  4. Round Robin")
        print("  5. Priority          (Non-Preemptive)")
        print("  6. Priority          (Preemptive)")
        print("  7. Priority + RR     (Round Robin with Priority)")

        # catches values above valid range, get_int with min_val=1 already blocks values below 1
        choice = get_int("\nEnter choice (1-7): ", min_val=1)
        while choice > 7:
            print("  Invalid choice. Please enter a number between 1 and 7.")
            choice = get_int("Enter choice (1-7): ", min_val=1)

        # ask for quantum if needed
        quantum = None
        if choice in NEEDS_QUANTUM:
            quantum = get_int("Enter Time Quantum: ", min_val=1)

        # collect process data
        needs_priority = choice in NEEDS_PRIORITY
        n = get_int("\nEnter number of processes: ", min_val=1)

        processes = []
        used_ids = set()  # tracks used process IDs to prevent duplicates

        for i in range(n):
            print(f"\n  -- Process {i + 1} --")

            # keep asking until a unique process ID is entered
            while True:
                pid = input("  Process ID: ").strip() or f"P{i+1}"
                if pid in used_ids:
                    print(f"  ID '{pid}' already exists. Please enter a unique ID.")
                else:
                    used_ids.add(pid)
                    break

            arrival = get_int("  Arrival Time: ", min_val=0)
            burst = get_int("  Burst Time:   ", min_val=1)

            priority = 0
            if needs_priority:
                priority = get_int("  Priority:     ")

            processes.append(Process(pid, arrival, burst, priority))

        # run chosen algorithm using dummy instance to call instance methods
        helper = Process("_", 0, 0)

        if choice == 1:
            result, avg_wt, avg_tat, gantt = helper.fcfs(processes)
        elif choice == 2:
            result, avg_wt, avg_tat, gantt = helper.sjf(processes)
        elif choice == 3:
            result, avg_wt, avg_tat, gantt = helper.srt(processes)
        elif choice == 4:
            result, avg_wt, avg_tat, gantt = helper.round_robin(processes, quantum)
        elif choice == 5:
            result, avg_wt, avg_tat, gantt = helper.priority_non_preemptive(processes)
        elif choice == 6:
            result, avg_wt, avg_tat, gantt = helper.priority_preemptive(processes)
        elif choice == 7:
            result, avg_wt, avg_tat, gantt = helper.priority_rr(processes, quantum)

        # show results window no terminal table, everything in matplotlib
        scheduler = Scheduler(processes)
        scheduler.draw_gantt_chart(
            gantt, result, avg_wt, avg_tat,
            algo_name=ALGO_NAMES[choice],
            show_priority=needs_priority
        )
        
        # ask to run again only accepts y or n, anything else re-prompts
        if get_yes_no("\nRun again? (y/n): ") != 'y':
            print("\nGoodbye!")
            break