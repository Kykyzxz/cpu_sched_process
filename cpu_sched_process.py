from collections import deque
import matplotlib.pyplot as plt

class Process:
    def __init__(self, pid, arrival, burst, priority=0):
        # storage of all inputs
        self.id = pid
        self.arrival = arrival
        self.burst = burst
        self.priority = priority
        self.remaining = burst #mutable copy of the original data while burst stays the same for original calculations
        self.end = 0
        self.turnaround = 0
        self.waiting = 0
        self.response = None
        self.gantt = []

    def fcfs(self, processes):
        processes = sorted(processes, key=lambda p: p.arrival)
        time = 0
        gantt = []

        for p in processes: #if there's an idle process jump time forward and also checks if there's a process that needs to process already
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
            #only considers process that already arrived
            ready = [p for p in remaining if p.arrival <= time]

            if not ready:
                time += 1
                continue
            
            shortest = min(ready, key=lambda p: p.burst) #picks the one with the shortest burst time
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

        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)

        return completed, avg_wt, avg_tat, gantt

    def srt(self, processes):
        time = 0
        completed = []
        gantt = []
        last_pid = None

        while len(completed) < len(processes):
            # filters for the arrived process and unfinished process
            ready = [p for p in processes if p.arrival <= time and p.remaining > 0]

            if not ready:
                time += 1
                continue

            current = min(ready, key=lambda p: p.remaining)

            if current.response is None:
                current.response = time - current.arrival

            if last_pid != current.id:
                gantt.append([current.id, time, time + 1])
            else:
                gantt[-1][2] += 1

            current.remaining -= 1 #runs one at a time so it can preempt at every tick
            time += 1
            last_pid = current.id

            if current.remaining == 0:
                current.end = time
                current.turnaround = current.end - current.arrival
                current.waiting = current.turnaround - current.burst
                completed.append(current)

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

        while len(completed) < len(processes): #it enqueue all the processes that have arrived by the current time
            while i < len(processes) and processes[i].arrival <= time:
                queue.append(processes[i])
                i += 1

            if not queue:
                time += 1
                continue

            current = queue.popleft()

            if current.response is None:
                current.response = time - current.arrival
            # runs whichever has the smaller quantum or what's left
            exec_time = min(quantum, current.remaining)
            start = time
            time += exec_time
            end = time

            gantt.append((current.id, start, end))
            current.remaining -= exec_time

            while i < len(processes) and processes[i].arrival <= time:
                queue.append(processes[i])
                i += 1

            if current.remaining == 0:
                current.end = time
                current.turnaround = current.end - current.arrival
                current.waiting = current.turnaround - current.burst
                completed.append(current)
            else:
                queue.append(current)

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

            if not ready:
                time += 1
                continue

            #controls whether lower number is equal to higher priority or vice versa
            key_func = (lambda p: p.priority) if higher_priority_smaller else (lambda p: -p.priority)
            #it picks the best priority process from those already arrived
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

        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)

        return completed, avg_wt, avg_tat, gantt

    def priority_preemptive(self, processes, higher_priority_smaller=True): #almost the same as srt
        time = 0
        completed = []
        gantt = []
        last_pid = None

        while len(completed) < len(processes):
            ready = [p for p in processes if p.arrival <= time and p.remaining > 0]

            if not ready:
                time += 1
                continue

            key_func = (lambda p: p.priority) if higher_priority_smaller else (lambda p: -p.priority)
            current = min(ready, key=key_func)

            if current.response is None:
                current.response = time - current.arrival

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

        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)

        return completed, avg_wt, avg_tat, gantt

    def priority_rr(self, processes, quantum, higher_priority_smaller=True):
        time = 0
        completed = []
        gantt = []

        #sorted by priority each iteration when a new process arrivad
        remaining = sorted(processes, key=lambda p: (p.arrival, p.priority if higher_priority_smaller else -p.priority))
        queue = deque()
        i = 0

        while len(completed) < len(processes):
            while i < len(remaining) and remaining[i].arrival <= time:
                queue.append(remaining[i])
                i += 1

            sorted_queue = sorted(queue, key=lambda p: p.priority if higher_priority_smaller else -p.priority)
            queue = deque(sorted_queue)

            if not queue:
                time += 1
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

        avg_wt = sum(p.waiting for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)

        return completed, avg_wt, avg_tat, gantt


class Scheduler:
    def __init__(self, processes):
        self.processes = processes
        self.gantt = []

    def draw_gantt_chart(self, gantt, result, avg_wt, avg_tat, show_priority=False):
        fig = plt.figure(figsize=(14, 7))
        fig.patch.set_facecolor('#f5f6fa')

        # title
        fig.suptitle('CPU Scheduling Simulator', fontsize=16, fontweight='bold',
                    color='#2c3e50', y=0.95)

        # table section
        ax_table = fig.add_axes([0.02, 0.32, 0.96, 0.60])
        ax_table.set_facecolor('#f5f6fa')
        ax_table.axis('off')

        if show_priority:
            headers = ['Process', 'Arrival', 'Burst', 'Priority', 'Waiting', 'Turnaround', 'End']
            rows = [[p.id, p.arrival, p.burst, p.priority, p.waiting, p.turnaround, p.end] for p in result]
        else:
            headers = ['Process', 'Arrival', 'Burst', 'Waiting', 'Turnaround', 'End']
            rows = [[p.id, p.arrival, p.burst, p.waiting, p.turnaround, p.end] for p in result]

        if show_priority:
            avg_row = ['Average', '', '', '', f'{avg_wt:.2f}', f'{avg_tat:.2f}', '']
        else:
            avg_row = ['Average', '', '', f'{avg_wt:.2f}', f'{avg_tat:.2f}', '']
        rows.append(avg_row)

        table = ax_table.table(
            cellText=rows,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # header row style
        for col in range(len(headers)):
            cell = table[0, col]
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_edgecolor('#ffffff')

        # data rows style
        for row in range(1, len(rows) + 1):
            for col in range(len(headers)):
                cell = table[row, col]
                if row == len(rows):
                    # averages row
                    cell.set_facecolor('#d5e8d4')   
                    cell.set_text_props(color='#27ae60', fontweight='bold')
                elif row % 2 == 0:
                    cell.set_facecolor('#eaf0fb')
                    cell.set_text_props(color='#2c3e50')
                else:
                    cell.set_facecolor('#ffffff')
                    cell.set_text_props(color='#2c3e50')
                cell.set_edgecolor('#d0d0d0')

        ax_table.set_title('Results', color='#2c3e50', fontsize=12,
                        fontweight='bold', pad=12, loc='left')

        # gantt chart
        ax_gantt = fig.add_axes([0.02, 0.05, 0.90, 0.18])
        ax_gantt.set_facecolor('#ffffff')

        # soft professional colors
        palette = [
            '#5b9bd5', '#ed7d31', '#a9d18e', '#ffc000',
            '#7030a0', '#00b0f0', '#ff0000', '#92d050'
        ]
        unique_ids = list(dict.fromkeys(pid for pid, _, _ in gantt))
        color_map = {pid: palette[i % len(palette)] for i, pid in enumerate(unique_ids)}

        for pid, start, end in gantt:
            color = '#bdc3c7' if pid == "Idle" else color_map[pid]
            ax_gantt.barh(0, end - start, left=start, color=color,
                        edgecolor='white', height=0.5)
            ax_gantt.text((start + end) / 2, 0, str(pid), ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

        # time markers
        all_times = sorted(set(t for _, s, e in gantt for t in (s, e)))
        for t in all_times:
            ax_gantt.axvline(x=t, color='#bdc3c7', linewidth=0.8)

        ax_gantt.set_xlabel('Time Units', color='#2c3e50', fontsize=10)
        ax_gantt.set_yticks([])
        ax_gantt.set_xticks(all_times)
        ax_gantt.tick_params(colors='#2c3e50')
        ax_gantt.set_title('Gantt Chart', color='#2c3e50', fontsize=12,
                        fontweight='bold', loc='left')

        for spine in ax_gantt.spines.values():
            spine.set_edgecolor('#d0d0d0')

        ax_gantt.set_facecolor('#f9f9f9')

        plt.show()

def get_int(prompt, min_val=None): #function to validate each inputs
    while True:
        try:
            value = int(input(prompt))
            if min_val is not None and value < min_val: #min_val it enforces a minimum bound if one was given
                print(f"  Please enter a value of at least {min_val}.")
                continue
            return value
        except ValueError:
            print("  Invalid input. Please enter a whole number.")


# algorithms that need priority input
NEEDS_PRIORITY = {5, 6, 7}

# algorithms that need a quantum input
NEEDS_QUANTUM = {4, 7}

# main functon
if __name__ == "__main__":
    print("\nChoose scheduling algorithm:")
    print("  1. FCFS              (First-Come, First-Served)")
    print("  2. SJF               (Shortest Job First — non-preemptive)")
    print("  3. SRT               (Shortest Remaining Time — preemptive)")
    print("  4. Round Robin")
    print("  5. Priority          (Non-Preemptive)")
    print("  6. Priority          (Preemptive)")
    print("  7. Priority + RR     (Round Robin with Priority)")

    choice = get_int("\nEnter choice (1-7): ", min_val=1)
    while choice > 7:
        print("  Invalid choice. Please enter a number between 1 and 7.")
        choice = get_int("Enter choice (1-7): ", min_val=1)

    # ask for quantum if needed
    quantum = None
    if choice in NEEDS_QUANTUM:
        quantum = get_int("Enter Time Quantum: ", min_val=1)

    #collect process data
    needs_priority = choice in NEEDS_PRIORITY

    n = get_int("\nEnter number of processes: ", min_val=1)

    processes = []
    for i in range(n):
        print(f"\n  -- Process {i + 1} --")
        pid     = input("  Process ID: ").strip() or f"P{i+1}"
        arrival = get_int("  Arrival Time: ", min_val=0)
        burst   = get_int("  Burst Time:   ", min_val=1)

        priority = 0
        if needs_priority:
            priority = get_int("  Priority:     ")

        processes.append(Process(pid, arrival, burst, priority))

    helper = Process("_", 0, 0)   # dummy instance just to call instance methods

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

    show_priority = needs_priority
    print("\n" + "=" * 65)
    if show_priority:
        print(f"{'Process':<10}{'Arrival':<10}{'Burst':<10}{'Priority':<10}{'Waiting':<10}{'Turnaround':<12}{'End':<10}")
    else:
        print(f"{'Process':<10}{'Arrival':<10}{'Burst':<10}{'Waiting':<10}{'Turnaround':<12}{'End':<10}")
    print("=" * 65)

    for p in result:
        if show_priority:
            print(f"{p.id:<10}{p.arrival:<10}{p.burst:<10}{p.priority:<10}{p.waiting:<10}{p.turnaround:<12}{p.end:<10}")
        else:
            print(f"{p.id:<10}{p.arrival:<10}{p.burst:<10}{p.waiting:<10}{p.turnaround:<12}{p.end:<10}")

    print("=" * 65)
    print(f"\nAverage Waiting Time:    {avg_wt:.2f}")
    print(f"Average Turnaround Time: {avg_tat:.2f}")

    scheduler = Scheduler(processes)
    scheduler.draw_gantt_chart(gantt, result, avg_wt, avg_tat, show_priority=needs_priority)