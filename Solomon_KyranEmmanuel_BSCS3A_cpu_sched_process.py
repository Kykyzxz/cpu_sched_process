# CPU Scheduling Simulator — simulates 7 algorithms with a Tkinter GUI and embedded Matplotlib output
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')  # tells Matplotlib to render inside Tkinter windows


# Holds all data for one process: timing inputs and computed results.
# Every scheduling algorithm is a method here and returns (processes, avg_wt, avg_tat, gantt).
class Process:
    def __init__(self, pid, arrival, burst, priority=0):
        self.id         = pid
        self.arrival    = arrival
        self.burst      = burst        # total CPU time this process needs
        self.priority   = priority
        self.remaining  = burst        # counts down as the process runs (used by preemptive algos)
        self.end        = 0            # clock time when the process finishes
        self.turnaround = 0            # end - arrival
        self.waiting    = 0            # turnaround - burst
        self.response   = None         # time from arrival until the process first touches the CPU

    def insert_idle(self, gantt):
        # Fills time gaps in the gantt list with ("Idle", gap_start, gap_end) entries.
        # Needed after algorithms that skip time without explicitly recording idle periods.
        if not gantt:
            return gantt
        filled = []
        for seg in gantt:
            pid, s, e = seg[0], seg[1], seg[2]
            if filled:
                prev_end = filled[-1][2]
                if s > prev_end:                          # gap detected between segments
                    filled.append(("Idle", prev_end, s))
            filled.append((pid, s, e))
        return filled

    def fcfs(self, processes):
        # First-Come First-Served: sort by arrival, run each process to completion in order.
        processes = sorted(processes, key=lambda p: p.arrival)
        time = 0
        gantt = []
        for p in processes:
            if time < p.arrival:                          # CPU is idle before this process arrives
                gantt.append(("Idle", time, p.arrival))
                time = p.arrival
            start = time
            time += p.burst
            p.end        = time
            p.turnaround = p.end - p.arrival
            p.waiting    = p.turnaround - p.burst
            p.response   = start - p.arrival
            gantt.append((p.id, start, time))
        avg_wt  = sum(p.waiting    for p in processes) / len(processes)
        avg_tat = sum(p.turnaround for p in processes) / len(processes)
        return processes, avg_wt, avg_tat, gantt

    def sjf(self, processes):
        # Shortest Job First (non-preemptive): pick the ready process with the smallest burst.
        time = 0
        completed = []
        remaining = processes[:]
        gantt = []
        while remaining:
            ready = [p for p in remaining if p.arrival <= time]
            if not ready:
                next_t = min(p.arrival for p in remaining)  # no process ready — jump to next arrival
                gantt.append(("Idle", time, next_t))
                time = next_t
                continue
            shortest = min(ready, key=lambda p: p.burst)    # pick process with smallest burst
            start = time
            time += shortest.burst
            shortest.end        = time
            shortest.turnaround = time - shortest.arrival
            shortest.waiting    = shortest.turnaround - shortest.burst
            shortest.response   = start - shortest.arrival
            gantt.append((shortest.id, start, time))
            completed.append(shortest)
            remaining.remove(shortest)
        gantt   = self.insert_idle(gantt)
        avg_wt  = sum(p.waiting    for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)
        return completed, avg_wt, avg_tat, gantt

    def srt(self, processes):
        # Shortest Remaining Time (preemptive SJF): always run the process with the least remaining time.
        # Instead of tick-by-tick, we run until the next arrival so we can re-evaluate at each event.
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
                    gantt[-1][2] = next_t                  # extend existing idle bar
                time = next_t
                last_pid = "Idle"
                continue
            current = min(ready, key=lambda p: p.remaining)  # process with least remaining time
            if current.response is None:
                current.response = time - current.arrival
            # Run only until the next process arrives so we can re-check who has the shortest remaining time
            next_arrival = min(
                (p.arrival for p in processes if p.arrival > time and p.remaining > 0),
                default=float('inf')
            )
            run_time = min(current.remaining, next_arrival - time)
            if last_pid != current.id:
                gantt.append([current.id, time, time + run_time])
            else:
                gantt[-1][2] += run_time                   # same process continues — extend its bar
            current.remaining -= run_time
            time += run_time
            last_pid = current.id
            if current.remaining == 0:
                current.end        = time
                current.turnaround = current.end - current.arrival
                current.waiting    = current.turnaround - current.burst
                completed.append(current)
        gantt   = self.insert_idle(gantt)
        avg_wt  = sum(p.waiting    for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)
        return completed, avg_wt, avg_tat, gantt

    def round_robin(self, processes, quantum):
        # Round Robin: each process gets a fixed time slice (quantum) before going to the back of the queue.
        time = 0
        queue = deque()
        completed = []
        gantt = []
        processes = sorted(processes, key=lambda p: p.arrival)
        i = 0
        while len(completed) < len(processes):
            while i < len(processes) and processes[i].arrival <= time:  # admit newly arrived processes
                queue.append(processes[i])
                i += 1
            if not queue:
                next_t = processes[i].arrival               # CPU idle — jump to next arrival
                gantt.append(("Idle", time, next_t))
                time = next_t
                continue
            current = queue.popleft()
            if current.response is None:
                current.response = time - current.arrival
            exec_time = min(quantum, current.remaining)     # run for at most one quantum
            start = time
            time += exec_time
            gantt.append((current.id, start, time))
            current.remaining -= exec_time
            while i < len(processes) and processes[i].arrival <= time:  # admit arrivals during this slice
                queue.append(processes[i])
                i += 1
            if current.remaining == 0:
                current.end        = time
                current.turnaround = current.end - current.arrival
                current.waiting    = current.turnaround - current.burst
                completed.append(current)
            else:
                queue.append(current)                       # process not finished — send it back
        gantt   = self.insert_idle(gantt)
        avg_wt  = sum(p.waiting    for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)
        return completed, avg_wt, avg_tat, gantt

    def priority_non_preemptive(self, processes, higher_priority_smaller=True):
        # Non-preemptive Priority: pick the highest-priority ready process and run it to completion.
        # By default, a smaller priority number means higher priority.
        time = 0
        completed = []
        remaining = processes[:]
        gantt = []
        while remaining:
            ready = [p for p in remaining if p.arrival <= time]
            if not ready:
                next_t = min(p.arrival for p in remaining)
                gantt.append(("Idle", time, next_t))
                time = next_t
                continue
            key_func = (lambda p: p.priority) if higher_priority_smaller else (lambda p: -p.priority)
            current = min(ready, key=key_func)              # pick highest-priority process
            start = time
            time += current.burst
            current.end        = time
            current.turnaround = time - current.arrival
            current.waiting    = current.turnaround - current.burst
            current.response   = start - current.arrival
            gantt.append((current.id, start, time))
            completed.append(current)
            remaining.remove(current)
        gantt   = self.insert_idle(gantt)
        avg_wt  = sum(p.waiting    for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)
        return completed, avg_wt, avg_tat, gantt

    def priority_preemptive(self, processes, higher_priority_smaller=True):
        # Preemptive Priority: runs tick-by-tick so a higher-priority arrival can immediately preempt.
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
            if last_pid != current.id:
                gantt.append([current.id, time, time + 1])  # new process on CPU — start a new bar
            else:
                gantt[-1][2] += 1                            # same process — extend existing bar by 1 tick
            current.remaining -= 1
            time += 1
            last_pid = current.id
            if current.remaining == 0:
                current.end        = time
                current.turnaround = current.end - current.arrival
                current.waiting    = current.turnaround - current.burst
                completed.append(current)
        gantt   = self.insert_idle(gantt)
        avg_wt  = sum(p.waiting    for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)
        return completed, avg_wt, avg_tat, gantt

    def priority_rr(self, processes, quantum, higher_priority_smaller=True):
        # Priority + Round Robin: queue is sorted by priority at every step;
        # processes with equal priority share the CPU using Round Robin time slices.
        time = 0
        completed = []
        gantt = []
        remaining = sorted(processes, key=lambda p: (
            p.arrival, p.priority if higher_priority_smaller else -p.priority))
        queue = deque()
        i = 0
        while len(completed) < len(processes):
            while i < len(remaining) and remaining[i].arrival <= time:
                queue.append(remaining[i])
                i += 1
            # Re-sort the ready queue so the highest-priority process is always at the front
            sorted_queue = sorted(
                queue, key=lambda p: p.priority if higher_priority_smaller else -p.priority)
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
                current.end        = time
                current.turnaround = current.end - current.arrival
                current.waiting    = current.turnaround - current.burst
                completed.append(current)
            else:
                queue.append(current)                       # not finished — re-queue it
        gantt   = self.insert_idle(gantt)
        avg_wt  = sum(p.waiting    for p in completed) / len(completed)
        avg_tat = sum(p.turnaround for p in completed) / len(completed)
        return completed, avg_wt, avg_tat, gantt


# Softer, eye-friendly colour palette
PALETTE       = ['#5B9BD5','#70A870','#C97B84','#C9A96E','#8B7EC8','#5BADB5','#C8765A','#7AAF7A']
HEADER_COLOR  = '#2C3E6B'   # deep navy — panel and table header backgrounds
ROW_ODD       = '#FFFFFF'
ROW_EVEN      = '#EEF2FA'
AVG_ROW_BG    = '#E8F5E9'
AVG_ROW_FG    = '#2E7D32'
TEXT_DARK     = '#1E2A45'
BG_COLOR      = '#F4F6FB'
IDLE_COLOR    = '#B8C4D4'   # muted slate-grey for idle Gantt segments
STARVATION_BG = '#FFF0F0'
STARVATION_FG = '#C0392B'
STARVATION_THRESHOLD = 20   # waiting time above this flags a process as starved (highlighted red)

# Left-panel UI colours
PANEL_BG      = '#1E2D50'   # dark navy sidebar background
PANEL_ACCENT  = '#5B9BD5'   # soft blue accent (dividers, headings, run button)
CARD_BG       = '#253660'   # slightly lighter card background
ENTRY_BG      = '#1A2744'   # entry field background
LABEL_DIM     = '#8BAAC8'   # dimmed label text

# Integer key → algorithm display name, used by the dropdown and the results banner
ALGO_NAMES = {
    1: 'FCFS — First-Come, First-Served',
    2: 'SJF — Shortest Job First (Non-Preemptive)',
    3: 'SRT — Shortest Remaining Time (Preemptive)',
    4: 'Round Robin',
    5: 'Priority Scheduling (Non-Preemptive)',
    6: 'Priority Scheduling (Preemptive)',
    7: 'Priority + Round Robin',
}

NEEDS_PRIORITY     = {5, 6, 7}  # algorithms that require a Priority column in the process cards
NEEDS_QUANTUM      = {4, 7}     # algorithms that require the Time Quantum entry field
PROCESSES_PER_PAGE = 5          # max process cards visible at once before pagination appears
GANTT_SEGS_PER_ROW = 16         # segments per Gantt row; excess wraps to the next line


# Two-panel Tkinter app: left panel = controls, right panel = scrollable results
class CPUSchedulerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CPU Scheduling Simulator")
        self.root.configure(bg='#1E2A45')
        self.root.state('zoomed')            # launch the window already maximized

        self.process_entries = []            # list of dicts, one dict per process card
        self.current_page    = 0            # index of the currently visible page of cards
        self.n_processes     = 0            # total number of processes entered by the user
        self.current_fig     = None         # most recent Matplotlib figure, saved here for export

        self.build_layout()

    def build_layout(self):
        #  Left panel: fixed width, never shrinks
        self.left_outer = tk.Frame(self.root, bg=PANEL_BG, width=390)
        self.left_outer.pack(side='left', fill='y')
        self.left_outer.pack_propagate(False)

        #  Right panel fills the rest 
        self.right_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.right_frame.pack(side='right', fill='both', expand=True)

        self.build_left_panel()
        self.build_right_panel()

    def build_left_panel(self):
        #  Button bar — packed FIRST with side='bottom' so it is ALWAYS pinned 
        # Packing bottom_frame before the scrollable area guarantees the buttons
        # are never pushed off-screen regardless of how many cards appear above.
        self.bottom_frame = tk.Frame(self.left_outer, bg=PANEL_BG)
        self.bottom_frame.pack(side='bottom', fill='x', padx=20, pady=(0, 16))

        tk.Frame(self.bottom_frame, bg=PANEL_ACCENT, height=2).pack(fill='x', pady=(0, 8))

        btn_row = tk.Frame(self.bottom_frame, bg=PANEL_BG)
        btn_row.pack(fill='x')

        tk.Button(btn_row, text='▶  RUN',
                  command=self.run_simulation,
                  bg=PANEL_ACCENT, fg='white',
                  font=('Courier New', 13, 'bold'),
                  relief='flat', cursor='hand2',
                  pady=12).pack(side='left', fill='x', expand=True, padx=(0, 4))

        # Save button is disabled on launch; it becomes active after the first simulation runs
        self.save_btn = tk.Button(btn_row, text='💾  SAVE',
                                  command=self.save_results,
                                  bg='#3A7D44', fg='white',
                                  font=('Courier New', 11, 'bold'),
                                  relief='flat', cursor='hand2',
                                  pady=12, state='disabled')
        self.save_btn.pack(side='left', fill='x', expand=False, padx=(4, 0), ipadx=10)

        # Title 
        tk.Label(self.left_outer, text="CPU Scheduling\nSimulator",
                 bg=PANEL_BG, fg='white',
                 font=('Courier New', 16, 'bold'),
                 justify='center').pack(side='top', pady=(18, 5))

        # Decorative divider
        tk.Frame(self.left_outer, bg=PANEL_ACCENT, height=2).pack(
            side='top', fill='x', padx=20, pady=5)

        # Algorithm label + dropdown
        self.section_label(self.left_outer, "① Algorithm")
        self.algo_var = tk.StringVar(value='FCFS — First-Come, First-Served')
        algo_cb = ttk.Combobox(self.left_outer, textvariable=self.algo_var,
                               values=list(ALGO_NAMES.values()),
                               state='readonly', width=36,
                               font=('Courier New', 10))
        algo_cb.pack(side='top', padx=20, pady=(0, 6), anchor='w')
        algo_cb.bind('<<ComboboxSelected>>', self.on_algo_change)  # fires when user picks an algorithm

        #  Time Quantum — hidden by default, shown for RR / Priority+RR 
        # quantum_frame is packed into left_outer dynamically by on_algo_change.
        self.quantum_frame = tk.Frame(self.left_outer, bg=PANEL_BG)
        tk.Label(self.quantum_frame, text="Time Quantum",
                 bg=PANEL_BG, fg=LABEL_DIM,
                 font=('Courier New', 11)).pack(anchor='w')
        self.quantum_var = tk.StringVar(value='2')
        tk.Entry(self.quantum_frame, textvariable=self.quantum_var,
                 width=12, font=('Courier New', 11),
                 bg=ENTRY_BG, fg='white',
                 insertbackground='white',
                 relief='flat').pack(anchor='w', pady=(2, 6))
        # quantum_frame is intentionally not packed here; on_algo_change controls its visibility

        # Number of Processes 
        self.section_label(self.left_outer, "② Number of Processes")
        n_frame = tk.Frame(self.left_outer, bg=PANEL_BG)
        n_frame.pack(side='top', fill='x', padx=20, pady=(0, 6))
        self.n_var = tk.StringVar(value='3')
        tk.Entry(n_frame, textvariable=self.n_var,
                 width=12, font=('Courier New', 11),
                 bg=ENTRY_BG, fg='white',
                 insertbackground='white',
                 relief='flat').pack(side='left')
        # "Set" triggers generate_process_fields which builds or rebuilds the process cards
        tk.Button(n_frame, text='Set',
                  command=self.generate_process_fields,
                  bg=PANEL_ACCENT, fg='white',
                  font=('Courier New', 10, 'bold'),
                  relief='flat', cursor='hand2',
                  padx=12).pack(side='left', padx=(10, 0))

        # Process cards area — scrollable so cards are never cut off 
        # A canvas + scrollbar wrapper means process cards scroll vertically when
        # there is not enough room when the Time Quantum field is visible
        # page_frame lives BELOW the canvas so it is always visible.
        self.cards_canvas  = tk.Canvas(self.left_outer, bg=PANEL_BG,
                                       highlightthickness=0)
        self.cards_scrollbar = ttk.Scrollbar(self.left_outer, orient='vertical',
                                             command=self.cards_canvas.yview)
        self.cards_canvas.configure(yscrollcommand=self.cards_scrollbar.set)

        # pack scrollbar first so it sits to the right of the canvas
        # scrollbar is hidden by default — only shown for RR and Priority+RR
        self.cards_canvas.pack(side='top', fill='both', expand=True)

        # inner frame that holds the actual process cards
        self.cards_container = tk.Frame(self.cards_canvas, bg=PANEL_BG)
        self.cards_canvas.create_window((0, 0), window=self.cards_container, anchor='nw')

        self.cards_container.bind('<Configure>',
            lambda e: self.cards_canvas.configure(
                scrollregion=self.cards_canvas.bbox('all')))

        # stretch inner frame to match canvas width
        self.cards_canvas.bind('<Configure>',
            lambda e: self.cards_canvas.itemconfig('all', width=e.width))

        # mousewheel scrolls the cards only when scrolling is active (RR / Priority+RR)
        self.cards_canvas.bind('<Enter>',
            lambda e: self.cards_canvas.bind_all(
                '<MouseWheel>',
                lambda ev: self.cards_canvas.yview_scroll(-1*(ev.delta//120), 'units'))
            if self.get_algo_choice() in NEEDS_QUANTUM else None)
        self.cards_canvas.bind('<Leave>',
            lambda e: self.cards_canvas.unbind_all('<MouseWheel>'))

        self.process_area = tk.Frame(self.cards_container, bg=PANEL_BG)
        self.process_area.pack(fill='x', padx=20)

        # page_frame is packed inside cards_container so it always sits right below the cards
        self.page_frame = tk.Frame(self.cards_container, bg=PANEL_BG)
        self.page_frame.pack(fill='x', padx=20, pady=(0, 4))

    def section_label(self, parent, text):
        # Reusable helper that renders a blue bold section heading in the left panel
        tk.Label(parent, text=text,
                 bg=PANEL_BG, fg=PANEL_ACCENT,
                 font=('Courier New', 12, 'bold')).pack(
                     side='top', anchor='w', padx=20, pady=(12, 3))

    def build_right_panel(self):
        # The right panel is a tk.Canvas + scrollbar so results can scroll vertically.
        # results_inner is the actual content frame embedded inside the canvas via create_window.
        self.right_canvas = tk.Canvas(self.right_frame, bg=BG_COLOR, highlightthickness=0)
        self.right_scrollbar = ttk.Scrollbar(self.right_frame, orient='vertical',
                                             command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        self.right_scrollbar.pack(side='right', fill='y')
        self.right_canvas.pack(side='left', fill='both', expand=True)

        self.results_inner = tk.Frame(self.right_canvas, bg=BG_COLOR)
        self.results_window = self.right_canvas.create_window(  # pins results_inner inside the canvas
            (0, 0), window=self.results_inner, anchor='nw')

        self.results_inner.bind('<Configure>', self.on_results_configure)   # update scroll region on resize
        self.right_canvas.bind('<Configure>', self.on_right_canvas_resize)  # stretch inner frame to canvas width

        # Mousewheel scrolling on the right panel
        self.right_canvas.bind_all(
            '<MouseWheel>',
            lambda e: self.right_canvas.yview_scroll(-1*(e.delta//120), 'units'))

        # Placeholder label shown before any simulation has been run
        self.placeholder = tk.Label(
            self.results_inner,
            text="Results will appear here\nafter running the simulation.",
            bg=BG_COLOR, fg='#A0B0C8',
            font=('Courier New', 13),
            justify='center')
        self.placeholder.pack(expand=True, pady=120)

    def on_results_configure(self, event):
        # Called whenever results_inner changes size — keeps the scrollbar range accurate
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox('all'))

    def on_right_canvas_resize(self, event):
        # Keeps results_inner as wide as the canvas so it never looks narrower than the panel
        self.right_canvas.itemconfig(self.results_window, width=event.width)

    def on_algo_change(self, event=None):
        # Shows or hides the Time Quantum field based on the selected algorithm,
        # and rebuilds the process cards to add or remove the Priority column.
        choice = self.get_algo_choice()
        if choice in NEEDS_QUANTUM:
            # Insert quantum_frame just above the cards canvas so it never displaces pagination
            self.quantum_frame.pack(side='top', fill='x', padx=20,
                                    before=self.cards_canvas)
            # Enable scrolling on the cards canvas — extra space is taken by the quantum field
            self.cards_canvas.configure(yscrollcommand=self.cards_scrollbar.set)
            self.cards_scrollbar.pack(side='right', fill='y',
                                      before=self.cards_canvas)
        else:
            self.quantum_frame.pack_forget()
            # Disable scrolling for algorithms that don't need the quantum field
            self.cards_canvas.configure(yscrollcommand='')
            self.cards_scrollbar.pack_forget()
        if self.process_entries:
            self.generate_process_fields()

    def get_algo_choice(self):
        # Looks up the dropdown text in ALGO_NAMES and returns its integer key (1–7)
        name = self.algo_var.get()
        for k, v in ALGO_NAMES.items():
            if v == name:
                return k
        return 1

    def generate_process_fields(self):
        # Destroys existing process cards and creates fresh ones for the requested count.
        # Each card has entry fields for PID, Arrival, Burst, and Priority (if needed).
        try:
            n = int(self.n_var.get())
            if n < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of processes (min 1).")
            return

        self.n_processes     = n
        self.process_entries = []
        self.current_page    = 0

        for w in self.process_area.winfo_children():  # clear old cards
            w.destroy()
        for w in self.page_frame.winfo_children():    # clear old pagination buttons
            w.destroy()

        choice         = self.get_algo_choice()
        needs_priority = choice in NEEDS_PRIORITY

        for i in range(n):
            # Each process is stored as a dict of StringVars so run_simulation can read the values later
            row = {
                'frame':    None,
                'pid':      tk.StringVar(value=f'P{i+1}'),
                'arrival':  tk.StringVar(value='0'),
                'burst':    tk.StringVar(value='1'),
                'priority': tk.StringVar(value='1'),
            }
            pf = tk.Frame(self.process_area, bg=CARD_BG, bd=0, relief='flat')
            pf.pack(fill='x', pady=4)
            row['frame'] = pf

            tk.Label(pf, text=f'Process {i+1}',
                     bg=CARD_BG, fg=PANEL_ACCENT,
                     font=('Courier New', 11, 'bold')).grid(
                         row=0, column=0, columnspan=4,
                         sticky='w', padx=8, pady=(8, 2))

            fields = [('ID', row['pid']), ('Arrival', row['arrival']), ('Burst', row['burst'])]
            if needs_priority:
                fields.append(('Priority', row['priority']))  # only add Priority column when required

            for col, (lbl, var) in enumerate(fields):
                tk.Label(pf, text=lbl, bg=CARD_BG, fg=LABEL_DIM,
                         font=('Courier New', 10)).grid(
                             row=1, column=col, padx=(8, 2), pady=(0, 2), sticky='w')
                tk.Entry(pf, textvariable=var, width=8,
                         font=('Courier New', 11),
                         bg=ENTRY_BG, fg='white',
                         insertbackground='white',
                         relief='flat').grid(
                             row=2, column=col, padx=(8, 2), pady=(0, 10), sticky='w')

            self.process_entries.append(row)

        self.apply_pagination()

    def apply_pagination(self):
        # Hides all cards, then shows only the slice that belongs to the current page
        for row in self.process_entries:
            row['frame'].pack_forget()

        start = self.current_page * PROCESSES_PER_PAGE
        end   = min(start + PROCESSES_PER_PAGE, self.n_processes)
        for row in self.process_entries[start:end]:
            row['frame'].pack(fill='x', pady=4)

        for w in self.page_frame.winfo_children():    # rebuild pagination buttons for this page
            w.destroy()

        total_pages = (self.n_processes - 1) // PROCESSES_PER_PAGE + 1
        if total_pages > 1:
            tk.Label(self.page_frame,
                     text=f'Page {self.current_page+1} of {total_pages}',
                     bg=PANEL_BG, fg=LABEL_DIM,
                     font=('Courier New', 8)).pack(side='left')
            if self.current_page > 0:                 # only show Prev when not on the first page
                tk.Button(self.page_frame, text='◀ Prev',
                          command=self.prev_page,
                          bg=CARD_BG, fg='white',
                          font=('Courier New', 8), relief='flat',
                          cursor='hand2', padx=6).pack(side='left', padx=4)
            if self.current_page < total_pages - 1:   # only show Next when not on the last page
                tk.Button(self.page_frame, text='Next ▶',
                          command=self.next_page,
                          bg=CARD_BG, fg='white',
                          font=('Courier New', 8), relief='flat',
                          cursor='hand2', padx=6).pack(side='left', padx=4)

    def prev_page(self):
        self.current_page -= 1
        self.apply_pagination()

    def next_page(self):
        self.current_page += 1
        self.apply_pagination()

    def run_simulation(self):
        # Reads and validates all inputs, builds Process objects, calls the chosen algorithm,
        # then hands the results to draw_results() for display.
        if not self.process_entries:
            messagebox.showerror("Error", "Please set the number of processes first.")
            return
        try:
            if len(self.process_entries) < 3:
                raise ValueError("Minimum of 3 processes is required to run the simulation.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        choice         = self.get_algo_choice()
        needs_priority = choice in NEEDS_PRIORITY

        quantum = None
        if choice in NEEDS_QUANTUM:
            try:
                quantum = int(self.quantum_var.get())
                if quantum < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Time Quantum must be a positive integer.")
                return

        processes = []
        used_ids  = set()
        for i, row in enumerate(self.process_entries):
            pid = row['pid'].get().strip() or f'P{i+1}'
            if pid in used_ids:                             # duplicate IDs cause ambiguous table rows
                messagebox.showerror("Error",
                    f"Duplicate Process ID '{pid}'. Please use unique IDs.")
                return
            used_ids.add(pid)
            try:
                arrival = int(row['arrival'].get())
                burst   = int(row['burst'].get())
                if arrival < 0 or burst < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error",
                    f"Process {pid}: Arrival must be ≥ 0 and Burst must be ≥ 1.")
                return
            priority = 0
            if needs_priority:
                try:
                    priority = int(row['priority'].get())
                except ValueError:
                    messagebox.showerror("Error", f"Process {pid}: Priority must be an integer.")
                    return
            processes.append(Process(pid, arrival, burst, priority))

        # A throwaway Process instance is created just to call the algorithm methods on it
        helper = Process("_", 0, 0)
        if   choice == 1:
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

        self.draw_results(gantt, result, avg_wt, avg_tat,
                          algo_name=ALGO_NAMES[choice],
                          show_priority=needs_priority)

    def save_results(self):
        # Opens a save dialog and exports the stored Matplotlib figure to PNG / PDF / SVG / JPEG
        if self.current_fig is None:
            messagebox.showwarning("No Results", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[
                ('PNG Image',    '*.png'),
                ('PDF Document', '*.pdf'),
                ('SVG Vector',   '*.svg'),
                ('JPEG Image',   '*.jpg'),
            ],
            title='Save Results As')
        if not path:
            return
        try:
            self.current_fig.savefig(path, dpi=150, bbox_inches='tight',
                                     facecolor=self.current_fig.get_facecolor())
            messagebox.showinfo("Saved", f"Results saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def draw_results(self, gantt, result, avg_wt, avg_tat, algo_name='', show_priority=False):
        # Clears old results, builds a Matplotlib figure with four stacked sections
        # (title banner → results table → metrics strip → Gantt chart),
        # then embeds it in the scrollable right panel.

        for w in self.results_inner.winfo_children():
            w.destroy()
        self.right_canvas.yview_moveto(0)  # scroll back to the top so the banner is visible first

        total_time = max(p.end for p in result)
        idle_time  = sum(e - s for pid, s, e in gantt if pid == 'Idle')
        cpu_util   = ((total_time - idle_time) / total_time * 100) if total_time > 0 else 100.0
        throughput = len(result) / total_time if total_time > 0 else 0.0
        starved    = {p.id for p in result if p.waiting > STARVATION_THRESHOLD}  # processes waiting too long

        # Figure height scales up when there are more processes (taller table) or more Gantt rows
        n_gantt_rows  = (len(gantt) - 1) // GANTT_SEGS_PER_ROW + 1
        n_data_rows   = len(result) + 2             # header + one row per process + average row
        table_h_ratio = max(0.40, 0.065 * n_data_rows)
        gantt_h_ratio = 0.16 * n_gantt_rows

        panel_w = max(self.right_frame.winfo_width(), 820)
        fig_w   = panel_w / 96                      # convert pixels to inches for Matplotlib
        fig_h   = max(7.5, 1.2 + table_h_ratio * 10 + 0.8 + gantt_h_ratio * 10)

        fig = plt.Figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
        self.current_fig = fig  # keep a reference so save_results() can export it

        # GridSpec splits the figure vertically into title / table / metrics / gantt
        gs = gridspec.GridSpec(
            4, 1, figure=fig, hspace=0.0,
            height_ratios=[0.10, table_h_ratio, 0.07, gantt_h_ratio],
            left=0.03, right=0.97, top=0.97, bottom=0.03)

        # Title banner — solid dark background, pure white / gold text for maximum contrast
        ax_title = fig.add_subplot(gs[0])
        ax_title.set_facecolor('#1A2744')
        ax_title.axis('off')
        ax_title.text(0.5, 0.68, 'CPU Scheduling Simulator',
                      ha='center', va='center', transform=ax_title.transAxes,
                      fontsize=13, fontweight='bold', color='#1E2A45')
        ax_title.text(0.5, 0.22, f'Algorithm:  {algo_name}',  # subtitle shows which algorithm was run
                      ha='center', va='center', transform=ax_title.transAxes,
                      fontsize=9, fontweight='bold', color='#1E2A45')

        # Results table — columns vary based on whether the algorithm uses priority
        ax_table = fig.add_subplot(gs[1])
        ax_table.set_facecolor(BG_COLOR)
        ax_table.axis('off')

        if show_priority:
            headers = ['Process','Arrival','Burst','Priority',
                       'Response','Waiting','Turnaround','End Time']
            rows = [[p.id, p.arrival, p.burst, p.priority,
                     p.response, p.waiting, p.turnaround, p.end]
                    for p in result]
            avg_response = sum(p.response for p in result) / len(result)
            avg_row = ['— Average —','','','',
                       f'{avg_response:.2f}', f'{avg_wt:.2f}', f'{avg_tat:.2f}', '']
        else:
            headers = ['Process','Arrival','Burst',
                       'Response','Waiting','Turnaround','End Time']
            rows = [[p.id, p.arrival, p.burst,
                     p.response, p.waiting, p.turnaround, p.end]
                    for p in result]
            avg_response = sum(p.response for p in result) / len(result)
            avg_row = ['— Average —','','',
                       f'{avg_response:.2f}', f'{avg_wt:.2f}', f'{avg_tat:.2f}', '']

        rows.append(avg_row)  # average row goes at the bottom of the table
        tbl = ax_table.table(cellText=rows, colLabels=headers,
                             loc='center', cellLoc='center',
                             bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        n_cols = len(headers)

        for c in range(n_cols):  # style the header row: dark navy background, white bold text
            cell = tbl[0, c]
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_edgecolor('#2e3f6e')
            cell.set_height(0.13)

        for r in range(1, len(rows) + 1):
            is_avg     = (r == len(rows))                          # last row is always the average
            process_id = result[r-1].id if not is_avg else None
            is_starved = (process_id in starved) if process_id else False
            for c in range(n_cols):
                cell = tbl[r, c]
                cell.set_edgecolor('#d4dbe8')
                cell.set_height(0.10)
                if is_avg:
                    cell.set_facecolor(AVG_ROW_BG)                 # green tint for the average row
                    cell.set_text_props(color=AVG_ROW_FG, fontweight='bold')
                elif is_starved:
                    cell.set_facecolor(STARVATION_BG)              # red tint for starved processes
                    cell.set_text_props(color=STARVATION_FG, fontweight='bold')
                elif r % 2 == 0:
                    cell.set_facecolor(ROW_EVEN)                   # alternating row shading
                    cell.set_text_props(color=TEXT_DARK)
                else:
                    cell.set_facecolor(ROW_ODD)
                    cell.set_text_props(color=TEXT_DARK)

        if starved:  # print starvation warning below the table if any process waited too long
            ax_table.text(0.0, -0.01,
                          f'⚠  Starvation detected (waiting > {STARVATION_THRESHOLD}): '
                          + ', '.join(sorted(starved)),
                          transform=ax_table.transAxes,
                          fontsize=8, color=STARVATION_FG, va='top')

        # Metrics strip — one line summarising CPU utilisation, throughput, total time, idle time
        ax_metrics = fig.add_subplot(gs[2])
        ax_metrics.set_facecolor('#EEF2FA')
        ax_metrics.axis('off')
        ax_metrics.text(0.5, 0.5,
                        f'CPU Utilization: {cpu_util:.1f}%   |   '
                        f'Throughput: {throughput:.4f} proc/time   |   '
                        f'Total Time: {total_time}   |   '
                        f'Idle Time: {idle_time}',
                        ha='center', va='center',
                        transform=ax_metrics.transAxes,
                        fontsize=9, color=TEXT_DARK)

        # Gantt chart — segments are split into rows of GANTT_SEGS_PER_ROW to avoid horizontal overflow.
        # Each row wraps underneath the previous one, like text wrapping in a paragraph.
        ax_gantt = fig.add_subplot(gs[3])
        ax_gantt.set_facecolor('#ffffff')
        ax_gantt.axis('off')

        unique_ids = list(dict.fromkeys(pid for pid, _, _ in gantt))
        color_map  = {pid: PALETTE[i % len(PALETTE)]  # assign a consistent colour to each process ID
                      for i, pid in enumerate(unique_ids)}

        SEG_W  = 2.0           # fixed visual width for every bar in data units
        ROW_H  = 0.80          # vertical gap between each wrapped Gantt row
        BAR_H  = ROW_H * 0.68  # height of the bar itself (slightly shorter than the row spacing)

        chunks     = [gantt[i:i + GANTT_SEGS_PER_ROW]  # split gantt into page-width slices
                      for i in range(0, len(gantt), GANTT_SEGS_PER_ROW)]
        total_rows = len(chunks)
        row_width  = GANTT_SEGS_PER_ROW * SEG_W

        ax_gantt.set_xlim(-2.5, row_width + 0.3)
        ax_gantt.set_ylim(-(total_rows * ROW_H + 0.4), ROW_H * 0.9)

        ax_gantt.text(0.0, ROW_H * 0.82, 'Gantt Chart',
                      fontsize=9, fontweight='bold', color=TEXT_DARK,
                      va='top', ha='left')

        for row_idx, chunk in enumerate(chunks):
            y = -(row_idx * ROW_H)  # each row is drawn one ROW_H unit lower than the previous

            ax_gantt.text(-0.3, y, f't={chunk[0][1]}',  # time label at the start of each row
                          fontsize=6.5, color='#5a6a88',
                          ha='right', va='center')

            for seg_idx, (pid, seg_s, seg_e) in enumerate(chunk):
                x0    = seg_idx * SEG_W
                color = IDLE_COLOR if pid == 'Idle' else color_map.get(pid, '#aaa')
                lc    = '#555555' if pid == 'Idle' else 'white'  # label colour contrasts with bar colour

                ax_gantt.barh(y, SEG_W, left=x0,         # draw the coloured bar
                              color=color, edgecolor='white',
                              height=BAR_H, linewidth=0.9)
                ax_gantt.text(x0 + SEG_W / 2, y, str(pid),  # process ID label centred inside the bar
                              ha='center', va='center',
                              fontsize=7, fontweight='bold',
                              color=lc, clip_on=True)
                ax_gantt.plot([x0, x0],                   # vertical tick at the left edge of the bar
                              [y - BAR_H/2, y + BAR_H/2],
                              color='#c8d0dc', linewidth=0.7, zorder=0)
                ax_gantt.text(x0, y - BAR_H/2 - 0.06, str(seg_s),  # start time below the tick
                              fontsize=5.5, color=TEXT_DARK,
                              ha='center', va='top')

            end_x = len(chunk) * SEG_W
            ax_gantt.plot([end_x, end_x],                 # final tick at the right edge of the last bar
                          [y - BAR_H/2, y + BAR_H/2],
                          color='#c8d0dc', linewidth=0.7, zorder=0)
            ax_gantt.text(end_x, y - BAR_H/2 - 0.06, str(chunk[-1][2]),  # end time of this row
                          fontsize=5.5, color=TEXT_DARK,
                          ha='center', va='top')

        # Embed the finished figure inside the scrollable results_inner frame
        canvas = FigureCanvasTkAgg(fig, master=self.results_inner)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

        self.save_btn.config(state='normal')  # unlock the save button now that a figure exists


if __name__ == '__main__':
    root = tk.Tk()
    app  = CPUSchedulerApp(root)
    root.mainloop()