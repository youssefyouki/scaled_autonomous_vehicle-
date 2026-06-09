#!/usr/bin/env python3
"""
Controller Selection Panel
===========================
Tkinter GUI + ROS2 node.

Window layout:
  ┌─────────────────────────────────────────┐
  │ F1TENTH Controller Panel                 │
  │  ○ Pure Pursuit  ○ Stanley  ○ Linderoth  │
  │  [▶ Start]  [■ Stop]   ● STOPPED        │
  │─────────────────────────────────────────│
  │ Parameters — <selected controller>       │
  │  Speed (m/s)  0.50  [━━━●━━━━━━━━━]    │
  │  ...                                     │
  │─────────────────────────────────────────│
  │ Perception:  CTE: +0.000 m  HDG: +0.0°  │
  └─────────────────────────────────────────┘

The panel starts/stops controller subprocesses via `ros2 run` and pushes
parameter changes to the running node through the ROS2 SetParameters service.
"""

import math
import os
import queue
import signal
import subprocess
import threading
import tkinter as tk
import yaml

import rclpy
from rcl_interfaces.msg import Parameter as RosParam
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry

# ── Controller registry ────────────────────────────────────────────────────────
# Each entry: (param_name, default, min, max, resolution, display_label)

CONTROLLERS = {
    'Pure Pursuit': {
        'executable': 'pure_pursuit_node.py',
        'node_name':  'pure_pursuit_controller',
        'params': [
            # (name, default, min, max, resolution, label)
            ('target_speed',   0.50, 0.10, 1.50, 0.01, 'Speed         (m/s)'),
            ('lookahead_dist', 0.30, 0.05, 1.00, 0.01, 'Lookahead     (m)'),
            ('alpha',          0.50, 0.00, 1.00, 0.05, 'alpha         EMA'),
        ],
    },
    'Stanley': {
        'executable': 'stanley_control_node.py',
        'node_name':  'stanley_controller',
        'params': [
            ('target_speed', 0.50, 0.10, 1.50, 0.01, 'Speed         (m/s)'),
            ('k',            1.00, 0.10, 5.00, 0.10, 'k             crosstrack gain'),
            ('k_soft',       0.40, 0.01, 2.00, 0.01, 'k_soft        low-speed softening'),
            ('alpha',        0.50, 0.00, 1.00, 0.05, 'alpha         EMA'),
        ],
    },
    'Linderoth': {
        'executable': 'Linderoth_control_node.py',
        'node_name':  'linderoth_controller',
        'params': [
            ('target_speed',   0.35, 0.10, 1.50, 0.01, 'Speed         (m/s)'),
            ('k1',             0.30, 0.01, 2.00, 0.01, 'k1            stability gain'),
            ('k2',             0.60, 0.01, 2.00, 0.01, 'k2            crosstrack gain'),
            ('heading_scale',  0.00, 0.00, 1.00, 0.05, 'hdg_scale     0=crosstrack 1=full'),
            ('alpha',          0.40, 0.00, 1.00, 0.05, 'alpha         EMA'),
        ],
    },
}

# ── Colours (Catppuccin Mocha) ─────────────────────────────────────────────────
BG     = '#1e1e2e'
PANEL  = '#313244'
FG     = '#cdd6f4'
ACCENT = '#89b4fa'
GREEN  = '#a6e3a1'
RED    = '#f38ba8'
YELLOW = '#f9e2af'
TROUGH = '#181825'


# ── ROS2 side ──────────────────────────────────────────────────────────────────

class PanelNode(Node):
    """ROS2 node side of the panel."""

    def __init__(self):
        super().__init__('controller_panel')
        self.cte = 0.0
        self.hdg = 0.0
        self._current_pose = Pose()

        self.create_subscription(Float32, '/perception/crosstrack_error',
                                 lambda m: setattr(self, 'cte', m.data), 10)
        self.create_subscription(Float32, '/perception/heading_error',
                                 lambda m: setattr(self, 'hdg', m.data), 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Pre-create one SetParameters client per controller — all done in
        # __init__ (executor thread) so rclpy node ops stay thread-safe.
        self._param_clients = {
            info['node_name']: self.create_client(
                SetParameters, f'/{info["node_name"]}/set_parameters')
            for info in CONTROLLERS.values()
        }

        # Queue lets the tkinter thread post param updates; the 20 Hz timer
        # drains it inside the executor so all ROS calls stay on one thread.
        self._param_queue: queue.Queue = queue.Queue()

        # When True the tick loop continuously publishes zero cmd_vel.
        # Starts True so the car is held still before any controller is started.
        self._holding_stop = True
        self.create_timer(0.05, self._tick)   # 20 Hz

    # ── Timer tick (runs inside rclpy executor) ───────────────────────────────

    def _tick(self):
        # 1. Drain parameter update queue
        while not self._param_queue.empty():
            try:
                node_name, param_name, value = self._param_queue.get_nowait()
            except queue.Empty:
                break
            client = self._param_clients.get(node_name)
            if client and client.service_is_ready():
                pv = ParameterValue()
                pv.type = ParameterType.PARAMETER_DOUBLE
                pv.double_value = float(value)
                req = SetParameters.Request()
                req.parameters = [RosParam(name=param_name, value=pv)]
                client.call_async(req)

        # 2. Continuous zero hold while no controller is running
        if self._holding_stop:
            self._cmd_pub.publish(Twist())

    # ── Public API (safe to call from tkinter thread) ─────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._current_pose = msg.pose.pose

    def set_param(self, node_name: str, param_name: str, value: float):
        """Thread-safe: enqueue a parameter update for the executor to send."""
        self._param_queue.put((node_name, param_name, value))

    def controller_started(self):
        """Call when a controller subprocess is launched — releases the zero hold."""
        self._holding_stop = False

    def publish_stop(self):
        """Re-engage the continuous zero hold."""
        self._holding_stop = True


# ── GUI side ───────────────────────────────────────────────────────────────────

class ControllerPanel:

    def __init__(self, ros_node: PanelNode):
        self.node  = ros_node
        self.proc: subprocess.Popen | None = None
        self.active_ctrl: str | None       = None

        self.root = tk.Tk()
        self.root.title('F1TENTH Controller Panel')
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        self._slider_vars: dict[str, dict[str, tk.DoubleVar]] = {}
        self._param_frames: dict[str, tk.Frame]               = {}

        self._build()
        self._refresh()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _lbl(self, parent, text, **kw):
        kw.setdefault('bg', BG)
        kw.setdefault('fg', FG)
        kw.setdefault('font', ('Courier', 10))
        return tk.Label(parent, text=text, **kw)

    def _sep(self, parent):
        tk.Frame(parent, bg=PANEL, height=1).pack(fill='x', padx=6, pady=2)

    def _build(self):
        W = 540

        # ── Title ──────────────────────────────────────────────────────────────
        tk.Label(self.root, text='  F1TENTH Controller Panel',
                 bg=BG, fg=ACCENT, font=('Courier', 13, 'bold'),
                 anchor='w').pack(fill='x', padx=10, pady=(10, 2))
        self._sep(self.root)

        # ── Controller selection ───────────────────────────────────────────────
        sel = tk.Frame(self.root, bg=PANEL)
        sel.pack(fill='x', padx=8, pady=4)

        self._lbl(sel, 'Controller:', bg=PANEL, fg=ACCENT,
                  font=('Courier', 10, 'bold')).pack(anchor='w', padx=8, pady=(6, 2))

        self.ctrl_var = tk.StringVar(value=list(CONTROLLERS)[0])
        radio_row = tk.Frame(sel, bg=PANEL)
        radio_row.pack(anchor='w', padx=8, pady=2)
        for name in CONTROLLERS:
            tk.Radiobutton(
                radio_row, text=name, variable=self.ctrl_var, value=name,
                bg=PANEL, fg=FG, selectcolor=BG,
                activebackground=PANEL, activeforeground=ACCENT,
                font=('Courier', 10),
                command=self._on_ctrl_select,
            ).pack(side='left', padx=8)

        btn_row = tk.Frame(sel, bg=PANEL)
        btn_row.pack(anchor='w', padx=8, pady=(4, 8))

        tk.Button(
            btn_row, text='▶  Start', command=self._start,
            bg='#40a02b', fg='white', font=('Courier', 10, 'bold'),
            relief='flat', padx=10, pady=2,
        ).pack(side='left', padx=(0, 6))

        tk.Button(
            btn_row, text='■  Stop', command=self._stop,
            bg='#d20f39', fg='white', font=('Courier', 10, 'bold'),
            relief='flat', padx=10, pady=2,
        ).pack(side='left', padx=(0, 6))

        tk.Button(
            btn_row, text='💾  Save Config', command=self._save_config,
            bg='#7287fd', fg='white', font=('Courier', 10, 'bold'),
            relief='flat', padx=10, pady=2,
        ).pack(side='left', padx=(0, 12))

        self.status_lbl = tk.Label(
            btn_row, text='● STOPPED', fg=RED, bg=PANEL,
            font=('Courier', 10, 'bold'))
        self.status_lbl.pack(side='left')

        self._sep(self.root)

        # ── Parameter sliders ─────────────────────────────────────────────────
        params_container = tk.Frame(self.root, bg=BG)
        params_container.pack(fill='x', padx=8, pady=4)

        for ctrl_name, info in CONTROLLERS.items():
            frame = tk.Frame(params_container, bg=PANEL)

            self._lbl(frame, f'  Parameters — {ctrl_name}',
                      bg=PANEL, fg=ACCENT,
                      font=('Courier', 10, 'bold')).grid(
                row=0, column=0, columnspan=3, sticky='w', padx=4, pady=(6, 2))

            self._slider_vars[ctrl_name] = {}
            for r, (pname, default, pmin, pmax, res, plabel) in \
                    enumerate(info['params'], start=1):

                var = tk.DoubleVar(value=default)
                self._slider_vars[ctrl_name][pname] = var

                self._lbl(frame, f'{plabel:<30}',
                          bg=PANEL, fg=FG, font=('Courier', 9),
                          anchor='w').grid(row=r, column=0, sticky='w', padx=8)

                val_lbl = tk.Label(
                    frame, text=f'{default:>5.2f}', width=6,
                    bg=PANEL, fg=GREEN, font=('Courier', 9, 'bold'))
                val_lbl.grid(row=r, column=1, padx=6)

                tk.Scale(
                    frame, from_=pmin, to=pmax, resolution=res,
                    orient='horizontal', variable=var, length=260,
                    bg=PANEL, fg=FG, troughcolor=TROUGH,
                    highlightthickness=0, showvalue=False,
                    command=lambda v, cn=ctrl_name, pn=pname, lbl=val_lbl:
                        self._on_slider(cn, pn, float(v), lbl),
                ).grid(row=r, column=2, padx=(0, 10), pady=2)

            frame.grid_columnconfigure(2, weight=1)
            self._param_frames[ctrl_name] = frame

        self._on_ctrl_select()   # show the default controller's params
        self._load_config()      # restore saved values if file exists

        self._sep(self.root)

        # ── Perception readout ─────────────────────────────────────────────────
        perc = tk.Frame(self.root, bg=PANEL)
        perc.pack(fill='x', padx=8, pady=(4, 10))

        self._lbl(perc, '  Perception', bg=PANEL, fg=ACCENT,
                  font=('Courier', 10, 'bold')).pack(anchor='w', padx=4, pady=(6, 2))

        row = tk.Frame(perc, bg=PANEL)
        row.pack(anchor='w', padx=8, pady=(0, 8))

        self._lbl(row, 'CTE:', bg=PANEL).pack(side='left')
        self.cte_lbl = tk.Label(row, text=' +0.000 m', width=10,
                                bg=BG, fg=GREEN, font=('Courier', 10, 'bold'))
        self.cte_lbl.pack(side='left', padx=(4, 16))

        self._lbl(row, 'Heading:', bg=PANEL).pack(side='left')
        self.hdg_lbl = tk.Label(row, text='  +0.0 °', width=10,
                                bg=BG, fg=GREEN, font=('Courier', 10, 'bold'))
        self.hdg_lbl.pack(side='left', padx=4)

    # ── Config save / load ────────────────────────────────────────────────────

    CONFIG_FILE = os.path.expanduser('~/.ros/f1tenth_controller_params.yaml')

    def _save_config(self):
        config = {
            ctrl: {pname: var.get() for pname, var in sliders.items()}
            for ctrl, sliders in self._slider_vars.items()
        }
        os.makedirs(os.path.dirname(self.CONFIG_FILE), exist_ok=True)
        with open(self.CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)
        self.status_lbl.config(text='✔ Config saved', fg=GREEN)
        self.root.after(2000, self._restore_status_label)

    def _load_config(self):
        if not os.path.exists(self.CONFIG_FILE):
            return
        try:
            with open(self.CONFIG_FILE) as f:
                config = yaml.safe_load(f) or {}
            for ctrl, params in config.items():
                if ctrl not in self._slider_vars:
                    continue
                for pname, value in params.items():
                    if pname in self._slider_vars[ctrl]:
                        self._slider_vars[ctrl][pname].set(float(value))
        except Exception:
            pass  # corrupt file — silently ignore

    def _restore_status_label(self):
        if self.active_ctrl:
            self.status_lbl.config(text=f'● {self.active_ctrl}', fg=GREEN)
        else:
            self.status_lbl.config(text='● STOPPED', fg=RED)

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_ctrl_select(self):
        selected = self.ctrl_var.get()
        for name, frame in self._param_frames.items():
            if name == selected:
                frame.pack(fill='x')
            else:
                frame.pack_forget()

    def _on_slider(self, ctrl_name: str, param_name: str,
                   value: float, val_lbl: tk.Label):
        val_lbl.config(text=f'{value:>5.2f}')
        if self.active_ctrl == ctrl_name:
            node_name = CONTROLLERS[ctrl_name]['node_name']
            self.node.set_param(node_name, param_name, value)

    def _start(self):
        self._stop()
        name = self.ctrl_var.get()
        exe  = CONTROLLERS[name]['executable']
        # start_new_session gives the child its own process group so we can
        # kill ros2-run AND the Python node it spawns in one shot
        self.proc = subprocess.Popen(
            ['ros2', 'run', 'my_f1tenth_sim', exe],
            env=os.environ.copy(),
            start_new_session=True,
        )
        self.node.controller_started()   # release zero hold so controller drives
        self.active_ctrl = name
        self.status_lbl.config(text=f'● {name}', fg=GREEN)

    def _kill_proc(self, proc: subprocess.Popen):
        """Kill every process in the subprocess's session group."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        threading.Thread(target=proc.wait, daemon=True).start()

    def _stop(self):
        # Publish zeros immediately from this thread — rclpy publishers are thread-safe.
        # Do this before killing the process so zeros arrive before the controller's
        # last message drains from the DDS queue.
        self.node._cmd_pub.publish(Twist())
        self.node.publish_stop()        # engage continuous zero hold
        if self.proc is not None:
            proc = self.proc
            self.proc = None
            self._kill_proc(proc)
        self.active_ctrl = None
        self.status_lbl.config(text='● STOPPED', fg=RED)

    def _on_close(self):
        self._stop()
        self.root.destroy()

    # ── Periodic refresh (10 Hz) ──────────────────────────────────────────────

    def _refresh(self):
        self.cte_lbl.config(text=f'{self.node.cte:+.3f} m')
        self.hdg_lbl.config(text=f'{math.degrees(self.node.hdg):+.1f} °')

        # Backup zero hold: also publish from the tkinter thread at 10 Hz so
        # the car stays still even if the rclpy spin thread is momentarily busy.
        if self.active_ctrl is None:
            self.node._cmd_pub.publish(Twist())

        # detect subprocess crash
        if self.proc is not None and self.proc.poll() is not None:
            proc             = self.proc
            self.proc        = None
            self.active_ctrl = None
            self._kill_proc(proc)
            self.node.publish_stop()
            self.status_lbl.config(text='● CRASHED', fg=YELLOW)

        self.root.after(100, self._refresh)

    def run(self):
        self.root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

def _kill_stale_controllers():
    """Kill any leftover controller processes from a previous session."""
    for exe in ['pure_pursuit_node.py', 'stanley_control_node.py',
                'Linderoth_control_node.py']:
        subprocess.run(['pkill', '-9', '-f', exe], capture_output=True)


def main(args=None):
    _kill_stale_controllers()
    rclpy.init(args=args)
    ros_node = PanelNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    gui = ControllerPanel(ros_node)
    gui.run()

    if rclpy.ok():
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
