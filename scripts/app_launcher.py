#!/usr/bin/env python3
"""Desktop launcher for video-evaluation mode and live-camera mode."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable or "python3"


class ProcessController:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.running_name: str = ""

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], cwd: Path, name: str) -> None:
        if self.is_running():
            raise RuntimeError("A process is already running.")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self.running_name = name

        def _reader() -> None:
            assert self.proc is not None
            if self.proc.stdout is None:
                return
            for line in self.proc.stdout:
                self.log_queue.put(line.rstrip("\n"))
            code = self.proc.wait()
            self.log_queue.put(f"[{name}] exited with code {code}")

        self.reader_thread = threading.Thread(target=_reader, daemon=True)
        self.reader_thread.start()
        self.log_queue.put(f"[{name}] started: {' '.join(cmd)}")

    def stop(self) -> None:
        if not self.is_running():
            return
        assert self.proc is not None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log_queue.put(f"[{self.running_name}] stopped by user")


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Badminton AI Coach Launcher")
        self.root.geometry("980x760")

        self.proc_ctl = ProcessController()
        self._build_ui()
        self._poll_logs()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            top,
            text="Badminton AI Coach - Dual Mode Launcher (Video Evaluation / Live Camera)",
            font=("Arial", 13, "bold"),
        )
        title.pack(anchor="w", pady=(0, 8))

        self.status_var = tk.StringVar(value="Idle")
        status = ttk.Label(top, textvariable=self.status_var, foreground="#0A58CA")
        status.pack(anchor="w", pady=(0, 8))

        notebook = ttk.Notebook(top)
        notebook.pack(fill=tk.X, expand=False)

        self.video_tab = ttk.Frame(notebook, padding=10)
        self.camera_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.video_tab, text="Video Mode")
        notebook.add(self.camera_tab, text="Camera Mode")

        self._build_video_tab()
        self._build_camera_tab()

        btn_bar = ttk.Frame(top)
        btn_bar.pack(fill=tk.X, pady=(10, 8))
        self.start_video_btn = ttk.Button(btn_bar, text="Start Video Evaluation", command=self.run_video_mode)
        self.start_video_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.start_camera_btn = ttk.Button(btn_bar, text="Start Live Camera", command=self.run_camera_mode)
        self.start_camera_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(btn_bar, text="Stop Current Task", command=self.stop_running)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.clear_log_btn = ttk.Button(btn_bar, text="Clear Logs", command=self.clear_logs)
        self.clear_log_btn.pack(side=tk.LEFT)

        log_label = ttk.Label(top, text="Runtime Logs")
        log_label.pack(anchor="w")
        self.log_text = tk.Text(top, height=22, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _make_browse_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        is_dir: bool = False,
        filetypes: tuple[tuple[str, str], ...] = (("All files", "*.*"),),
    ) -> None:
        ttk.Label(parent, text=label, width=14).grid(row=row, column=0, sticky="w", pady=4)
        ent = ttk.Entry(parent, textvariable=var, width=80)
        ent.grid(row=row, column=1, sticky="ew", padx=(6, 6), pady=4)

        def _browse() -> None:
            if is_dir:
                p = filedialog.askdirectory(initialdir=str(REPO_ROOT))
            else:
                p = filedialog.askopenfilename(initialdir=str(REPO_ROOT), filetypes=filetypes)
            if p:
                var.set(p)

        ttk.Button(parent, text="Browse", command=_browse).grid(row=row, column=2, sticky="e", pady=4)
        parent.grid_columnconfigure(1, weight=1)

    def _build_video_tab(self) -> None:
        self.video_path = tk.StringVar(value=str(REPO_ROOT / "archive" / "demo.mp4"))
        self.video_config = tk.StringVar(value=str(REPO_ROOT / "src" / "config" / "v3_realtime.yaml"))
        self.video_out_dir = tk.StringVar(value=str(REPO_ROOT / "reports" / "demo_friend"))
        self.video_match_name = tk.StringVar(value="demo_friend")
        self.video_no_llm = tk.BooleanVar(value=True)
        self.video_pose = tk.StringVar(value="auto")
        self.video_court_once = tk.StringVar(value="auto")
        self.video_court_idx = tk.StringVar(value="")

        self._make_browse_row(
            self.video_tab,
            0,
            "Video File",
            self.video_path,
            is_dir=False,
            filetypes=(("MP4", "*.mp4"), ("All files", "*.*")),
        )
        self._make_browse_row(self.video_tab, 1, "Config File", self.video_config)
        self._make_browse_row(self.video_tab, 2, "Output Directory", self.video_out_dir, is_dir=True)

        ttk.Label(self.video_tab, text="match-name", width=14).grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(self.video_tab, textvariable=self.video_match_name, width=24).grid(
            row=3, column=1, sticky="w", padx=(6, 6), pady=4
        )

        ttk.Label(self.video_tab, text="pose", width=14).grid(row=4, column=0, sticky="w", pady=4)
        ttk.Combobox(
            self.video_tab,
            textvariable=self.video_pose,
            values=("auto", "on", "off"),
            state="readonly",
            width=10,
        ).grid(row=4, column=1, sticky="w", padx=(6, 6), pady=4)

        ttk.Label(self.video_tab, text="court-detect", width=14).grid(row=5, column=0, sticky="w", pady=4)
        ttk.Combobox(
            self.video_tab,
            textvariable=self.video_court_once,
            values=("auto", "on", "off"),
            state="readonly",
            width=10,
        ).grid(row=5, column=1, sticky="w", padx=(6, 6), pady=4)

        ttk.Label(self.video_tab, text="frame-idx", width=14).grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(self.video_tab, textvariable=self.video_court_idx, width=10).grid(
            row=6, column=1, sticky="w", padx=(6, 6), pady=4
        )

        ttk.Checkbutton(self.video_tab, text="Disable LLM (recommended during debugging)", variable=self.video_no_llm).grid(
            row=7, column=1, sticky="w", padx=(6, 6), pady=4
        )

    def _build_camera_tab(self) -> None:
        self.cam_index = tk.StringVar(value="0")
        self.cam_config = tk.StringVar(value=str(REPO_ROOT / "src" / "config" / "v3_realtime.yaml"))
        self.cam_out_dir = tk.StringVar(value=str(REPO_ROOT / "reports" / "live_camera"))
        self.cam_session = tk.StringVar(value="live_cam")
        self.cam_court_report = tk.StringVar(value="")
        self.cam_lock_attempts = tk.StringVar(value="24")
        self.cam_lock_step = tk.StringVar(value="5")
        self.cam_lock_seconds = tk.StringVar(value="0")
        self.cam_lock_only = tk.BooleanVar(value=False)
        self.cam_save_debug = tk.BooleanVar(value=True)

        self._make_browse_row(self.camera_tab, 0, "Config File", self.cam_config)
        self._make_browse_row(self.camera_tab, 1, "Output Directory", self.cam_out_dir, is_dir=True)
        self._make_browse_row(
            self.camera_tab,
            2,
            "Reuse Court Report",
            self.cam_court_report,
            is_dir=False,
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )

        ttk.Label(self.camera_tab, text="camera-index", width=14).grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(self.camera_tab, textvariable=self.cam_index, width=10).grid(
            row=3, column=1, sticky="w", padx=(6, 6), pady=4
        )
        ttk.Label(self.camera_tab, text="session-name", width=14).grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(self.camera_tab, textvariable=self.cam_session, width=20).grid(
            row=4, column=1, sticky="w", padx=(6, 6), pady=4
        )

        ttk.Label(self.camera_tab, text="lock-attempts", width=14).grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(self.camera_tab, textvariable=self.cam_lock_attempts, width=10).grid(
            row=5, column=1, sticky="w", padx=(6, 6), pady=4
        )
        ttk.Label(self.camera_tab, text="lock-step", width=14).grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(self.camera_tab, textvariable=self.cam_lock_step, width=10).grid(
            row=6, column=1, sticky="w", padx=(6, 6), pady=4
        )
        ttk.Label(self.camera_tab, text="lock-seconds", width=14).grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(self.camera_tab, textvariable=self.cam_lock_seconds, width=10).grid(
            row=7, column=1, sticky="w", padx=(6, 6), pady=4
        )
        ttk.Checkbutton(self.camera_tab, text="Save court debug image", variable=self.cam_save_debug).grid(
            row=8, column=1, sticky="w", padx=(6, 6), pady=4
        )
        ttk.Checkbutton(self.camera_tab, text="Lock court only and write report", variable=self.cam_lock_only).grid(
            row=9, column=1, sticky="w", padx=(6, 6), pady=4
        )

    def append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def clear_logs(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_logs(self) -> None:
        while True:
            try:
                line = self.proc_ctl.log_queue.get_nowait()
            except queue.Empty:
                break
            self.append_log(line)
        self._refresh_status()
        self.root.after(120, self._poll_logs)

    def _refresh_status(self) -> None:
        running = self.proc_ctl.is_running()
        self.status_var.set(f"Running: {self.proc_ctl.running_name}" if running else "Idle")
        self.start_video_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.start_camera_btn.configure(state=tk.DISABLED if running else tk.NORMAL)

    def stop_running(self) -> None:
        self.proc_ctl.stop()

    def run_video_mode(self) -> None:
        video = Path(self.video_path.get()).expanduser()
        if not video.exists():
            messagebox.showerror("Error", f"Video not found: {video}")
            return
        config = Path(self.video_config.get()).expanduser()
        if not config.exists():
            messagebox.showerror("Error", f"Config file not found: {config}")
            return
        out_dir = Path(self.video_out_dir.get()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            PYTHON,
            "scripts/demo_friend.py",
            "--video",
            str(video),
            "--config",
            str(config),
            "--match-name",
            self.video_match_name.get().strip() or "demo_friend",
            "--out-dir",
            str(out_dir),
            "--pose",
            self.video_pose.get().strip() or "auto",
            "--court-detect-once",
            self.video_court_once.get().strip() or "auto",
        ]
        frame_idx = self.video_court_idx.get().strip()
        if frame_idx:
            if not frame_idx.isdigit():
                messagebox.showerror("Error", "frame-idx must be a non-negative integer.")
                return
            cmd.extend(["--court-detect-frame-idx", frame_idx])
        if self.video_no_llm.get():
            cmd.append("--no-llm")

        try:
            self.proc_ctl.start(cmd=cmd, cwd=REPO_ROOT, name="video_mode")
        except Exception as exc:
            messagebox.showerror("Launch Failed", str(exc))

    def run_camera_mode(self) -> None:
        config = Path(self.cam_config.get()).expanduser()
        if not config.exists():
            messagebox.showerror("Error", f"Config file not found: {config}")
            return
        out_dir = Path(self.cam_out_dir.get()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            cam_idx = int(self.cam_index.get().strip())
            lock_attempts = int(self.cam_lock_attempts.get().strip())
            lock_step = int(self.cam_lock_step.get().strip())
            lock_seconds = float(self.cam_lock_seconds.get().strip() or "0")
        except ValueError:
            messagebox.showerror(
                "Error",
                "camera-index / lock-attempts / lock-step must be integers, and lock-seconds must be a number.",
            )
            return

        cmd = [
            PYTHON,
            "scripts/demo_live_camera.py",
            "--camera-index",
            str(cam_idx),
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
            "--session-name",
            self.cam_session.get().strip() or "live_cam",
            "--court-lock-attempts",
            str(max(1, lock_attempts)),
            "--court-lock-step",
            str(max(1, lock_step)),
        ]
        if lock_seconds > 0:
            cmd.extend(["--court-lock-seconds", str(lock_seconds)])
        court_report = self.cam_court_report.get().strip()
        if court_report:
            cmd.extend(["--court-corners-report", court_report])
        if self.cam_lock_only.get():
            cmd.append("--lock-only")
        if self.cam_save_debug.get():
            cmd.append("--save-court-debug")

        try:
            self.proc_ctl.start(cmd=cmd, cwd=REPO_ROOT, name="camera_mode")
        except Exception as exc:
            messagebox.showerror("Launch Failed", str(exc))


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
