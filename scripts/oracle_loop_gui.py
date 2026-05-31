"""Tkinter GUI for the batch-delivery oracle-loop.

Launches the oracle-loop as a subprocess, streams its log into a scrolled
text widget, shows an iteration progress bar, and offers a *graceful* stop
button.  Stop writes a ``STOP_REQUESTED`` sentinel into the run directory;
the loop checks this sentinel between iterations and exits after the
current iteration finishes saving (training_matrix.csv, holdout_extreme.csv,
ml_cost_predictor.pkl, oracle_loop_history.csv, final report).

Usage:
    python scripts/oracle_loop_gui.py

The defaults resume the existing run under
``results/oracle_loop_overnight_2026_05_21`` and target a doubled pool size.
Adjust the fields before pressing *Start*.

Closing the window after a graceful stop is safe; closing during an
in-progress iteration triggers a confirmation dialog and SIGTERM.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    import psutil
except ImportError:  # pragma: no cover — psutil is an indirect dep
    psutil = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover — Pillow ships with the project
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]

if sys.platform == "win32":
    import winsound
else:  # pragma: no cover — winsound is Windows only
    winsound = None  # type: ignore[assignment]

import datetime as _dt

ROOT = Path(__file__).resolve().parents[1]
BASE_TITLE = "batch-delivery — Oracle Loop Control"
SETTINGS_PATH = Path.home() / ".batch_delivery_gui.json"
VROOM_ACCESS_LOG = ROOT / "vroom" / "access.log"
VROOM_HEALTH_URL = "http://localhost:3000/health"
VALHALLA_HEALTH_URL = "http://localhost:8002/status"

# Matches the access-log POST lines logged for every VROOM solve.  Health
# checks (GET /health) are excluded so the routing counter is not biased.
_POST_RE = re.compile(r'"POST\s+/\s+HTTP/1\.\d"')
_SWEEP_START_RE = re.compile(r"'total_combinations':\s*(\d+)")
_WEEKDAY_DE = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")


def _safe_pct(s: str) -> float:
    """Parse '12.34%' → 12.34; tolerates spaces and missing percent sign."""
    try:
        return float(s.replace("%", "").strip())
    except (ValueError, AttributeError):
        return 0.0

DEFAULTS = {
    "out_dir":       str(ROOT / "results" / "oracle_loop_extended_2026_05_22"),
    "training_csv":  str(ROOT / "results" / "oracle_loop_extended_2026_05_22" / "training_matrix.csv"),
    "initial_model": str(ROOT / "results" / "oracle_loop_overnight_2026_05_21" / "ml_cost_predictor.pkl"),
    "iterations":    "40",
    "seeds_per_iter":"4",
    "samples_per_iter":"1800",
    "base_seed":     "9000",
    "max_runtime_min":"1700",   # ~28 h budget
    "mape_target":   "0.5",
    "stability_threshold":"0.05",
    "stability_window":"5",
    "hot_regions":   "200",
    "variance_growth":"0.25",
    "variance_max_extra":"2",
    "extreme_holdout_frac":"0.2",
    "jobs":          "8",
    "arch":          "128,64,32",
    "alpha":         "1e-2",
    "sweep_config":  str(ROOT / "conf" / "sweep_multi_provider.yaml"),
    "resume_pool":   str(ROOT / "results" / "oracle_loop_overnight_2026_05_21" / "training_matrix.csv"),
}

# Presets only override the fields listed; everything else stays at DEFAULTS.
PRESETS: dict[str, dict[str, str]] = {
    "Custom": {},
    "Smoke (5 iter, 300 samples)": {
        "iterations": "5",
        "samples_per_iter": "300",
        "max_runtime_min": "60",
        "base_seed": "1000",
        "out_dir": str(ROOT / "results" / "oracle_loop_smoke"),
        "training_csv": str(ROOT / "results" / "oracle_loop_smoke" / "training_matrix.csv"),
        "resume_pool": "",
    },
    "Overnight (40 iter, 850 min)": {
        "iterations": "40",
        "samples_per_iter": "1500",
        "max_runtime_min": "850",
        "base_seed": "6000",
    },
    "Extended overnight (40 iter, 1700 min)": {
        "iterations": "40",
        "samples_per_iter": "1800",
        "max_runtime_min": "1700",
        "base_seed": "9000",
    },
}


class OracleLoopGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(BASE_TITLE)
        self.geometry("1100x880")
        self.proc: subprocess.Popen | None = None
        self.log_q: queue.Queue[str] = queue.Queue()
        self.cur_iter = 0
        self.tot_iter = int(DEFAULTS["iterations"])
        self._stop_pressed = False
        self.run_started_at: float | None = None

        # Per-iteration routing tracking (driven by sweep_start + access.log)
        self.routing_total = 0          # combinations expected this iter
        self.routing_done = 0           # POSTs to VROOM since sweep_start
        self.routing_t_start: float | None = None
        self._access_offset = 0         # incremental tail position of access.log

        # Iter-level history for global ETA: list of (iter_num, wall_time_s)
        self.iter_starts: list[tuple[int, float]] = []
        self.iter_durations_s: list[float] = []

        # Service health: True / False / None (unknown)
        self.vroom_up: bool | None = None
        self.valhalla_up: bool | None = None

        # Wall-clock ETA + window-title state.
        self.last_total_eta_s: float | None = None

        # Learning-curve image cache (avoid reloading when mtime unchanged).
        self._curve_mtime: float = 0.0
        self._curve_image_ref: object | None = None  # keep Tk PhotoImage alive

        # Log filter checkbox state (Tk vars created later in _build_widgets).
        self.var_hide_urllib_debug: tk.BooleanVar | None = None
        self.var_hide_sklearn_warn: tk.BooleanVar | None = None

        # Finish-notification: ring bell + flash once when subprocess exits.
        self._finish_announced = False

        # Resource monitoring (docker stats is slow → background thread).
        self._docker_stats_q: queue.Queue[dict[str, dict]] = queue.Queue(maxsize=4)
        self._docker_stats_stop = threading.Event()
        self._docker_available = shutil.which("docker") is not None
        if self._docker_available:
            threading.Thread(
                target=self._docker_stats_worker, daemon=True
            ).start()
        # Prime psutil's per-core deltas so the first sample is meaningful.
        if psutil is not None:
            psutil.cpu_percent(interval=None)

        # Persisted settings prefill DEFAULTS where applicable.
        self._settings = self._load_settings()
        geom = self._settings.get("_window_geometry")
        if isinstance(geom, str) and re.match(r"\d+x\d+(\+-?\d+\+-?\d+)?$", geom):
            self.geometry(geom)

        self._build_widgets()
        self.after(120, self._drain_log)
        self.after(2000, self._refresh_status)
        self.after(500, self._refresh_routing)
        self.after(1000, self._check_services)
        self.after(1500, self._refresh_resources)
        self.after(2500, self._refresh_learning_curve)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────
    def _build_widgets(self) -> None:
        # Preset selector
        presetrow = ttk.Frame(self)
        presetrow.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(presetrow, text="Preset:").pack(side="left")
        self.var_preset = tk.StringVar(value="Custom")
        self.cmb_preset = ttk.Combobox(
            presetrow, textvariable=self.var_preset,
            values=list(PRESETS.keys()), state="readonly", width=40,
        )
        self.cmb_preset.pack(side="left", padx=6)
        self.cmb_preset.bind("<<ComboboxSelected>>", self._on_preset_change)

        # top: config grid
        top = ttk.LabelFrame(self, text="Oracle-Loop Konfiguration")
        top.pack(fill="x", padx=8, pady=6)

        self.entries: dict[str, tk.Entry] = {}
        fields = [
            ("out_dir", "Out-Verzeichnis"),
            ("training_csv", "Training-CSV (Output)"),
            ("resume_pool", "Resume von Pool (Quelle)"),
            ("initial_model", "Resume von Modell"),
            ("sweep_config", "Sweep-YAML"),
            ("iterations", "Iterationen"),
            ("samples_per_iter", "Samples / Iter"),
            ("seeds_per_iter", "Seeds / Iter"),
            ("base_seed", "Base seed"),
            ("max_runtime_min", "Budget (min)"),
            ("mape_target", "MAPE-Ziel (%)"),
            ("stability_threshold", "Stabilität (%)"),
            ("stability_window", "Stabilität-Fenster"),
            ("hot_regions", "Hot regions"),
            ("variance_growth", "Variance growth"),
            ("variance_max_extra", "Variance max extra"),
            ("extreme_holdout_frac", "Extreme holdout-Anteil"),
            ("jobs", "Parallel jobs"),
            ("arch", "MLP arch"),
            ("alpha", "MLP alpha"),
        ]
        for i, (key, label) in enumerate(fields):
            r, c = i // 2, (i % 2) * 2
            ttk.Label(top, text=label).grid(row=r, column=c, sticky="w", padx=6, pady=2)
            e = ttk.Entry(top, width=58)
            e.insert(0, self._settings.get(key, DEFAULTS[key]))
            e.grid(row=r, column=c + 1, sticky="ew", padx=6, pady=2)
            self.entries[key] = e
        for c in (1, 3):
            top.grid_columnconfigure(c, weight=1)

        # buttons
        btnrow = ttk.Frame(self)
        btnrow.pack(fill="x", padx=8, pady=4)
        self.btn_start = ttk.Button(btnrow, text="▶ Start", command=self._on_start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(btnrow, text="■ Stop nach aktueller Iteration",
                                    command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.btn_kill = ttk.Button(btnrow, text="✖ Hart abbrechen", command=self._on_kill,
                                    state="disabled")
        self.btn_kill.pack(side="left", padx=4)
        ttk.Button(btnrow, text="Out-Verzeichnis öffnen",
                   command=self._open_outdir).pack(side="left", padx=4)
        ttk.Button(btnrow, text="Log öffnen",
                   command=self._open_logfile).pack(side="left", padx=4)
        ttk.Button(btnrow, text="Lernkurve öffnen",
                   command=self._open_learning_curve).pack(side="left", padx=4)

        # status
        statusf = ttk.LabelFrame(self, text="Status")
        statusf.pack(fill="x", padx=8, pady=4)

        # Top row: status label + service indicators
        statusrow = ttk.Frame(statusf)
        statusrow.pack(fill="x", padx=6, pady=2)
        self.lbl_status = tk.StringVar(value="bereit.")
        ttk.Label(statusrow, textvariable=self.lbl_status,
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        self.lbl_vroom = tk.StringVar(value="VROOM: ?")
        self.lbl_valhalla = tk.StringVar(value="Valhalla: ?")
        ttk.Label(statusrow, textvariable=self.lbl_valhalla,
                  font=("Segoe UI", 9)).pack(side="right", padx=8)
        ttk.Label(statusrow, textvariable=self.lbl_vroom,
                  font=("Segoe UI", 9)).pack(side="right", padx=8)

        # Iteration progress
        ttk.Label(statusf, text="Iterationen").pack(anchor="w", padx=6, pady=(4, 0))
        self.progress = ttk.Progressbar(statusf, mode="determinate",
                                         maximum=self.tot_iter)
        self.progress.pack(fill="x", padx=6, pady=(0, 2))
        self.lbl_iter_caption = tk.StringVar(value="Iter 0 / 0")
        ttk.Label(statusf, textvariable=self.lbl_iter_caption,
                  font=("Segoe UI", 9)).pack(anchor="w", padx=6)

        # Routing (within-iteration) progress
        ttk.Label(statusf, text="VROOM-Routing (aktuelle Iter)").pack(
            anchor="w", padx=6, pady=(6, 0))
        self.routing_bar = ttk.Progressbar(statusf, mode="determinate", maximum=1)
        self.routing_bar.pack(fill="x", padx=6, pady=(0, 2))
        self.lbl_routing_caption = tk.StringVar(value="–")
        ttk.Label(statusf, textvariable=self.lbl_routing_caption,
                  font=("Segoe UI", 9)).pack(anchor="w", padx=6)

        # Resource panel: system + docker containers + loop subprocess
        resf = ttk.LabelFrame(statusf, text="Ressourcen")
        resf.pack(fill="x", padx=6, pady=(6, 2))
        self.lbl_res_system = tk.StringVar(value="System: –")
        self.lbl_res_vroom = tk.StringVar(value="VROOM: –")
        self.lbl_res_valhalla = tk.StringVar(value="Valhalla: –")
        self.lbl_res_loop = tk.StringVar(value="Loop-Prozess: –")
        for i, var in enumerate([self.lbl_res_system, self.lbl_res_vroom,
                                  self.lbl_res_valhalla, self.lbl_res_loop]):
            r, c = i // 2, i % 2
            ttk.Label(resf, textvariable=var, width=50,
                      font=("Consolas", 9)).grid(row=r, column=c, sticky="w",
                                                  padx=4, pady=1)

        kpis = ttk.Frame(statusf)
        kpis.pack(fill="x", padx=6, pady=4)
        self.lbl_pool = tk.StringVar(value="Pool: –")
        self.lbl_hold = tk.StringVar(value="Holdout: –")
        self.lbl_mape = tk.StringVar(value="val MAPE: –")
        self.lbl_holdout_mape = tk.StringVar(value="holdout MAPE: –")
        self.lbl_elapsed = tk.StringVar(value="Laufzeit: 00:00")
        self.lbl_iter_eta = tk.StringVar(value="Iter-ETA: –")
        self.lbl_total_eta = tk.StringVar(value="Loop-ETA: –")
        kpi_vars = [self.lbl_pool, self.lbl_hold, self.lbl_mape,
                    self.lbl_holdout_mape, self.lbl_elapsed,
                    self.lbl_iter_eta, self.lbl_total_eta]
        for i, var in enumerate(kpi_vars):
            r, c = i // 4, i % 4
            ttk.Label(kpis, textvariable=var, width=26).grid(
                row=r, column=c, sticky="w", padx=2, pady=1)

        # Learning-curve thumbnail (refreshed from learning_curve.png).
        self.lbl_curve = tk.Label(statusf, bg="#000", anchor="center",
                                   text="(Lernkurve erscheint nach Iter 1)",
                                   fg="#666")
        self.lbl_curve.pack(fill="x", padx=6, pady=(2, 6))

        # log
        logf = ttk.LabelFrame(self, text="Live Log")
        logf.pack(fill="both", expand=True, padx=8, pady=6)

        # filter row above the text widget
        filterrow = ttk.Frame(logf)
        filterrow.pack(fill="x", padx=4, pady=(2, 0))
        self.var_hide_urllib_debug = tk.BooleanVar(
            value=self._settings.get("_hide_urllib_debug", True))
        self.var_hide_sklearn_warn = tk.BooleanVar(
            value=self._settings.get("_hide_sklearn_warn", True))
        ttk.Checkbutton(filterrow, text="urllib3 DEBUG verstecken",
                        variable=self.var_hide_urllib_debug).pack(side="left", padx=4)
        ttk.Checkbutton(filterrow, text="sklearn-Warnings verstecken",
                        variable=self.var_hide_sklearn_warn).pack(side="left", padx=4)
        ttk.Button(filterrow, text="Log leeren",
                   command=self._clear_log).pack(side="right", padx=4)

        self.log = scrolledtext.ScrolledText(logf, wrap="word", font=("Consolas", 9),
                                              bg="#111", fg="#ddd",
                                              insertbackground="#ddd")
        self.log.pack(fill="both", expand=True)
        self.log.tag_config("iter", foreground="#7fd")
        self.log.tag_config("warn", foreground="#fc6")
        self.log.tag_config("err",  foreground="#f88")
        self.log.tag_config("ok",   foreground="#8f8")

    # ── actions ───────────────────────────────────────────────────────
    def _on_start(self) -> None:
        if self.proc is not None:
            messagebox.showinfo("läuft schon", "Eine Loop läuft bereits.")
            return

        # Pre-flight: services must be up — VROOM is mandatory, Valhalla is
        # accessed indirectly through VROOM but we still surface the check.
        self._check_services()
        if not self.vroom_up:
            if not messagebox.askyesno(
                "VROOM nicht erreichbar",
                "VROOM auf localhost:3000 antwortet nicht.\n\n"
                "Ohne VROOM macht der Loop nichts Sinnvolles. "
                "Soll ich versuchen, die Docker-Services zu starten?",
            ):
                return
            self._log("$ docker compose up -d\n", "iter")
            try:
                subprocess.run(
                    ["docker", "compose", "up", "-d"],
                    cwd=str(ROOT), check=False, timeout=60,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                messagebox.showerror("docker fehlt", f"{exc}")
                return
            time.sleep(2)
            self._check_services()
            if not self.vroom_up:
                messagebox.showerror(
                    "VROOM weiterhin down",
                    "Services konnten nicht hochgefahren werden. "
                    "Bitte manuell prüfen.",
                )
                return

        # Verify the CLI is on PATH so we fail fast instead of opening a
        # subprocess that immediately exits.
        try:
            subprocess.run(
                ["batch-delivery", "--help"],
                capture_output=True, check=True, timeout=15,
            )
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            messagebox.showerror(
                "batch-delivery CLI fehlt",
                "Konnte `batch-delivery --help` nicht ausführen. "
                "Editable-Install vergessen?\n\n"
                f"Fehler: {exc}\n\n"
                "Lösung: python -m pip install -e \".[dev]\"",
            )
            return

        out_dir = Path(self.entries["out_dir"].get())
        out_dir.mkdir(parents=True, exist_ok=True)

        # Orphan / collision check: if a previous run wrote to this directory
        # within the last 5 minutes, warn before clobbering its state.
        prev_log = out_dir / "oracle_loop.log"
        if prev_log.exists():
            age_min = (time.time() - prev_log.stat().st_mtime) / 60.0
            if age_min < 5:
                if not messagebox.askyesno(
                    "Out-Verzeichnis frisch beschrieben",
                    f"{prev_log} wurde vor {age_min:.1f} min geschrieben.\n\n"
                    "Läuft eventuell noch ein anderer oracle-loop im "
                    "Hintergrund? Trotzdem starten?",
                ):
                    return

        # If resuming from a prior pool, copy it into the new out_dir so the
        # CLI starts with that file (it appends in-place).
        resume_src = Path(self.entries["resume_pool"].get())
        training_dst = Path(self.entries["training_csv"].get())
        if (resume_src.exists() and resume_src != training_dst
                and not training_dst.exists()):
            training_dst.parent.mkdir(parents=True, exist_ok=True)
            training_dst.write_bytes(resume_src.read_bytes())
            self._log(f"  copied resume pool -> {training_dst}\n", "ok")
            # also copy holdout if present so it's not lost
            resume_hold = resume_src.parent / "holdout_extreme.csv"
            if resume_hold.exists():
                (training_dst.parent / "holdout_extreme.csv").write_bytes(
                    resume_hold.read_bytes())
                self._log("  copied holdout_extreme.csv into new out_dir\n", "ok")

        # purge stale STOP sentinel from a prior run
        (out_dir / "STOP_REQUESTED").unlink(missing_ok=True)

        cmd = [
            "batch-delivery", "oracle-loop",
            "--sweep-config", self.entries["sweep_config"].get(),
            "--iterations", self.entries["iterations"].get(),
            "--seeds-per-iter", self.entries["seeds_per_iter"].get(),
            "--samples-per-iter", self.entries["samples_per_iter"].get(),
            "--base-seed", self.entries["base_seed"].get(),
            "--out-dir", str(out_dir),
            "--training-csv", self.entries["training_csv"].get(),
            "--initial-model", self.entries["initial_model"].get(),
            "--max-runtime-min", self.entries["max_runtime_min"].get(),
            "--mape-target", self.entries["mape_target"].get(),
            "--stability-threshold", self.entries["stability_threshold"].get(),
            "--stability-window", self.entries["stability_window"].get(),
            "--hot-regions", self.entries["hot_regions"].get(),
            "--variance-growth", self.entries["variance_growth"].get(),
            "--variance-max-extra", self.entries["variance_max_extra"].get(),
            "--extreme-holdout-frac", self.entries["extreme_holdout_frac"].get(),
            "--jobs", self.entries["jobs"].get(),
            "--arch", self.entries["arch"].get(),
            "--alpha", self.entries["alpha"].get(),
            "--all-providers", "--all-days",
        ]
        self.tot_iter = int(self.entries["iterations"].get())
        self.progress.configure(maximum=self.tot_iter, value=0)
        self.cur_iter = 0
        # Reset routing/iter timing state for the fresh run.
        self.routing_total = 0
        self.routing_done = 0
        self.routing_t_start = None
        self.routing_bar.configure(maximum=1, value=0)
        self.lbl_routing_caption.set("warte auf sweep_start …")
        self.iter_starts.clear()
        self.iter_durations_s.clear()
        # Skip everything VROOM already logged before now.
        self._access_offset = (
            VROOM_ACCESS_LOG.stat().st_size if VROOM_ACCESS_LOG.exists() else 0
        )

        # Reset finish notification so the next run can announce again.
        self._finish_announced = False
        self._curve_mtime = 0.0

        # Persist settings so a crash does not lose the user's entries.
        self._save_settings()

        self._log(f"$ {' '.join(cmd)}\n", "iter")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
                encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        except FileNotFoundError as e:
            self._log(f"start failed: {e}\n", "err")
            return
        self.run_started_at = time.time()
        self.lbl_status.set(f"RUNNING (PID {self.proc.pid}) — out_dir = {out_dir}")
        self.out_dir = out_dir
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_kill.configure(state="normal")
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _on_stop(self) -> None:
        if self.proc is None: return
        sentinel = self.out_dir / "STOP_REQUESTED"
        sentinel.write_text(f"requested via GUI at {time.strftime('%F %T')}\n")
        self._stop_pressed = True
        self.lbl_status.set("STOP-Signal gesetzt — wartet auf Iterationsabschluss …")
        self.btn_stop.configure(state="disabled")
        self._log(f"  STOP_REQUESTED geschrieben → {sentinel}\n", "warn")

    def _on_kill(self) -> None:
        if self.proc is None: return
        if not messagebox.askyesno("Hart abbrechen?",
                                    "Force-Kill bricht die Iteration mitten drin ab. "
                                    "Trainingsdaten der aktuellen Iter gehen verloren. "
                                    "Trotzdem killen?"):
            return
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(1)
            self.proc.terminate()
        except Exception as e:
            self._log(f"kill failed: {e}\n", "err")
        self.lbl_status.set("KILLED")

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            choice = messagebox.askyesnocancel(
                "Loop läuft noch",
                "Loop läuft noch.\n\n"
                "Ja  = STOP nach aktueller Iter + Fenster zu (Loop läuft weiter)\n"
                "Nein = hart abbrechen + zu\n"
                "Abbruch = Fenster offen lassen",
            )
            if choice is None:
                return
            if choice:
                # Graceful detach: write sentinel, leave subprocess alone.
                if hasattr(self, "out_dir"):
                    (self.out_dir / "STOP_REQUESTED").write_text(
                        f"requested via GUI close at {time.strftime('%F %T')}\n"
                    )
                self._save_settings()
                self._docker_stats_stop.set()
                self.destroy()
                return
            self._on_kill()
        self._save_settings()
        self._docker_stats_stop.set()
        self.destroy()

    def _open_outdir(self) -> None:
        p = Path(self.entries["out_dir"].get())
        if p.exists():
            os.startfile(str(p))   # noqa: S606  (Windows-only)
        else:
            messagebox.showinfo("nicht da", f"{p} existiert noch nicht.")

    # ── log + status ──────────────────────────────────────────────────
    def _reader_thread(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.log_q.put(line)
        rc = self.proc.wait()
        self.log_q.put(f"\n=== process exited with code {rc} ===\n")

    def _drain_log(self) -> None:
        try:
            while True:
                line = self.log_q.get_nowait()
                tag = None
                # User-controlled noise filter — sweep_start lines and explicit
                # markers are never filtered, but urllib3 / sklearn spam is.
                if (self._line_is_filtered(line)
                        and "sweep_start" not in line
                        and "sweep_done" not in line
                        and "=== oracle-loop iteration" not in line):
                    continue
                if "=== oracle-loop iteration" in line:
                    tag = "iter"
                    # parse "iteration K/N"
                    try:
                        seg = line.split("iteration", 1)[1].split("(", 1)[0]
                        k, n = seg.strip().split("/")
                        now = time.time()
                        prev_iter = self.cur_iter
                        self.cur_iter = int(k)
                        self.tot_iter = int(n)
                        self.progress.configure(maximum=self.tot_iter, value=self.cur_iter - 1)
                        self.lbl_iter_caption.set(
                            f"Iter {self.cur_iter} / {self.tot_iter}"
                        )
                        # Close the duration of the previous iter (if any).
                        if self.iter_starts and prev_iter == self.iter_starts[-1][0]:
                            self.iter_durations_s.append(
                                now - self.iter_starts[-1][1]
                            )
                        self.iter_starts.append((self.cur_iter, now))
                    except Exception:
                        pass
                elif "sweep_start" in line:
                    # Reset within-iter routing tracking and capture combo count.
                    m = _SWEEP_START_RE.search(line)
                    if m:
                        self.routing_total = int(m.group(1))
                        self.routing_done = 0
                        self.routing_t_start = time.time()
                        # Skip whatever VROOM logged before this iteration.
                        self._access_offset = (
                            VROOM_ACCESS_LOG.stat().st_size
                            if VROOM_ACCESS_LOG.exists() else 0
                        )
                        self.routing_bar.configure(
                            maximum=max(self.routing_total, 1), value=0
                        )
                        self.lbl_routing_caption.set(
                            f"0 / {self.routing_total} routes"
                        )
                    tag = "iter"
                elif "sweep_done" in line or "sweep: done in" in line:
                    self.routing_bar.configure(value=self.routing_total)
                    self.lbl_routing_caption.set(
                        f"{self.routing_total} / {self.routing_total} routes (fertig)"
                    )
                    tag = "ok"
                elif "early stop" in line or "STOP_REQUESTED" in line:
                    tag = "warn"
                elif "ERROR" in line or "Traceback" in line or "FAIL" in line:
                    tag = "err"
                elif "oracle-loop done" in line or "process exited with code 0" in line:
                    tag = "ok"
                    self.progress.configure(value=self.tot_iter)
                self._log(line, tag)
        except queue.Empty:
            pass
        # status check
        if self.proc is not None and self.proc.poll() is not None:
            rc = self.proc.returncode
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.btn_kill.configure(state="disabled")
            tag = "ok" if rc == 0 else "err"
            self.lbl_status.set(f"FERTIG (exit {rc})")
            self._log(f"final exit code = {rc}\n", tag)
            self.proc = None
            self._announce_finish(rc)
        self.after(120, self._drain_log)

    def _log(self, text: str, tag: str | None = None) -> None:
        # Smart auto-scroll: only follow the tail if the view was already at
        # the bottom.  This lets the user scroll up to inspect history without
        # the cursor jumping back every time a new line lands.
        try:
            at_bottom = float(self.log.yview()[1]) >= 0.999
        except (tk.TclError, IndexError, ValueError):
            at_bottom = True
        self.log.insert("end", text, tag) if tag else self.log.insert("end", text)
        if at_bottom:
            self.log.see("end")

    def _refresh_status(self) -> None:
        try:
            if hasattr(self, "out_dir"):
                pool = self.out_dir / "training_matrix.csv"
                hold = self.out_dir / "holdout_extreme.csv"
                hist = self.out_dir / "oracle_loop_history.csv"
                if pool.exists():
                    self.lbl_pool.set(f"Pool: {self._row_count(pool):,} Zeilen")
                if hold.exists():
                    self.lbl_hold.set(f"Holdout: {self._row_count(hold):,}")
                if hist.exists():
                    import csv as _csv
                    rows = list(_csv.DictReader(hist.open("r", encoding="utf-8")))
                    if rows:
                        last = rows[-1]
                        try:
                            self.lbl_mape.set(f"val MAPE: {float(last['mape_pct']):.2f} %")
                        except (KeyError, ValueError): pass
                        try:
                            self.lbl_holdout_mape.set(
                                f"holdout MAPE: {float(last['holdout_mape_pct']):.2f} %")
                        except (KeyError, ValueError): pass
            if self.run_started_at is not None:
                el = int(time.time() - self.run_started_at)
                h, m, s = el // 3600, (el % 3600) // 60, el % 60
                self.lbl_elapsed.set(f"Laufzeit: {h:02d}:{m:02d}:{s:02d}")
            self._update_etas()
            self._update_window_title()
        except Exception:
            pass
        self.after(3000, self._refresh_status)

    def _update_window_title(self) -> None:
        """Reflect run state in the title so taskbar entry is informative."""
        # Don't override an explicit FINISHED title set by _announce_finish.
        if self._finish_announced:
            return
        if self.proc is None:
            self.title(BASE_TITLE)
            return
        parts = [BASE_TITLE, f"iter {self.cur_iter}/{self.tot_iter}"]
        if self.routing_total > 0 and self.routing_done > 0:
            parts.append(
                f"routes {self.routing_done}/{self.routing_total}"
            )
        if self.last_total_eta_s is not None:
            parts.append(f"ETA {self._fmt_eta(self.last_total_eta_s)}")
        self.title(" — ".join(parts))

    @staticmethod
    def _row_count(p: Path) -> int:
        with p.open("rb") as f:
            return sum(1 for _ in f) - 1   # minus header

    # ── routing progress + ETA ────────────────────────────────────────
    def _refresh_routing(self) -> None:
        """Tail vroom/access.log and bump the routing progress bar."""
        try:
            if (self.proc is not None and self.routing_total > 0
                    and VROOM_ACCESS_LOG.exists()):
                size = VROOM_ACCESS_LOG.stat().st_size
                if size > self._access_offset:
                    with VROOM_ACCESS_LOG.open("rb") as fh:
                        fh.seek(self._access_offset)
                        chunk = fh.read(size - self._access_offset)
                    self._access_offset = size
                    new_posts = sum(
                        1 for ln in chunk.splitlines()
                        if _POST_RE.search(ln.decode("ascii", errors="replace"))
                    )
                    if new_posts:
                        self.routing_done = min(
                            self.routing_done + new_posts, self.routing_total
                        )
                        self.routing_bar.configure(value=self.routing_done)
                        self.lbl_routing_caption.set(
                            f"{self.routing_done} / {self.routing_total} routes  "
                            f"({self.routing_done / self.routing_total:.0%})"
                        )
        except Exception:
            pass
        self.after(500, self._refresh_routing)

    def _update_etas(self) -> None:
        # Per-iter ETA: extrapolate from current routing rate.
        iter_eta_s: float | None = None
        if (self.routing_total > 0 and self.routing_t_start is not None
                and self.routing_done > 0):
            elapsed = time.time() - self.routing_t_start
            rate = self.routing_done / max(elapsed, 1e-6)
            remaining = max(self.routing_total - self.routing_done, 0)
            iter_eta_s = remaining / rate if rate > 0 else None
            self.lbl_iter_eta.set(f"Iter-ETA: {self._fmt_eta(iter_eta_s)}")
        elif self.routing_total > 0:
            self.lbl_iter_eta.set("Iter-ETA: …")
        else:
            self.lbl_iter_eta.set("Iter-ETA: –")

        # Loop ETA: per-iter ETA + remaining iters * avg completed iter time.
        total_eta: float | None = None
        if self.tot_iter > 0 and self.cur_iter > 0:
            remaining_iters = max(self.tot_iter - self.cur_iter, 0)
            if self.iter_durations_s:
                avg = sum(self.iter_durations_s) / len(self.iter_durations_s)
            elif iter_eta_s is not None and self.routing_t_start:
                # No completed iter yet — extrapolate from current iter's
                # elapsed + projected remaining.
                avg = (time.time() - self.routing_t_start) + iter_eta_s
            else:
                avg = None
            if avg is not None:
                total_eta = (iter_eta_s or 0) + remaining_iters * avg
                eta_str = self._fmt_eta(total_eta)
                wall = self._fmt_wallclock(total_eta)
                self.lbl_total_eta.set(f"Loop-ETA: {eta_str}  → {wall}")
            else:
                self.lbl_total_eta.set("Loop-ETA: …")
        else:
            self.lbl_total_eta.set("Loop-ETA: –")
        self.last_total_eta_s = total_eta

    @staticmethod
    def _fmt_wallclock(eta_s: float | None) -> str:
        """Format absolute end-time from a relative ETA in seconds."""
        if eta_s is None or eta_s <= 0 or eta_s > 7 * 24 * 3600:
            return "–"
        end = _dt.datetime.now() + _dt.timedelta(seconds=int(eta_s))
        # Show date only if it's not today; always show weekday + HH:MM.
        if end.date() == _dt.date.today():
            return end.strftime("heute %H:%M")
        if end.date() == _dt.date.today() + _dt.timedelta(days=1):
            return end.strftime("morgen %H:%M")
        return f"{_WEEKDAY_DE[end.weekday()]} {end.strftime('%d.%m. %H:%M')}"

    @staticmethod
    def _fmt_eta(seconds: float | None) -> str:
        if seconds is None or seconds <= 0 or seconds > 7 * 24 * 3600:
            return "–"
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ── service health ────────────────────────────────────────────────
    def _check_services(self) -> None:
        self.vroom_up = self._http_ok(VROOM_HEALTH_URL, timeout=1.5)
        self.valhalla_up = self._http_ok(VALHALLA_HEALTH_URL, timeout=1.5)
        self.lbl_vroom.set(
            f"VROOM: {'● up' if self.vroom_up else '○ down'}"
        )
        self.lbl_valhalla.set(
            f"Valhalla: {'● up' if self.valhalla_up else '○ down'}"
        )
        # Re-poll every 10s while idle, every 30s while running (less noise).
        next_ms = 30_000 if (self.proc and self.proc.poll() is None) else 10_000
        self.after(next_ms, self._check_services)

    @staticmethod
    def _http_ok(url: str, timeout: float = 1.5) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return 200 <= resp.status < 500
        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionError, TimeoutError, OSError):
            return False

    # ── resources (system / docker / loop process) ────────────────────
    def _refresh_resources(self) -> None:
        """Update System + Loop-process labels (cheap) and drain docker queue."""
        try:
            self._update_system_and_loop_labels()
            self._drain_docker_stats()
        except Exception:
            pass
        # 2 s cadence is plenty: docker stats arrives every ~5 s, system every tick.
        self.after(2000, self._refresh_resources)

    def _update_system_and_loop_labels(self) -> None:
        if psutil is None:
            self.lbl_res_system.set("System: psutil fehlt — pip install psutil")
            return
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        used_gb = (vm.total - vm.available) / (1024 ** 3)
        total_gb = vm.total / (1024 ** 3)
        self.lbl_res_system.set(
            f"System:    cpu {cpu:5.1f} %   ram {used_gb:5.1f} / {total_gb:4.1f} GB "
            f"({vm.percent:.0f} %)"
        )

        if self.proc is None or self.proc.poll() is not None:
            self.lbl_res_loop.set("Loop-Prozess: –")
            return
        try:
            p = psutil.Process(self.proc.pid)
            # Aggregate the loop process tree (joblib workers in threading
            # backend share the parent; with loky/process backends they would
            # be children).
            procs = [p] + p.children(recursive=True)
            cpu_pct = sum(pp.cpu_percent(interval=None) for pp in procs)
            mem_mb = sum(pp.memory_info().rss for pp in procs) / (1024 ** 2)
            self.lbl_res_loop.set(
                f"Loop-Prozess: cpu {cpu_pct:5.1f} %   mem {mem_mb:6.0f} MB   "
                f"PID {self.proc.pid}  ({len(procs)} procs/threads)"
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.lbl_res_loop.set(f"Loop-Prozess: PID {self.proc.pid} (unzugänglich)")

    def _drain_docker_stats(self) -> None:
        """Pull the latest docker-stats payload off the worker queue."""
        latest: dict[str, dict] | None = None
        try:
            while True:
                latest = self._docker_stats_q.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            if not self._docker_available:
                self.lbl_res_vroom.set("VROOM:     docker CLI nicht gefunden")
                self.lbl_res_valhalla.set("Valhalla:  docker CLI nicht gefunden")
            return
        for name, var in [("vroom", self.lbl_res_vroom),
                           ("valhalla", self.lbl_res_valhalla)]:
            row = latest.get(name)
            if row is None:
                var.set(f"{name.capitalize():9s} container down / not found")
                continue
            label = "VROOM:    " if name == "vroom" else "Valhalla: "
            var.set(
                f"{label}cpu {row['cpu_pct']:5.1f} %   mem {row['mem']:>11s}   "
                f"({row['mem_pct']:.1f} %)"
            )

    def _docker_stats_worker(self) -> None:
        """Background thread: poll docker stats every 5 s, push to queue."""
        while not self._docker_stats_stop.is_set():
            try:
                proc = subprocess.run(
                    ["docker", "stats", "--no-stream",
                     "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}",
                     "vroom", "valhalla"],
                    capture_output=True, text=True, timeout=8,
                )
                parsed: dict[str, dict] = {}
                for ln in proc.stdout.splitlines():
                    parts = ln.strip().split("|")
                    if len(parts) != 4:
                        continue
                    name, cpu, mem, memp = parts
                    parsed[name.strip()] = {
                        "cpu_pct": _safe_pct(cpu),
                        "mem": mem.split("/")[0].strip(),
                        "mem_pct": _safe_pct(memp),
                    }
                try:
                    # Replace any stale entry so the GUI sees only the freshest.
                    while True:
                        self._docker_stats_q.get_nowait()
                except queue.Empty:
                    pass
                self._docker_stats_q.put(parsed)
            except (FileNotFoundError, subprocess.TimeoutExpired,
                    subprocess.CalledProcessError):
                # docker disappeared mid-run — flag and back off.
                self._docker_available = shutil.which("docker") is not None
            # Sleep cooperatively so _on_close can stop us quickly.
            if self._docker_stats_stop.wait(timeout=5.0):
                return

    # ── preset / quick-buttons / log filter ───────────────────────────
    def _on_preset_change(self, _event=None) -> None:
        name = self.var_preset.get()
        overrides = PRESETS.get(name, {})
        for key, value in overrides.items():
            if key not in self.entries:
                continue
            entry = self.entries[key]
            entry.delete(0, "end")
            entry.insert(0, value)

    def _open_logfile(self) -> None:
        if not hasattr(self, "out_dir"):
            messagebox.showinfo(
                "kein Lauf gestartet",
                "Out-Verzeichnis ist erst nach Start verfügbar.",
            )
            return
        p = Path(self.out_dir) / "oracle_loop.log"
        if not p.exists():
            messagebox.showinfo("nicht da", f"{p} existiert noch nicht.")
            return
        os.startfile(str(p))  # noqa: S606 — Windows-only

    def _open_learning_curve(self) -> None:
        if not hasattr(self, "out_dir"):
            messagebox.showinfo(
                "kein Lauf gestartet",
                "Out-Verzeichnis ist erst nach Start verfügbar.",
            )
            return
        p = Path(self.out_dir) / "learning_curve.png"
        if not p.exists():
            messagebox.showinfo(
                "noch keine Lernkurve",
                f"{p} entsteht erst nach Iter 1.",
            )
            return
        os.startfile(str(p))  # noqa: S606 — Windows-only

    def _clear_log(self) -> None:
        self.log.delete("1.0", "end")

    def _line_is_filtered(self, line: str) -> bool:
        """Return True if user toggles say this line should be hidden."""
        if self.var_hide_urllib_debug and self.var_hide_urllib_debug.get():
            if "DEBUG Starting new HTTP" in line:
                return True
            if "DEBUG http://localhost:" in line:
                return True
        if self.var_hide_sklearn_warn and self.var_hide_sklearn_warn.get():
            if ("InconsistentVersionWarning" in line
                    or "scikit-learn.org" in line
                    or "warnings.warn(" in line
                    or "sklearn/base.py" in line):
                return True
        return False

    # ── learning-curve thumbnail ──────────────────────────────────────
    def _refresh_learning_curve(self) -> None:
        try:
            if Image is None or ImageTk is None or not hasattr(self, "out_dir"):
                return
            curve = Path(self.out_dir) / "learning_curve.png"
            if not curve.exists():
                return
            mtime = curve.stat().st_mtime
            if mtime == self._curve_mtime:
                return
            # Read into memory before letting Tk hold the file handle.
            img = Image.open(curve)
            img.load()
            # Fit to the label's current width, max 220 px tall.
            target_w = max(self.lbl_curve.winfo_width(), 600)
            scale = min(target_w / img.width, 220 / img.height, 1.0)
            new_size = (int(img.width * scale), int(img.height * scale))
            if new_size[0] > 0 and new_size[1] > 0:
                img = img.resize(new_size, Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self.lbl_curve.configure(image=tk_img, text="")
            self._curve_image_ref = tk_img    # prevent GC
            self._curve_mtime = mtime
        except Exception:
            pass
        finally:
            self.after(5000, self._refresh_learning_curve)

    # ── finish notification ───────────────────────────────────────────
    def _announce_finish(self, rc: int) -> None:
        if self._finish_announced:
            return
        self._finish_announced = True
        try:
            self.bell()
            if winsound is not None:
                # SystemAsterisk on success, SystemHand on failure.
                alias = "SystemAsterisk" if rc == 0 else "SystemHand"
                winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception:
            pass
        # Flash the window in the taskbar (Windows: -topmost toggle works).
        try:
            self.attributes("-topmost", True)
            self.lift()
            self.after(1500, lambda: self.attributes("-topmost", False))
        except Exception:
            pass
        prefix = "✓ " if rc == 0 else "✗ "
        self.title(f"{prefix}{BASE_TITLE} — FERTIG (exit {rc})")

    # ── settings persistence ──────────────────────────────────────────
    def _load_settings(self) -> dict:
        try:
            if SETTINGS_PATH.exists():
                return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_settings(self) -> None:
        try:
            data = {k: e.get() for k, e in self.entries.items()}
            data["_window_geometry"] = self.geometry()
            if self.var_hide_urllib_debug is not None:
                data["_hide_urllib_debug"] = bool(self.var_hide_urllib_debug.get())
            if self.var_hide_sklearn_warn is not None:
                data["_hide_sklearn_warn"] = bool(self.var_hide_sklearn_warn.get())
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    OracleLoopGUI().mainloop()
