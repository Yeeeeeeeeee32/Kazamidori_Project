"""
ui_qt/workers.py

Background worker threads for the Kazamidori simulation engine.

SimulationWorker — QThread subclass that runs the Monte Carlo physics
engine (or this mock stand-in) entirely off the GUI thread.

Signals
-------
progress(int)   0–100 percentage; emitted once per MC iteration.
finished(dict)  Final results payload; always emitted (even on cancel).
                Callers should check result["cancelled"] before use.
error(str)      Human-readable message if an unhandled exception occurs.

Usage
-----
    worker = SimulationWorker(params)
    worker.progress.connect(progress_bar.setValue)
    worker.finished.connect(on_finished_slot)
    worker.error.connect(on_error_slot)
    worker.start()          # returns immediately; run() is on the new thread

    # To cancel gracefully from any thread:
    worker.stop()
"""

from __future__ import annotations

import math
import random
import threading
import time
from typing import Any

from PySide6.QtCore import QThread, Signal


class SimulationWorker(QThread):
    """
    Monte Carlo trajectory simulation worker.

    Thread lifecycle
    ----------------
    start()  → run() on new thread → finished(result) → thread exits.
    stop()   sets the stop event; the current iteration finishes, then
             run() emits finished({"cancelled": True, ...}) and returns.
    """

    progress = Signal(int)    # 0–100
    finished = Signal(dict)   # results payload (always emitted)
    error    = Signal(str)    # only on unhandled exception

    def __init__(self, params: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self._params     = dict(params)
        self._stop_event = threading.Event()

    # ── Public control (thread-safe) ───────────────────────────────────────────

    def stop(self) -> None:
        """Request graceful cancellation. Safe to call from any thread."""
        self._stop_event.set()

    # ── QThread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            result = self._run_mc()
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    # ── Simulation body ────────────────────────────────────────────────────────

    def _run_mc(self) -> dict:
        """
        Mock Monte Carlo loop.  Each iteration represents one simulated flight.

        The real implementation would call core.simulation.simulate_once here.
        The mock uses a simplified ballistic model with Gaussian perturbations
        so the result values are physically plausible for UI testing.

        Cap at 50 iterations to keep the mock under ~2 s regardless of the
        mc_runs parameter (the real engine will honour the full count).
        """
        p       = self._params
        n_full  = int(p.get("mc_runs",    200))
        n       = min(n_full, 50)           # mock cap; remove for real engine
        w_spd   = float(p.get("wind_speed", 4.0))
        w_dir   = float(p.get("wind_dir",  100.0))
        w_unc   = float(p.get("wind_unc",   0.20))
        t_unc   = float(p.get("thrust_unc", 0.05))

        w_dir_rad          = math.radians(w_dir)
        land_e_samples: list[float] = []
        land_n_samples: list[float] = []
        apogee_samples: list[float] = []
        rng = random.Random()

        for i in range(n):
            if self._stop_event.is_set():
                break

            # ── Perturb wind and thrust ────────────────────────────────────────
            spd         = max(0.0, w_spd * (1.0 + rng.gauss(0.0, w_unc)))
            angle       = w_dir_rad + rng.gauss(0.0, math.radians(10.0 * w_unc))
            thrust_mult = 1.0 + rng.gauss(0.0, t_unc)

            # ── Simplified ballistic arc ───────────────────────────────────────
            v0    = 120.0 * thrust_mult          # representative muzzle speed
            g     = 9.81
            tof   = 2.0 * v0 / g                # time of flight (s)
            apex  = v0 ** 2 / (2.0 * g)         # apogee altitude (m)

            land_e_samples.append(spd * math.sin(angle) * tof)
            land_n_samples.append(spd * math.cos(angle) * tof)
            apogee_samples.append(apex)

            # ── Progress heartbeat ─────────────────────────────────────────────
            time.sleep(0.04)
            self.progress.emit(int((i + 1) / n * 100))

        return self._aggregate(
            land_e_samples, land_n_samples, apogee_samples,
            w_spd, w_dir,
            cancelled=self._stop_event.is_set(),
        )

    # ── Result aggregation ─────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(
        land_e:    list[float],
        land_n:    list[float],
        apogee:    list[float],
        wind_speed: float,
        wind_dir:   float,
        *,
        cancelled: bool,
    ) -> dict:
        if not land_e:
            return {"cancelled": cancelled}

        n      = len(land_e)
        mean_e = sum(land_e) / n
        mean_n = sum(land_n) / n
        mean_ap = sum(apogee) / n

        # Radial distances from the mean landing point
        radii = sorted(
            math.hypot(e - mean_e, nn - mean_n)
            for e, nn in zip(land_e, land_n)
        )
        cep50 = radii[max(0, int(0.50 * n) - 1)]
        r90   = radii[max(0, int(0.90 * n) - 1)]

        return {
            "cancelled":     cancelled,
            "land_offset_e": mean_e,
            "land_offset_n": mean_n,
            "r90_radius":    r90,
            "mc_cep":        cep50,
            "apogee_alt":    mean_ap,
            "flight_time":   2.0 * math.sqrt(max(0.0, 2.0 * mean_ap / 9.81)),
            "n_runs":        n,
            "wind_speed":    wind_speed,
            "wind_dir":      wind_dir,
            "has_sim_result": not cancelled,
        }
