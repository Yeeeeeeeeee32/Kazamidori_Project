"""
ui_qt/workers.py

Background worker thread that executes the full RocketPy physics pipeline
entirely off the GUI thread.

Thread lifecycle
----------------
    worker = SimulationWorker(params)
    worker.progress.connect(progress_bar.setValue)           # 0–100
    worker.sig_nominal_done.connect(on_nominal_slot)         # fired after nominal run
    worker.sig_progress_updated.connect(on_mc_tick_slot)     # (current, total) per MC run
    worker.finished.connect(on_finished_slot)                # always emitted
    worker.error.connect(on_error_slot)                      # only on exception
    worker.start()   # returns immediately; physics runs on the new thread

    worker.stop()    # graceful cancel from any thread

Result dict keys  (finished signal payload)
-------------------------------------------
  cancelled       bool   True if stop() was called
  has_sim_result  bool   True on successful full run

  Nominal trajectory (all arrays as Python lists):
    t_vals, x_vals, y_vals, z_vals  — time (s), East/North/Up (m) arrays
    apogee_m        float  — peak altitude above launch (m)
    hang_time       float  — total flight time (s)
    impact_x        float  — nominal landing East offset from launch (m)
    impact_y        float  — nominal landing North offset from launch (m)
    r_horiz         float  — horizontal range at landing (m)

  Monte Carlo statistics:
    scatter         list[(x_m, y_m)]  — successful landing positions
    r_N_radius      float  — landing_prob-th percentile radius (m)
    cep             float  — CEP50: 50th-percentile radius (m)
    ellipse         dict|None  — error ellipse {cx, cy, a, b, angle_rad}
    kde_contours    list[dict] — KDE contour dicts {points_m, prob_frac, label}
    n_runs          int    — number of successful MC runs
    landing_prob    int    — percentile used (e.g. 90)
"""

from __future__ import annotations

import math
import threading
from typing import Any

import traceback
from PySide6.QtCore import QThread, Signal

from core.simulation   import simulate_once
from core.wind_model   import create_wind_profile, sample_wind_nodes
from core.monte_carlo  import (
    run_mc_scatter,
    compute_error_ellipse,
    compute_cep,
    compute_cep_circle,
    compute_kde_contours,
)

# ── Default rocket / motor configuration ─────────────────────────────────────
# Represents a typical 70 mm-diameter competition rocket with an E-class motor
# (~65 Ns total impulse, 1.2 s burn).  These values are used whenever the
# user has not loaded a motor file or overridden rocket params via the UI.

_DEFAULT_ROCKET: dict[str, Any] = {
    "rail":             1.5,      # launch rail length (m)
    "airframe_mass":    1.0,      # airframe dry mass (kg)
    "airframe_cg":      0.50,     # centre of gravity from nose (m)
    "airframe_len":     1.10,     # total airframe length (m)
    "radius":           0.035,    # body radius (m) — 70 mm diameter
    "nose_len":         0.20,     # nose cone length (m)
    "fin_root":         0.12,     # fin root chord (m)
    "fin_tip":          0.06,     # fin tip chord (m)
    "fin_span":         0.08,     # fin semi-span (m)
    "fin_pos":          0.95,     # fin leading-edge position from nose (m)
    "motor_pos":        1.00,     # motor CG from nose (m)
    "motor_dry_mass":   0.10,     # motor dry mass (kg)
    "backfire_delay":   0.5,      # ejection charge fires this many s after burnout
    "para_cd":          1.5,      # parachute drag coefficient
    "para_area":        0.28,     # parachute reference area (m²) — ≈ 60 cm diameter
    "para_lag":         0.8,      # deployment lag (s)
    # E-class thrust curve: peak 80 N, average ~54 N, total ~65 Ns
    "thrust_data": [
        [0.000,  0.0],
        [0.050, 60.0],
        [0.100, 80.0],
        [0.400, 70.0],
        [0.800, 65.0],
        [1.100, 50.0],
        [1.200,  0.0],
    ],
    "motor_burn_time": 1.2,
}

# Number of MC runs per progress tick.  Smaller → finer progress granularity
# but slightly more overhead per RocketPy call.
_MC_BATCH_SIZE: int = 10


# ── Worker ────────────────────────────────────────────────────────────────────

class SimulationWorker(QThread):
    """
    Executes the full RocketPy + Monte Carlo pipeline on a dedicated thread.

    Progress milestones
    -------------------
     2 %  wind profile construction
    10 %  nominal (single) simulation started
    25 %  nominal simulation complete
    25–90 %  Monte Carlo loop (per-batch ticks)
    92 %  statistical analysis started
    98 %  packaging result
    """

    progress             = Signal(int)        # 0–100 coarse progress bar
    finished             = Signal(dict)       # always emitted — check result["cancelled"]
    error                = Signal(str)        # only on unhandled exception
    sig_nominal_done     = Signal(dict)       # emitted immediately after nominal run
    sig_progress_updated = Signal(int, int)   # (current_iteration, total_iterations)

    def __init__(self, params: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self._params     = dict(params)
        self._stop_event = threading.Event()

    # ── Public control ─────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Request graceful cancellation. Safe to call from any thread."""
        self._stop_event.set()

    # ── QThread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            result = self._run_physics()
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())

    # ── Physics pipeline ───────────────────────────────────────────────────────

    def _run_physics(self) -> dict:
        p = self._params

        # ── Wind profile + diagnostic nodes ───────────────────────────────────
        self.progress.emit(2)
        u_prof, v_prof = self._build_wind_profiles(p)
        wind_nodes = sample_wind_nodes(u_prof, v_prof)

        # ── Assemble full simulation params ────────────────────────────────────
        self.progress.emit(5)
        sim_params = self._build_sim_params(p, u_prof, v_prof)

        # ── Nominal single run ─────────────────────────────────────────────────
        self.progress.emit(10)
        nominal = simulate_once(
            elev=float(p.get("elev", 85.0)),
            azi=float(p.get("azim",  0.0)),
            params=sim_params,
        )
        if not nominal["ok"]:
            raise RuntimeError(f"Nominal simulation failed: {nominal['error']}")
        self.progress.emit(25)

        # ── Emit nominal result immediately — UI can render 3-D plot now ───────
        # This crosses the thread boundary via Qt's queued connection before
        # the MC phase begins, so the operator sees the trajectory without
        # waiting for all Monte Carlo runs to finish.
        self.sig_nominal_done.emit(self._package_nominal(nominal, wind_nodes))

        if self._stop_event.is_set():
            return {"cancelled": True, "has_sim_result": False}

        # ── Monte Carlo loop with per-iteration heartbeat ──────────────────────
        scatter = self._run_mc_loop(sim_params, p)

        if self._stop_event.is_set():
            return {
                "cancelled":      True,
                "has_sim_result": False,
                "impact_x":  nominal["impact_x"],
                "impact_y":  nominal["impact_y"],
                "apogee_m":  nominal["apogee_m"],
                "hang_time": nominal["hang_time"],
            }

        # ── Statistical analysis ───────────────────────────────────────────────
        self.progress.emit(92)
        prob_pct = int(p.get("cep_prob", 90))
        stats    = self._compute_stats(scatter, prob_pct)

        self.progress.emit(98)
        result = self._package_result(nominal, scatter, stats, prob_pct,
                                      cancelled=self._stop_event.is_set())
        result["nominal_surf_spd"] = float(p.get("surf_spd", 0.0))
        result["nominal_surf_dir"] = float(p.get("surf_dir", 0.0))
        result["wind_nodes"]       = wind_nodes
        return result

    # ── Step helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_wind_profiles(
        p: dict,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Construct a smooth vertical wind profile from surface + upper obs.

        The surface reading is blended with the upper-wind GPV level using
        create_wind_profile's three-zone approach (surface ramp → blend →
        pure GPV).  Upper wind is assumed at 500 m AGL unless overridden.
        """
        upper_alt: float = float(p.get("upper_alt", 500.0))
        gpv_levels = [
            (upper_alt, float(p.get("up_spd", 8.0)), float(p.get("up_dir", 90.0))),
        ]
        return create_wind_profile(
            gpv_levels=gpv_levels,
            obs_speed=float(p.get("surf_spd",  4.0)),
            obs_dir=float(p.get("surf_dir",  100.0)),
            obs_alt=3.0,
            blend_alt=100.0,
        )

    @staticmethod
    def _build_sim_params(
        p: dict,
        u_prof: list,
        v_prof: list,
    ) -> dict:
        """
        Merge UI params with the default rocket configuration.

        Any key present in *p* that also exists in _DEFAULT_ROCKET will be
        overridden; all other defaults are preserved.  This allows future UI
        widgets (motor file, rocket dimensions) to inject their values here
        without changing the worker logic.
        """
        params = dict(_DEFAULT_ROCKET)
        params.update({
            "launch_lat":   float(p.get("launch_lat", 35.0)),
            "launch_lon":   float(p.get("launch_lon", 135.0)),
            "elev":         float(p.get("elev",       85.0)),
            "azi":          float(p.get("azim",        0.0)),
            "wind_u_prof":  u_prof,
            "wind_v_prof":  v_prof,
        })
        # Allow caller to override any rocket param (e.g. from a loaded motor file)
        for key in _DEFAULT_ROCKET:
            if key in p and key not in ("wind_u_prof", "wind_v_prof"):
                params[key] = p[key]
        return params

    @staticmethod
    def _package_nominal(nominal: dict, wind_nodes: list[dict]) -> dict:
        """Package the nominal trajectory for sig_nominal_done.

        The UI can connect to sig_nominal_done and render the 3-D trajectory
        as soon as the nominal run completes — before any MC runs start.
        Arrays are converted to lists so no numpy types cross the signal
        boundary.
        """
        def _to_list(v):
            return v.tolist() if hasattr(v, "tolist") else list(v)

        return {
            "t_vals":     _to_list(nominal["t_vals"]),
            "x_vals":     _to_list(nominal["x_vals"]),
            "y_vals":     _to_list(nominal["y_vals"]),
            "z_vals":     _to_list(nominal["z_vals"]),
            "apogee_m":   float(nominal["apogee_m"]),
            "hang_time":  float(nominal["hang_time"]),
            "impact_x":   float(nominal["impact_x"]),
            "impact_y":   float(nominal["impact_y"]),
            "r_horiz":    float(nominal["r_horiz"]),
            "wind_nodes": wind_nodes,
        }

    def _run_mc_loop(
        self,
        sim_params: dict,
        p: dict,
    ) -> list[tuple[float, float]]:
        """Run the MC scatter one iteration at a time, emitting a heartbeat
        signal after every run.

        sig_progress_updated(i+1, n_total) fires on every iteration so the
        UI's progress bar updates smoothly without waiting for a batch to
        complete.  The coarser `progress` signal (0–100 int) is also emitted
        at the same cadence for backwards compatibility with any slot already
        connected to it.

        Progress moves from 25 % to 90 % as runs complete.
        """
        n_total    = int(p.get("mc_runs",    50))
        wind_unc   = float(p.get("wind_unc",   0.20))
        thrust_unc = float(p.get("thrust_unc", 0.05))
        scatter: list[tuple[float, float]] = []

        for i in range(n_total):
            if self._stop_event.is_set():
                break

            batch_scatter, _ = run_mc_scatter(
                sim_params, 1,
                wind_unc, thrust_unc,
                stop_flag=self._stop_event,
            )
            scatter.extend(batch_scatter)

            # Per-iteration heartbeat — drives fine-grained UI updates
            self.sig_progress_updated.emit(i + 1, n_total)
            # Coarse 0-100 progress for any connected progress bar
            pct = 25 + int((i + 1) / n_total * 65)
            self.progress.emit(min(pct, 90))

        return scatter

    @staticmethod
    def _compute_stats(
        scatter: list[tuple[float, float]],
        prob_pct: int,
    ) -> dict:
        """
        Compute all statistical outputs from the landing scatter.

        r_N_radius  — the prob_pct-th percentile radial distance from the
                      scatter centroid.  Used as the displayed landing radius.
        cep         — CEP50: 50th-percentile radius from the centroid.
        ellipse     — chi-squared scaled covariance ellipse at prob_pct.
        cep_circle  — metric polygon of the CEP50 circle.
        kde_contours — KDE probability-mass contour dicts.
        """
        empty = {
            "r_N_radius":   0.0,
            "cep":          0.0,
            "ellipse":      None,
            "cep_circle":   None,
            "kde_contours": [],
        }
        if not scatter:
            return empty

        # Percentile radius from centroid
        cx = sum(x for x, _ in scatter) / len(scatter)
        cy = sum(y for _, y in scatter) / len(scatter)
        radii = sorted(math.hypot(x - cx, y - cy) for x, y in scatter)
        n    = len(radii)
        idx  = min(int(prob_pct / 100.0 * n), n - 1)
        r_N  = radii[idx]

        return {
            "r_N_radius":   r_N,
            "cep":          compute_cep(scatter),
            "ellipse":      compute_error_ellipse(scatter, prob_pct=prob_pct),
            "cep_circle":   compute_cep_circle(scatter),
            "kde_contours": compute_kde_contours(scatter, conf_pct=prob_pct),
        }

    @staticmethod
    def _package_result(
        nominal:    dict,
        scatter:    list,
        stats:      dict,
        prob_pct:   int,
        *,
        cancelled:  bool,
    ) -> dict:
        """
        Assemble the finished-signal payload.

        Numpy arrays from simulate_once are converted to Python lists so the
        dict is safe to pass through Qt's queued connection type system and
        remains JSON-serialisable for future persistence.
        """
        def _to_list(v):
            return v.tolist() if hasattr(v, "tolist") else list(v)

        t_vals = _to_list(nominal["t_vals"])
        x_vals = _to_list(nominal["x_vals"])
        y_vals = _to_list(nominal["y_vals"])
        z_vals = _to_list(nominal["z_vals"])

        return {
            "cancelled":      cancelled,
            "has_sim_result": not cancelled,
            # Nominal trajectory (separate arrays)
            "t_vals":    t_vals,
            "x_vals":    x_vals,
            "y_vals":    y_vals,
            "z_vals":    z_vals,
            "apogee_m":  float(nominal["apogee_m"]),
            "hang_time": float(nominal["hang_time"]),
            "impact_x":  float(nominal["impact_x"]),   # East offset (m)
            "impact_y":  float(nominal["impact_y"]),   # North offset (m)
            "r_horiz":   float(nominal["r_horiz"]),
            # MC statistics
            "scatter":      scatter,
            "r_N_radius":   float(stats.get("r_N_radius", 0.0)),
            "cep":          float(stats.get("cep", 0.0)),
            "ellipse":      stats.get("ellipse"),
            "kde_contours": stats.get("kde_contours", []),
            "n_runs":       len(scatter),
            "landing_prob": prob_pct,
            # ── Alias keys consumed by General B / future views ────────────────
            # trajectory_3d: list of [East_m, North_m, Up_m] per time-step
            "trajectory_3d":     list(zip(x_vals, y_vals, z_vals)),
            "mc_scatter_points": scatter,                   # alias for scatter
            "apogee":            float(nominal["apogee_m"]),
            "impact_distance":   float(nominal["r_horiz"]),
        }
