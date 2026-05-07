"""
ui_qt/workers.py

Background worker thread that executes the full RocketPy physics pipeline
entirely off the GUI thread.

Thread lifecycle — two-stage pipeline
--------------------------------------
    worker = SimulationWorker(params)
    worker.progress.connect(progress_bar.setValue)           # 0–100
    worker.sig_nominal_done.connect(on_nominal_slot)         # Stage 1 complete
    worker.sig_progress_updated.connect(on_mc_tick_slot)     # (current, total) per MC run
    worker.sig_mc_done.connect(on_mc_done_slot)              # Stage 2 complete
    worker.finished.connect(on_finished_slot)                # always emitted last
    worker.error.connect(on_error_slot)                      # only on exception
    worker.start()   # returns immediately; physics runs on the new thread

    worker.stop()    # graceful cancel from any thread

Emission order on a successful run
------------------------------------
  sig_nominal_done  → sig_progress_updated × n_runs → sig_mc_done → finished

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
    finished             = Signal(dict)       # always emitted last — check result["cancelled"]
    error                = Signal(str)        # only on unhandled exception
    sig_nominal_done     = Signal(dict)       # Stage 1: emitted after nominal run
    sig_progress_updated = Signal(int, int)   # Stage 2: (current_iteration, total_iterations)
    sig_mc_done          = Signal(dict)       # Stage 2: emitted after MC loop + statistics
    sig_status_text      = Signal(str)        # Human-readable stage label for the status bar

    def __init__(self, params: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self._params     = dict(params)
        self._stop_event = threading.Event()

    # ── Public control ─────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Request graceful cancellation. Safe to call from any thread."""
        self._stop_event.set()

    # ── QThread entry point — two-stage pipeline ──────────────────────────────

    def run(self) -> None:
        """Execute the two-stage simulation pipeline on the worker thread.

        All data crosses the thread boundary exclusively through Qt signals —
        no shared mutable state is accessed from the GUI thread during a run.
        """
        try:
            p = self._params

            # ════════════════════════════════════════════════════════════════
            # STAGE 1 — Nominal (baseline) trajectory
            # Build wind profile → run single deterministic RocketPy flight
            # → emit sig_nominal_done so the UI can paint the 3-D path now.
            # ════════════════════════════════════════════════════════════════

            self.progress.emit(2)
            u_prof, v_prof = self._build_wind_profiles(p)
            # sample_wind_nodes reads the 5 explicit grid points inserted into
            # the profile by create_wind_profile (3, 10, 150, 300, 600 m AGL)
            wind_nodes = sample_wind_nodes(u_prof, v_prof)

            self.progress.emit(5)
            sim_params = self._build_sim_params(p, u_prof, v_prof)

            self.progress.emit(10)
            self.sig_status_text.emit("Simulating nominal trajectory...")
            nominal = simulate_once(
                elev=float(p.get("elev", 85.0)),
                azi=float(p.get("azim",  0.0)),
                params=sim_params,
            )
            if not nominal["ok"]:
                raise RuntimeError(f"Nominal simulation failed: {nominal['error']}")
            self.progress.emit(25)

            # Cross thread boundary: UI renders trajectory immediately
            self.sig_nominal_done.emit(self._package_nominal(nominal, wind_nodes))

            # ── Mandatory stage boundary ──────────────────────────────────
            # sig_nominal_done is a queued signal: it was posted to the GUI
            # thread's event queue but has NOT been processed yet.  Without
            # an explicit yield the worker immediately calls _run_mc_loop,
            # which blocks on the first RocketPy call for several seconds.
            # During that time Python's GIL is held and the GUI thread cannot
            # drain its event queue, so the trajectory never appears until
            # the full MC loop finishes — defeating the two-stage UX.
            # msleep(0) releases the GIL and lets the event queue flush.
            QThread.msleep(0)

            if self._stop_event.is_set():
                self.finished.emit({"cancelled": True, "has_sim_result": False})
                return

            # ════════════════════════════════════════════════════════════════
            # STAGE 2 — Monte Carlo scatter + statistical analysis
            # Run n_runs perturbed flights → emit sig_progress_updated per
            # iteration → compute stats → emit sig_mc_done with MC payload
            # → emit finished with the full combined result.
            # ════════════════════════════════════════════════════════════════

            self.sig_status_text.emit("Running Monte Carlo...")
            scatter = self._run_mc_loop(sim_params, p)

            if self._stop_event.is_set():
                self.finished.emit({
                    "cancelled":      True,
                    "has_sim_result": False,
                    "impact_x":  nominal["impact_x"],
                    "impact_y":  nominal["impact_y"],
                    "apogee_m":  nominal["apogee_m"],
                    "hang_time": nominal["hang_time"],
                })
                return

            self.progress.emit(92)
            prob_pct = int(p.get("cep_prob", 90))
            stats    = self._compute_stats(scatter, prob_pct)
            self.progress.emit(98)

            # Emit MC-specific payload for dedicated statistics consumers
            self.sig_mc_done.emit(self._package_mc(scatter, stats, prob_pct))

            # Emit full combined result for the main controller
            result = self._package_result(
                nominal, scatter, stats, prob_pct, cancelled=False)
            result["nominal_surf_spd"] = float(p.get("surf_spd", 0.0))
            result["nominal_surf_dir"] = float(p.get("surf_dir", 0.0))
            result["wind_nodes"]       = wind_nodes
            self.finished.emit(result)

        except Exception:
            self.error.emit(traceback.format_exc())

    # ── Step helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_wind_profiles(
        p: dict,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Construct a smooth vertical wind profile from surface + upper obs.

        Data sources
        ------------
        Surface (obs_alt = 3 m):
            ``surf_spd`` / ``surf_dir`` — measured by the 自作風速計
            (custom on-site anemometer).  This is the only data point treated
            as ground truth; it anchors the profile at obs_alt.

        Upper wind (GPV levels, default 500 m AGL):
            ``up_spd`` / ``up_dir`` — fetched from the upper-wind API
            (JMA MSM / GPV).  The altitude is configurable via ``upper_alt``.

        The three-zone blending in create_wind_profile (surface ramp →
        blend zone 0–100 m → pure GPV above 100 m) smoothly connects the
        two independent data sources into a single continuous profile.
        """
        upper_alt: float = float(p.get("upper_alt", 500.0))

        # Upper-wind levels: API-derived GPV data
        gpv_levels = [
            (upper_alt, float(p.get("up_spd", 8.0)), float(p.get("up_dir", 90.0))),
        ]
        return create_wind_profile(
            gpv_levels=gpv_levels,
            # Surface truth from the on-site anemometer (自作風速計)
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
        """Merge UI params with the default rocket configuration.

        Any key present in *p* that also exists in _DEFAULT_ROCKET will be
        overridden; all other defaults are preserved.

        SI CONTRACT — all values passed to simulate_once MUST be in SI:
          Lengths   → metres  (m)    NOT centimetres
          Masses    → kilograms (kg) NOT grams
          Areas     → m²             NOT cm²
          Angles    → degrees        (elev, azi)
          Time      → seconds        (backfire_delay, para_lag, motor_burn_time)

        AppState stores all geometry in SI; _collect_params reads from AppState
        so the values in *p* are already SI by the time they reach here.
        No conversion is applied in this method — that is intentional.
        If altitudes displayed in the UI appear wrong (×100 or ÷100), the
        root cause will be in _collect_params or the AppState spinbox bindings,
        NOT here.
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
        # Override _DEFAULT_ROCKET keys with caller-supplied SI values.
        # Keys not present in p (e.g. thrust_data, motor_burn_time) retain
        # their _DEFAULT_ROCKET values unless a motor file has been loaded.
        for key in _DEFAULT_ROCKET:
            if key in p and key not in ("wind_u_prof", "wind_v_prof"):
                params[key] = p[key]
        return params

    @staticmethod
    def _package_nominal(nominal: dict, wind_nodes: list[dict]) -> dict:
        """Package the nominal trajectory for sig_nominal_done.

        Trajectory phases (phase-split arrays)
        ---------------------------------------
        ``phases["thrust"]``  — 推進: launch → motor burnout
        ``phases["coast"]``   — 滑空: burnout → parachute open (or end of flight)
        ``phases["chute"]``   — パラシュート: parachute open → landing
                                Empty lists when deployment time exceeds flight time.

        Key-event coordinates (x_m, y_m, z_m tuples)
        ----------------------------------------------
        ``events["burnout"]`` — 燃焼終了点: position at motor burnout
        ``events["apogee"]``  — 最高点: position at peak altitude
        ``events["chute"]``   — 開傘点: position at parachute opening; None if not deployed

        Wind nodes
        ----------
        ``wind_by_alt``       — dict keyed by altitude (float) for O(1) per-level lookup
                                e.g. payload["wind_by_alt"][3.0] → node dict for 3 m AGL
        ``wind_nodes``        — ordered list of all 5 nodes (3, 10, 150, 300, 600 m)
        ``surface_wind``      — alias for wind_nodes[0] (anemometer, 3 m)
        ``upper_wind_nodes``  — alias for wind_nodes[1:] (API levels)
        """
        def _to_list(v):
            return v.tolist() if hasattr(v, "tolist") else list(v)

        t = _to_list(nominal["t_vals"])
        x = _to_list(nominal["x_vals"])
        y = _to_list(nominal["y_vals"])
        z = _to_list(nominal["z_vals"])

        i_burn = nominal.get("idx_burnout", 0)
        i_apo  = nominal.get("apogee_idx",  len(t) - 1)
        i_para = nominal.get("idx_para",    -1)

        # Coast phase ends at parachute open (or the final sample if not deployed)
        coast_end = i_para if i_para != -1 else len(t) - 1

        phases = {
            "thrust": {
                "t": t[:i_burn + 1], "x": x[:i_burn + 1],
                "y": y[:i_burn + 1], "z": z[:i_burn + 1],
            },
            "coast": {
                "t": t[i_burn:coast_end + 1], "x": x[i_burn:coast_end + 1],
                "y": y[i_burn:coast_end + 1], "z": z[i_burn:coast_end + 1],
            },
            "chute": {
                "t": t[i_para:] if i_para != -1 else [],
                "x": x[i_para:] if i_para != -1 else [],
                "y": y[i_para:] if i_para != -1 else [],
                "z": z[i_para:] if i_para != -1 else [],
            },
        }

        events = {
            "burnout": (x[i_burn], y[i_burn], z[i_burn]),
            "apogee":  (x[i_apo],  y[i_apo],  z[i_apo]),
            "chute":   (x[i_para], y[i_para], z[i_para]) if i_para != -1 else None,
        }

        # Keyed by altitude float for O(1) per-level lookup in UI slots
        wind_by_alt = {node["alt_m"]: node for node in wind_nodes}

        return {
            "t_vals":     t,
            "x_vals":     x,
            "y_vals":     y,
            "z_vals":     z,
            "apogee_m":   float(nominal["apogee_m"]),
            "hang_time":  float(nominal["hang_time"]),
            "impact_x":   float(nominal["impact_x"]),
            "impact_y":   float(nominal["impact_y"]),
            "r_horiz":    float(nominal["r_horiz"]),
            # Phase-split trajectory arrays (推進 / 滑空 / パラシュート)
            "phases":     phases,
            # Key-event (x_m, y_m, z_m) tuples (燃焼終了点 / 最高点 / 開傘点)
            "events":     events,
            # Wind per altitude — O(1) dict + ordered list + convenience aliases
            "wind_by_alt":      wind_by_alt,
            "wind_nodes":       wind_nodes,
            "surface_wind":     wind_nodes[0] if wind_nodes else None,
            "upper_wind_nodes": wind_nodes[1:] if len(wind_nodes) > 1 else [],
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

        GIL note: RocketPy simulations hold Python's GIL for their duration.
        QThread.yieldCurrentThread() is called after every iteration to release
        the GIL and allow Qt's event queue to deliver queued signals (including
        sig_progress_updated) to the GUI thread before the next simulation starts.
        """
        n_total    = int(p.get("mc_runs",    50))
        wind_unc   = float(p.get("wind_unc",  0.20))
        thrust_unc = float(p.get("thrust_unc", 0.05))
        # _collect_params stores gust under "gust_speed"; accept both names so
        # callers that use "gust_intensity" (the run_mc_scatter parameter name)
        # also work.  Without this fix gust was silently 0.0 for every MC run.
        gust_intensity = float(p.get("gust_speed", p.get("gust_intensity", 0.0)))
        scatter: list[tuple[float, float]] = []

        for i in range(n_total):
            if self._stop_event.is_set():
                break

            # Run exactly 1 perturbed simulation per iteration so that
            # sig_progress_updated fires at every run, not every batch.
            batch_scatter, _ = run_mc_scatter(
                sim_params,
                1,                          # one run per iteration
                wind_unc,
                thrust_unc,
                gust_intensity=gust_intensity,
                stop_flag=self._stop_event,
            )
            scatter.extend(batch_scatter)

            # Yield the GIL so Qt can drain its event queue before the
            # next blocking simulation starts.  Without this the GUI thread
            # may not process the queued signal until all n_total runs finish.
            QThread.yieldCurrentThread()

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
    def _package_mc(
        scatter:  list[tuple[float, float]],
        stats:    dict,
        prob_pct: int,
    ) -> dict:
        """Package the MC statistics payload for sig_mc_done.

        Carries only the Monte Carlo outputs so consumers connected to
        sig_mc_done do not receive the full nominal trajectory arrays.
        """
        return {
            "scatter":      scatter,
            "r_N_radius":   float(stats.get("r_N_radius", 0.0)),
            "cep":          float(stats.get("cep", 0.0)),
            "ellipse":      stats.get("ellipse"),
            "cep_circle":   stats.get("cep_circle"),
            "kde_contours": stats.get("kde_contours", []),
            "n_runs":       len(scatter),
            "landing_prob": prob_pct,
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
