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

import numpy as np
import os
import random as _random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from PySide6.QtCore import QThread, Signal

from core.simulation   import simulate_once
from core.wind_model   import create_wind_profile, sample_wind_nodes
from core.monte_carlo  import (
    _perturb_wind_profile,
    compute_error_ellipse,
    compute_cep,
    compute_cep_circle,
    compute_kde_contours,
    compute_kde_grid,
)

# ── Airframe defaults (NO motor data) ────────────────────────────────────────
# Only airframe geometry and recovery system have defaults.
# Motor data (thrust_data, motor_burn_time) must ALWAYS be supplied by the
# caller via a loaded thrust-curve file — no silent fallback motor exists.
# If thrust_data is absent the worker raises RuntimeError before simulation.

_DEFAULT_AIRFRAME: dict[str, Any] = {
    "rail":             1.5,     # launch rail length (m)
    "airframe_mass":    0.0872,  # airframe dry mass (kg)
    "airframe_cg":      0.21,    # centre of gravity from nose (m)
    "airframe_len":     0.383,   # total airframe length (m)
    "radius":           0.015,   # body radius (m) — 30 mm diameter
    "nose_len":         0.08,    # nose cone length (m)
    "fin_root":         0.04,    # fin root chord (m)
    "fin_tip":          0.02,    # fin tip chord (m)
    "fin_span":         0.03,    # fin semi-span (m)
    "fin_pos":          0.35,    # fin leading-edge position from nose (m)
    "motor_pos":        0.38,    # motor CG from nose (m)
    "motor_dry_mass":   0.015,   # motor dry mass (kg)
    "backfire_delay":   4.0,     # ejection charge fires this many s after burnout
    "para_cd":          0.8,     # parachute drag coefficient
    "para_area":        0.126,   # parachute reference area (m²)
    "para_lag":         0.5,     # deployment lag (s)
    # NOTE: body_cd / motor_isp / motor_propellant_density used to live here
    # as Phase A fallback defaults.  They were superseded in Phase B by the
    # split power_on_cd / power_off_cd plus the new motor_isp and
    # motor_propellant_density properties on AppState (defaults: 0.45 / 0.40 /
    # 80.0 s / 1700 kg/m³).  The physics core in core/simulation.py also
    # provides its own ``params.get(..., default)`` fallbacks aligned with
    # those AppState defaults, so listing stale duplicates here only risks
    # silent drift if a future refactor reorders the merge.
}

# Number of MC runs per progress tick.  Smaller → finer progress granularity
# but slightly more overhead per RocketPy call.
_MC_BATCH_SIZE: int = 10


def _mc_worker_task(
    sim_params: dict,
    wind_unc: float,
    gust_sigma: float,
    tu: float,
    raw_thrust: list,
    elev: float,
    azi: float,
    base_u: list,
    base_v: list,
    flight_mode: str,
    target_radius: float,
    seed: int,
    trial_idx: int = 0,
) -> dict | None:
    """
    Top-level, picklable worker function for ProcessPoolExecutor.
    Runs a single perturbed Monte Carlo simulation and returns purely scalar values.
    The Flight array data is aggressively discarded to keep memory O(1).
    """
    rng = _random.Random(seed)

    # ── Wind perturbation
    u_prof, v_prof, _ = _perturb_wind_profile(
        base_u, base_v, rng,
        wind_unc, gust_intensity=gust_sigma,
    )

    # ── Thrust perturbation
    thrust_scale = max(0.1, 1.0 + rng.gauss(0.0, tu))
    perturbed_thrust = [[t, T * thrust_scale] for t, T in raw_thrust]

    trial_p = {
        **sim_params,
        "wind_u_prof": u_prof,
        "wind_v_prof": v_prof,
        "thrust_data": perturbed_thrust,
    }

    r = simulate_once(elev, azi, trial_p, trial_idx=trial_idx)

    if r["ok"]:
        from core.optimization import p1_objective_score
        score = p1_objective_score(r, flight_mode, target_radius)

        # Only return the scalars
        res = {
            "x": float(r["impact_x"]),
            "y": float(r["impact_y"]),
            "apogee": float(r["apogee_m"]),
            "hang_time": float(r["hang_time"]),
            "score": score
        }
    else:
        res = None

    # Memory optimization: forcefully garbage collect / remove refs
    del r, trial_p, u_prof, v_prof, perturbed_thrust

    return res


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
    sig_finished         = Signal(dict)       # always emitted last — check result["cancelled"]
    error                = Signal(str)        # only on unhandled exception
    sig_nominal_done     = Signal(dict)       # Stage 1: emitted after nominal run
    sig_progress         = Signal(int, int, str)   # Stage 2: (completed, total, msg)
    sig_mc_done          = Signal(dict)       # Stage 2: emitted after MC loop + statistics
    sig_status_text      = Signal(str)        # Human-readable stage label for the status bar
    sig_early_warning    = Signal(str)        # Nominal pre-eval: emitted before MC if out of target

    def __init__(
        self,
        params: dict[str, Any],
        parent=None,
        *,
        app_state: "object | None" = None,
    ) -> None:
        super().__init__(parent)
        self._params     = dict(params)
        self._stop_event = threading.Event()
        # Optional reference to the global AppState.  Used read-only from
        # within ``run()`` to pull MoI / late-binding parameters into the
        # simulation params dict.  ``None`` means "no AppState wired" — the
        # physics core then falls back to its built-in approximations.
        self._app_state = app_state

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

            # ── Inject accurate Moment of Inertia from AppState ──────────────
            # ``core/geometry_math.py`` computes (Ixx, Iyy, Izz) when the
            # operator loads an .rkt file and stores them in AppState via
            # ``set_moi()``.  Forward those values to the physics core so it
            # stops falling back to the solid-cylinder approximation.
            # Map:
            #   AppState.moi_roll  (Ixx, longitudinal)  → params['I_z']
            #   AppState.moi_pitch (Iyy, lateral)       → params['I_xy']
            # Skip injection when AppState is absent or MoI is still zero
            # (no .rkt loaded yet) — the Phase A fallback then takes over.
            if self._app_state is not None:
                _ixx = float(getattr(self._app_state, "moi_roll",  0.0))
                _iyy = float(getattr(self._app_state, "moi_pitch", 0.0))
                if _ixx > 0.0 and _iyy > 0.0:
                    sim_params["I_z"]  = _ixx
                    sim_params["I_xy"] = _iyy

                # ── Forward Mach-dependent Cd curves (Phase C) ───────────────
                # ``None`` means "no curve loaded" → simulate_once falls back
                # to the scalar power_on_cd / power_off_cd values.  A list of
                # ``(Mach, Cd)`` tuples is consumed directly by RocketPy.
                sim_params["cd_curve_power_on"]  = getattr(
                    self._app_state, "cd_curve_power_on",  None)
                sim_params["cd_curve_power_off"] = getattr(
                    self._app_state, "cd_curve_power_off", None)

            self.progress.emit(10)
            self.sig_status_text.emit("Simulating...")
            nominal = simulate_once(
                elev=float(p.get("elev", 85.0)),
                azi=float(p.get("azim",  0.0)),
                params=sim_params,
            )
            if not nominal["ok"]:
                raise RuntimeError(f"Nominal simulation failed: {nominal['error']}")
            self.progress.emit(25)

            # Build and augment the nominal payload with turbulence stats so
            # the UI can lock the Phase B baseline immediately on receipt.
            nom_pkg = self._package_nominal(nominal, wind_nodes)
            nom_pkg.update({
                "turb_mu":        float(p.get("turb_mu",        0.0)),
                "turb_sigma":     float(p.get("turb_sigma",     0.0)),
                "turb_intensity": float(p.get("turb_intensity", 0.0)),
            })

            # Cross thread boundary: UI renders trajectory immediately
            self.sig_nominal_done.emit(nom_pkg)

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

            # ── Nominal pre-evaluation — early NO-GO warning ───────────────────
            # Compare the deterministic nominal landing distance against
            # target_radius so the operator gets an immediate red flag before
            # the full MC scatter is even started.
            # IMPORTANT: this block NEVER returns — the MC loop is always
            # executed so the operator has complete statistical evidence even
            # for a NO-GO result (forced continuation per mission requirement).
            # In Free Mode the rocket may land anywhere — NO-GO is meaningless.
            # Also skip if target_radius was not provided (None).
            _is_free  = bool(p.get("is_free_mode", False))
            _target_r = p.get("target_radius")
            if not _is_free and _target_r is not None:
                _nom_dist = math.hypot(
                    float(nom_pkg["impact_x"]),
                    float(nom_pkg["impact_y"]),
                )
                if _nom_dist > float(_target_r):
                    self.sig_early_warning.emit(
                        f"NO-GO (OUT OF TARGET)  —  nominal impact "
                        f"{_nom_dist:.0f} m  >  {float(_target_r):.0f} m radius"
                    )

            if self._stop_event.is_set():
                self.sig_finished.emit({
                    "cancelled":      True,
                    "has_sim_result": False,
                    "impact_x":  nom_pkg["impact_x"],
                    "impact_y":  nom_pkg["impact_y"],
                    "apogee_m":  nom_pkg["apogee_m"],
                    "hang_time": nom_pkg["hang_time"],
                })
                return

            # ════════════════════════════════════════════════════════════════
            # STAGE 2 — Monte Carlo scatter + statistical analysis
            # Run n_runs perturbed flights → emit sig_progress_updated per
            # iteration → compute stats → emit sig_mc_done with MC payload
            # → emit finished with the full combined result.
            # ════════════════════════════════════════════════════════════════

            self.sig_status_text.emit("Monte Carlo...")
            scatter = self._run_mc_loop(sim_params, p)

            if self._stop_event.is_set():
                self.sig_finished.emit({
                    "cancelled":      True,
                    "has_sim_result": False,
                    "impact_x":  nom_pkg["impact_x"],
                    "impact_y":  nom_pkg["impact_y"],
                    "apogee_m":  nom_pkg["apogee_m"],
                    "hang_time": nom_pkg["hang_time"],
                })
                return

            # ════════════════════════════════════════════════════════════════
            # STAGE 2b — Statistical analysis — ALL on this background thread
            #
            # Every scipy / numpy / matplotlib call below is intentionally
            # placed here, in the worker's run() method, so the GUI thread
            # is never blocked by heavy math.
            #
            # Execution order and GIL notes:
            #   92 %  r_N / cov_matrix — pure Python + numpy (fast, ~10 ms)
            #   93 %  error ellipse    — numpy eigendecomposition (~20 ms)
            #   95 %  KDE contours     — scipy gaussian_kde + matplotlib
            #                            contour extraction (HEAVY: 1–3 s)
            #   97 %  KDE grid         — scipy gaussian_kde grid eval (HEAVY: ~1 s)
            #   98 %  payload assembly
            #
            # yieldCurrentThread() is called before each heavy step so the OS
            # scheduler can give the GUI thread a timeslice to paint and process
            # input events between the long-running C-extension calls.
            # ════════════════════════════════════════════════════════════════

            prob_pct = int(p.get("cep_prob", 90))
            n_pts    = len(scatter)

            # ── Percentile radius + covariance (fast) ────────────────────────
            self.progress.emit(92)
            self.sig_status_text.emit("Computing landing statistics...")

            if n_pts > 0:
                # Handle list of dicts or list of tuples
                if scatter and isinstance(scatter[0], dict):
                    pts_tuples = [(pt["x"], pt["y"]) for pt in scatter]
                    scores = [pt["score"] for pt in scatter if pt["score"] != float('-inf')]
                    avg_score = sum(scores) / len(scores) if scores else float('-inf')
                    min_score = min(scores) if scores else float('-inf')
                    avg_alt = sum(pt["apogee"] for pt in scatter) / n_pts
                    avg_hang = sum(pt["hang_time"] for pt in scatter) / n_pts
                else:
                    pts_tuples = scatter
                    avg_score = float('-inf')
                    min_score = float('-inf')
                    avg_alt = 0.0
                    avg_hang = 0.0

                _cx = sum(x for x, _ in pts_tuples) / n_pts
                _cy = sum(y for _, y in pts_tuples) / n_pts
                _radii = sorted(math.hypot(x - _cx, y - _cy) for x, y in pts_tuples)
                _idx_n = min(int(prob_pct / 100.0 * n_pts), n_pts - 1)
                r_N    = _radii[_idx_n]
                cep_val   = compute_cep(pts_tuples)
                cov_matrix = None
                if n_pts >= 4:
                    _arr = np.array(pts_tuples, dtype=np.float64)
                    cov_matrix = np.cov(_arr[:, 0], _arr[:, 1]).tolist()
            else:
                r_N = 0.0
                cep_val    = 0.0
                cov_matrix = None
                pts_tuples = []
                avg_score = float('-inf')
                min_score = float('-inf')
                avg_alt = 0.0
                avg_hang = 0.0

            # ── Error ellipse — numpy eigendecomposition ─────────────────────
            # Yield before so the GUI can paint the 92% bar before we enter
            # the numpy call that may briefly hold the GIL.
            QThread.yieldCurrentThread()
            self.progress.emit(93)
            self.sig_status_text.emit("Fitting error ellipse...")
            ellipse    = compute_error_ellipse(pts_tuples, prob_pct=prob_pct)


            # ── KDE contours — scipy + matplotlib (HEAVY) ────────────────────
            # Yield before: allows the OS to switch to the GUI thread so the
            # status bar ("Fitting error ellipse...") is visible before the
            # 1-3 s scipy call starts and the GIL is intermittently held.
            QThread.yieldCurrentThread()
            self.progress.emit(95)
            self.sig_status_text.emit("Computing KDE contours...")
            kde_contours = compute_kde_contours(pts_tuples, conf_pct=prob_pct)

            # ── KDE density grid — scipy (HEAVY) ─────────────────────────────
            QThread.yieldCurrentThread()
            self.progress.emit(97)
            self.sig_status_text.emit("Computing KDE density grid...")
            kde_grid = compute_kde_grid(pts_tuples)

            # ── Assemble stats dict (all pre-computed above) ──────────────────
            stats = {
                "r_N_radius":   r_N,
                "cep":          cep_val,
                "ellipse":      ellipse,
                "cep_circle":   compute_cep_circle(pts_tuples),
                "kde_contours": kde_contours,
                "kde":          kde_grid,
                "cov_matrix":   cov_matrix,
                "mc_avg_score": avg_score,
                "mc_min_score": min_score,
                "mc_avg_alt":   avg_alt,
                "mc_avg_hang_time": avg_hang,
            }

            self.progress.emit(98)
            self.sig_status_text.emit("Packaging results...")

            # Emit MC-specific payload for dedicated statistics consumers
            self.sig_mc_done.emit(self._package_mc(scatter, stats, prob_pct))

            # Emit single combined payload (nominal + MC + turbulence baseline)
            result = self._package_result(nom_pkg, scatter, stats, prob_pct, cancelled=False)
            result["nominal_surf_spd"] = float(p.get("surf_spd", 0.0))
            result["nominal_surf_dir"] = float(p.get("surf_dir", 0.0))
            self.sig_finished.emit(result)

        except Exception as _exc:
            if self._stop_event.is_set() or str(_exc) == 'cancelled':
                self.sig_finished.emit({"cancelled": True})
            else:
                self.error.emit(traceback.format_exc())

    # ── Step helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_wind_profiles(
        p: dict,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        wind_profile_data = p.get("wind_profile_data", [])
        return create_wind_profile(wind_profile_data)

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
        # Motor data is mandatory — no silent fallback.
        thrust_data = p.get("thrust_data")
        if not thrust_data:
            raise RuntimeError(
                "モーターがロードされていません。\n"
                "「推力曲線を読み込む」ボタンでスラストカーブ CSV を\n"
                "読み込んでからシミュレーションを実行してください。\n"
                "(No motor loaded: thrust_data is empty. "
                "Load a thrust-curve file before running.)"
            )
        if not p.get("motor_burn_time"):
            raise RuntimeError(
                "motor_burn_time が指定されていません。"
                "モーターファイルを再読み込みしてください。"
            )

        # Base: airframe defaults; caller values override on a per-key basis.
        # None values are skipped — they preserve the valid _DEFAULT_AIRFRAME value
        # rather than clobbering it (AppState returns None for fields not yet set).
        params = dict(_DEFAULT_AIRFRAME)
        for key, val in p.items():
            if key not in ("wind_u_prof", "wind_v_prof") and val is not None:
                params[key] = val
        # Wind profiles always come from the freshly built profile, not p
        params["wind_u_prof"] = u_prof
        params["wind_v_prof"] = v_prof
        # Ensure launch coords are float
        params["launch_lat"] = float(p.get("launch_lat", 35.0))
        params["launch_lon"] = float(p.get("launch_lon", 135.0))
        params["elev"]       = float(p.get("elev",       85.0))
        params["azi"]        = float(p.get("azim",        0.0))
        SimulationWorker._validate_si_units(params)
        return params

    @staticmethod
    def _validate_si_units(params: dict) -> None:
        """Raise ValueError if obvious CGS values are detected in the params dict.

        Thresholds are chosen so that any physically plausible competition
        rocket in SI will pass, while CGS leaks (e.g. radius in cm instead
        of metres) produce an immediate, actionable error before RocketPy
        receives nonsense inputs.

        All spinboxes display MKS values directly; AppState stores MKS; no
        unit conversion is applied anywhere in the chain.
        """
        limits = [
            ("radius",         params.get("radius",         0), 5.0,   "m   — expected < 5 m   (was cm passed?)"),
            ("airframe_len",   params.get("airframe_len",   0), 50.0,  "m   — expected < 50 m  (was cm passed?)"),
            ("nose_len",       params.get("nose_len",       0), 20.0,  "m   — expected < 20 m  (was cm passed?)"),
            ("airframe_mass",  params.get("airframe_mass",  0), 500.0, "kg  — expected < 500 kg (was g passed?)"),
            ("motor_dry_mass", params.get("motor_dry_mass", 0), 50.0,  "kg  — expected < 50 kg  (was g passed?)"),
            ("para_area",      params.get("para_area",      0), 100.0, "m²  — expected < 100 m² (was cm² passed?)"),
        ]
        problems = [
            f"  {name} = {val:.4g}  {unit}"
            for name, val, limit, unit in limits
            if val > limit
        ]
        if problems:
            raise ValueError(
                "SI unit validation failed — CGS leak suspected in sim params:\n"
                + "\n".join(problems)
            )

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
        """Run the MC scatter inline — O(1) peak memory per iteration.

        The perturbation and simulation are inlined directly rather than
        delegated to run_mc_scatter.  This eliminates three per-call
        overheads that compound at large n_runs:
          - no fresh Random() instance per call
          - no wind_profiles accumulation list
          - simulate_once result dict (full trajectory arrays) is explicitly
            deleted at the end of each iteration via `del r`, so Python's
            reference-counter frees the arrays before the next run starts
            rather than waiting for the next GC cycle.

        Only the two scalar landing coordinates are retained per run —
        the scatter list grows O(n_runs) in scalar tuples (~40 bytes each).

        sig_progress_updated(i+1, n_total) fires after every iteration.
        Progress moves from 25 % to 90 % as runs complete.
        """
        import random as _random

        n_total      = int(p.get("mc_runs",    50))
        wind_unc     = float(p.get("wind_unc",  0.20))
        thrust_unc   = float(p.get("thrust_unc", 0.05))
        # Gust noise priority:
        #   1. auto_gust_ms: σ computed from the 60s surface wind history buffer
        #   2. gust_speed / gust_intensity: manual UI value, fallback only.
        auto_gust    = float(p.get("auto_gust_ms", 0.0))
        manual_gust  = float(p.get("gust_speed", p.get("gust_intensity", 0.0)))
        gust_sigma   = auto_gust if auto_gust > 0.0 else manual_gust

        # Extract sim_params invariants once — avoids repeated dict lookup
        tu         = max(thrust_unc, 0.0)
        raw_thrust = sim_params["thrust_data"]
        elev       = sim_params["elev"]
        azi        = sim_params["azi"]
        base_u: list = sim_params.get("wind_u_prof", [])
        base_v: list = sim_params.get("wind_v_prof", [])
        flight_mode = p.get("flight_mode", "Altitude Competition")
        target_radius = p.get("target_radius", float('inf'))

        scatter: list[tuple[float, float]] = []

        # Throttle progress signal emissions to at most _MAX_PROG_SIGNALS
        # cross-thread events regardless of n_total.
        _MAX_PROG_SIGNALS = 20
        _emit_every = max(1, n_total // _MAX_PROG_SIGNALS)

        with open("mc_diagnostics.log", "w", encoding="utf-8") as f:
            f.write("--- Kazamidori Diagnostics Log ---\n")

        # ── ProcessPoolExecutor for CPU-bound Monte Carlo ─────────────────────
        # max_workers=os.cpu_count() fully utilizes available CPU cores,
        # completely bypassing the Python GIL for heavy RocketPy math.
        from itertools import repeat

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Start random seed block to avoid generating exactly the same stream
            # of random numbers in each process if we were not passing seeds.
            base_seed = _random.randint(0, 2**31 - 1)

            chunksize = max(1, n_total // (os.cpu_count() * 4))

            # Gather results as they complete
            for _done, res in enumerate(executor.map(
                _mc_worker_task,
                repeat(sim_params, n_total),
                repeat(wind_unc, n_total),
                repeat(gust_sigma, n_total),
                repeat(tu, n_total),
                repeat(raw_thrust, n_total),
                repeat(elev, n_total),
                repeat(azi, n_total),
                repeat(base_u, n_total),
                repeat(base_v, n_total),
                repeat(flight_mode, n_total),
                repeat(target_radius, n_total),
                range(base_seed, base_seed + n_total),
                range(1, n_total + 1),
                chunksize=chunksize
            ), start=1):
                if self._stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                if res is not None:
                    scatter.append(res)

                # ── Yield every iteration — scheduler fairness ────────────────────
                # yieldCurrentThread() lets the OS give the GUI thread a timeslice
                # to process any already-queued signals (e.g. repaint the progress
                # bar from the PREVIOUS emission) without flooding the queue with
                # a new signal on this iteration.
                QThread.yieldCurrentThread()

                # Emit progress signals at most _MAX_PROG_SIGNALS times total.
                # Always emit on the final iteration so the bar reaches 90%.
                if _done % _emit_every == 0 or _done == n_total:
                    self.sig_progress.emit(_done, n_total, f"Phase 2: Monte Carlo Simulation ({_done}/{n_total})")
                    self.progress.emit(min(25 + int(_done / n_total * 65), 90))

        return scatter

    @staticmethod
    def _compute_stats(
        scatter: list[tuple[float, float]],
        prob_pct: int,
    ) -> dict:
        """Compute all statistical outputs from the landing scatter.

        The production code path inlines each sub-step directly in run() with
        explicit status text, progress milestones, and yieldCurrentThread()
        calls between heavy scipy operations.  This helper exists for unit
        tests and ad-hoc calls where those GUI-side side effects are unwanted.

        Returns a dict with keys: r_N_radius, cep, ellipse, cep_circle,
        kde_contours, kde, cov_matrix.
        """
        empty = {
            "r_N_radius":   0.0,
            "cep":          0.0,
            "ellipse":      None,

            "kde_contours": [],
            "kde":          None,
            "cov_matrix":   None,
        }
        if not scatter:
            return empty

        # Percentile radius from centroid (= CEP at prob_pct %)
        if scatter and isinstance(scatter[0], dict):
            pts = [(pt["x"], pt["y"]) for pt in scatter]
            scores = [pt["score"] for pt in scatter if pt["score"] != float('-inf')]
            avg_score = sum(scores) / len(scores) if scores else float('-inf')
            min_score = min(scores) if scores else float('-inf')
            avg_alt = sum(pt["apogee"] for pt in scatter) / len(scatter)
            avg_hang = sum(pt["hang_time"] for pt in scatter) / len(scatter)
        else:
            pts = scatter
            avg_score = float('-inf')
            min_score = float('-inf')
            avg_alt = 0.0
            avg_hang = 0.0

        cx = sum(x for x, y in pts) / len(pts)
        cy = sum(y for x, y in pts) / len(pts)
        radii = sorted(math.hypot(x - cx, y - cy) for x, y in pts)
        n    = len(radii)
        idx  = min(int(prob_pct / 100.0 * n), n - 1)
        r_N  = radii[idx]

        # 2×2 sample covariance matrix (metres²).  None when fewer than 4 points.
        cov_matrix = None
        if len(scatter) >= 4:
            arr = np.array(pts, dtype=float)
            cov_matrix = np.cov(arr[:, 0], arr[:, 1]).tolist()

        # KDE grid: raw density field for heatmap / custom contours in the UI.
        # compute_kde_grid returns Python lists of lists — Qt signal safe.
        kde_grid = compute_kde_grid(pts)

        return {
            "r_N_radius":   r_N,
            "cep":          compute_cep(pts),
            # ellipse carries both math keys and UI-ready keys:
            #   math: cx, cy, a, b, angle_rad
            #   UI:   x, y, width (=2a), height (=2b), angle_deg
            "ellipse":      compute_error_ellipse(pts, prob_pct=prob_pct),

            "kde_contours": compute_kde_contours(pts, conf_pct=prob_pct),
            # kde: {X_m, Y_m, Z, x_min_m, x_max_m, y_min_m, y_max_m}
            # All arrays are Python lists of lists — ready for heatmap rendering.
            "kde":          kde_grid,
            "cov_matrix":   cov_matrix,
            "mc_avg_score": avg_score,
            "mc_min_score": min_score,
            "mc_avg_alt":   avg_alt,
            "mc_avg_hang_time": avg_hang,
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
        pts = [(pt["x"], pt["y"]) for pt in scatter] if scatter and isinstance(scatter[0], dict) else scatter
        return {
            # ── Canonical unified payload keys ────────────────────────────────
            "scatter_points": pts,   # list[(x_east_m, y_north_m)] — canonical name
            # ellipse: covariance error ellipse at prob_pct.
            #   Math keys:  cx, cy, a (semi-major m), b (semi-minor m), angle_rad
            #   UI keys:    x, y, width (=2a m), height (=2b m), angle_deg
            "ellipse":        stats.get("ellipse"),
            # kde: raw density grid for heatmap / client-side contour rendering.
            #   Keys: X_m, Y_m, Z  (all list[list[float]]), plus bounding scalars.
            "kde":            stats.get("kde"),
            # ── MC statistics ─────────────────────────────────────────────────
            "r_N_radius":   float(stats.get("r_N_radius", 0.0)),
            "cep_radius":   float(stats.get("r_N_radius", 0.0)),
            "cep":          float(stats.get("cep", 0.0)),

            "kde_contours": stats.get("kde_contours", []),
            "cov_matrix":   stats.get("cov_matrix"),
            "n_runs":       len(scatter),
            "landing_prob": prob_pct,
            "mc_avg_score": stats.get("mc_avg_score", 0.0),
            "mc_min_score": stats.get("mc_min_score", 0.0),
            "mc_avg_alt":   stats.get("mc_avg_alt", 0.0),
            "mc_avg_hang_time": stats.get("mc_avg_hang_time", 0.0),
            # ── Backward-compat alias ─────────────────────────────────────────
            "scatter":      pts,
        }

    @staticmethod
    def _package_result(
        nom_pkg:   dict,
        scatter:   list,
        stats:     dict,
        prob_pct:  int,
        *,
        cancelled: bool,
    ) -> dict:
        pts = [(pt["x"], pt["y"]) for pt in scatter] if scatter and isinstance(scatter[0], dict) else scatter
        """Assemble the single finished-signal payload.

        Takes the pre-packaged nominal payload (nom_pkg, already containing
        list-converted arrays, phases, events, wind_by_alt, and turbulence
        stats) and merges in the MC statistics so the finished signal carries
        the complete simulation record in one dict.

        All arrays are Python lists (not numpy) — safe for Qt queued signals
        and JSON serialisation.
        """
        t_vals = nom_pkg["t_vals"]
        x_vals = nom_pkg["x_vals"]
        y_vals = nom_pkg["y_vals"]
        z_vals = nom_pkg["z_vals"]

        return {
            "cancelled":      cancelled,
            "has_sim_result": not cancelled,

            # ── Canonical unified payload (structured, matches sig_mc_done) ───
            # trajectory: full nominal flight data grouped under one key
            "trajectory": {
                "t_vals":    t_vals,
                "x_vals":    x_vals,
                "y_vals":    y_vals,
                "z_vals":    z_vals,
                "apogee_m":  nom_pkg["apogee_m"],
                "hang_time": nom_pkg["hang_time"],
                "impact_x":  nom_pkg["impact_x"],
                "impact_y":  nom_pkg["impact_y"],
                "r_horiz":   nom_pkg["r_horiz"],
                "phases":    nom_pkg.get("phases", {}),
                "events":    nom_pkg.get("events", {}),
            },
            # scatter_points: canonical MC landing positions (x_east_m, y_north_m)
            "scatter_points": pts,
            # ellipse: covariance error ellipse — math keys + UI keys (see _package_mc)
            "ellipse":        stats.get("ellipse"),
            # kde: raw density grid {X_m, Y_m, Z, bounds} for heatmap rendering
            "kde":            stats.get("kde"),

            # ── Flat nominal keys (backward compat — direct field access) ─────
            "t_vals":    t_vals,
            "x_vals":    x_vals,
            "y_vals":    y_vals,
            "z_vals":    z_vals,
            "apogee_m":  nom_pkg["apogee_m"],
            "hang_time": nom_pkg["hang_time"],
            "impact_x":  nom_pkg["impact_x"],
            "impact_y":  nom_pkg["impact_y"],
            "r_horiz":   nom_pkg["r_horiz"],
            "phases":    nom_pkg.get("phases", {}),
            "events":    nom_pkg.get("events", {}),
            # ── Wind data (all 5 diagnostic altitudes) ────────────────────────
            "wind_nodes":       nom_pkg.get("wind_nodes",       []),
            "wind_by_alt":      nom_pkg.get("wind_by_alt",      {}),
            "surface_wind":     nom_pkg.get("surface_wind"),
            "upper_wind_nodes": nom_pkg.get("upper_wind_nodes", []),
            # ── Phase A turbulence baseline (locked for Phase B GO/NO-GO) ─────
            "turb_mu":        nom_pkg.get("turb_mu",        0.0),
            "turb_sigma":     nom_pkg.get("turb_sigma",     0.0),
            "turb_intensity": nom_pkg.get("turb_intensity", 0.0),
            # ── MC statistics (local metric frame, origin at launch) ───────────
            "scatter":      pts,   # backward-compat alias for scatter_points
            "r_N_radius":   float(stats.get("r_N_radius", 0.0)),
            "cep_radius":   float(stats.get("r_N_radius", 0.0)),
            "cep":          float(stats.get("cep",         0.0)),

            "kde_contours": stats.get("kde_contours", []),
            "kde":          stats.get("kde"),
            "cov_matrix":   stats.get("cov_matrix"),
            "n_runs":       len(scatter),
            "landing_prob": prob_pct,
            "mc_avg_score": stats.get("mc_avg_score", 0.0),
            "mc_min_score": stats.get("mc_min_score", 0.0),
            "mc_avg_alt":   stats.get("mc_avg_alt", 0.0),
            "mc_avg_hang_time": stats.get("mc_avg_hang_time", 0.0),
            # ── Legacy alias keys ─────────────────────────────────────────────
            "trajectory_3d":     list(zip(x_vals, y_vals, z_vals)),
            "mc_scatter_points": scatter,
            "apogee":            nom_pkg["apogee_m"],
            "impact_distance":   nom_pkg["r_horiz"],
        }



class MapDownloadWorker(QThread):
    """
    Executes the map download in a background thread to prevent UI freezing.
    """
    sig_progress = Signal(int, int, str)
    sig_finished = Signal(dict)
    error = Signal(str)

    def __init__(self, lat: float, lon: float, parent=None):
        super().__init__(parent)
        self.lat = lat
        self.lon = lon

    def run(self) -> None:
        try:
            import os
            import subprocess
            import json

            self.sig_progress.emit(10, 100, "Starting map download...")

            # Using run_in_bash_session style via subprocess to call our CLI tool natively
            import sys
            cmd = [sys.executable, "utils/map_downloader.py", "--lat", str(self.lat), "--lon", str(self.lon)]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                if "Downloaded" in line:
                    self.sig_progress.emit(50, 100, "Downloading tiles...")
                elif "Saved stitched" in line:
                    self.sig_progress.emit(90, 100, "Stitching map...")
            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Map downloader script failed with return code {process.returncode}")

            meta_path = "assets/offline_map/map_meta.json"
            if not os.path.exists(meta_path):
                raise RuntimeError("Map download succeeded but metadata file is missing.")

            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.sig_progress.emit(100, 100, "Download complete")
            self.sig_finished.emit(meta)
        except Exception as _exc:
            import traceback
            self.error.emit(traceback.format_exc())

class OptimizationWorker(QThread):
    """
    Executes the Phase 1 Coarse Search optimization + full Monte Carlo scatter
    seamlessly on a background thread.
    """

    progress             = Signal(int)
    sig_finished         = Signal(dict)
    error                = Signal(str)
    sig_nominal_done     = Signal(dict)
    sig_progress         = Signal(int, int, str)
    sig_mc_done          = Signal(dict)
    sig_status_text      = Signal(str)
    sig_early_warning    = Signal(str)
    sig_optimization_done = Signal(float, float) # elev, azi

    def __init__(
        self,
        params: dict[str, Any],
        parent=None,
        *,
        app_state: "object | None" = None,
    ) -> None:
        super().__init__(parent)
        self._params     = dict(params)
        self._stop_event = threading.Event()
        # Same read-only AppState reference contract as SimulationWorker.
        # Used to forward MoI / Cd / motor parameters into the params dict
        # that feeds optimize_launch_angle and the subsequent MC pipeline.
        self._app_state = app_state

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            p = self._params

            # --- 1. Optimization phase ---
            self.progress.emit(2)
            self.sig_status_text.emit("Optimising launch angle...")
            u_prof, v_prof = SimulationWorker._build_wind_profiles(p)
            wind_nodes = sample_wind_nodes(u_prof, v_prof)
            sim_params = SimulationWorker._build_sim_params(p, u_prof, v_prof)

            # ── Phase D: forward advanced parameters from AppState ───────────
            # ``optimize_launch_angle`` performs hundreds of ``simulate_once``
            # evaluations; if we skip this step the optimiser silently uses
            # static defaults and the chosen (elev, azi) drifts away from
            # the operating point that the live UI thinks it is using.
            # Inject everything Phase B/C exposed: precise MoI, split Cds,
            # motor chemistry, and Mach-dependent drag curves.  All keys are
            # preserved through ``dict(base_params)`` copies inside
            # core/optimization.py (p1_params_at_wind, p1_mc_points, ...).
            if self._app_state is not None:
                _ixx = float(getattr(self._app_state, "moi_roll",  0.0))
                _iyy = float(getattr(self._app_state, "moi_pitch", 0.0))
                if _ixx > 0.0 and _iyy > 0.0:
                    sim_params["I_z"]  = _ixx
                    sim_params["I_xy"] = _iyy

                sim_params["power_on_cd"]   = float(getattr(
                    self._app_state, "power_on_cd",  0.45))
                sim_params["power_off_cd"]  = float(getattr(
                    self._app_state, "power_off_cd", 0.40))
                sim_params["motor_isp"]     = float(getattr(
                    self._app_state, "motor_isp", 80.0))
                sim_params["motor_propellant_density"] = float(getattr(
                    self._app_state, "motor_propellant_density", 1700.0))

                sim_params["cd_curve_power_on"]  = getattr(
                    self._app_state, "cd_curve_power_on",  None)
                sim_params["cd_curve_power_off"] = getattr(
                    self._app_state, "cd_curve_power_off", None)

            # Important parameter routing
            target_r = p.get("target_radius", 50.0)
            mode = p.get("flight_mode", "Altitude Competition")

            from core.optimization import optimize_launch_angle
            from core.simulation import simulate_once

            def prog_cb(msg: str, frac: float):
                # core.optimization calls back with (msg, frac in 0..1).
                # sig_progress carries (completed, total, msg), so map the
                # fraction onto a /100 scale that _on_progress_updated converts
                # straight to a percent without surprises.
                self.sig_progress.emit(int(frac * 100), 100, msg)

            opt_res = optimize_launch_angle(
                mode=mode,
                base_params=sim_params,
                r_max=target_r,
                landing_prob=int(p.get("cep_prob", 90)),
                wind_uncertainty=float(p.get("wind_unc", 0.20)),
                thrust_uncertainty=float(p.get("thrust_unc", 0.05)),
                stop_flag=self._stop_event,
                progress_cb=prog_cb
            )

            if self._stop_event.is_set():
                self.sig_finished.emit({"cancelled": True})
                return

            best_e = opt_res["elev"]
            best_a = opt_res["azi"]

            # Update params with optimized values so MC phase uses them
            self.sig_optimization_done.emit(best_e, best_a)
            sim_params["elev"] = best_e
            sim_params["azi"] = best_a
            p["elev"] = best_e
            p["azim"] = best_a

            self.progress.emit(20)

            # --- 2. Full simulation pipeline (mimic SimulationWorker) ---
            self.sig_status_text.emit("Simulating...")
            nominal = simulate_once(best_e, best_a, sim_params)
            if not nominal["ok"]:
                raise RuntimeError(f"Nominal simulation failed: {nominal['error']}")

            self.progress.emit(25)

            nom_pkg = SimulationWorker._package_nominal(nominal, wind_nodes)
            nom_pkg.update({
                "turb_mu":        float(p.get("turb_mu",        0.0)),
                "turb_sigma":     float(p.get("turb_sigma",     0.0)),
                "turb_intensity": float(p.get("turb_intensity", 0.0)),
            })
            self.sig_nominal_done.emit(nom_pkg)
            QThread.msleep(0)

            _is_free  = bool(p.get("is_free_mode", False))
            _target_r = p.get("target_radius")
            if not _is_free and _target_r is not None:
                _nom_dist = math.hypot(
                    float(nom_pkg["impact_x"]),
                    float(nom_pkg["impact_y"]),
                )
                if _nom_dist > float(_target_r):
                    self.sig_early_warning.emit(
                        f"NO-GO (OUT OF TARGET)  —  nominal impact "
                        f"{_nom_dist:.0f} m  >  {float(_target_r):.0f} m radius"
                    )

            if self._stop_event.is_set():
                self.sig_finished.emit({"cancelled": True})
                return

            self.sig_status_text.emit("Monte Carlo...")

            # Since SimulationWorker._run_mc_loop is an instance method, we can create a dummy
            # instance or just use a helper. But it's easier to create a dummy worker here:
            dummy_worker = SimulationWorker(p)
            dummy_worker._stop_event = self._stop_event
            dummy_worker.progress.connect(self.progress.emit)
            dummy_worker.sig_progress.connect(self.sig_progress.emit)

            scatter = dummy_worker._run_mc_loop(sim_params, p)

            if self._stop_event.is_set():
                self.sig_finished.emit({"cancelled": True})
                return

            prob_pct = int(p.get("cep_prob", 90))
            n_pts    = len(scatter)

            self.progress.emit(92)
            self.sig_status_text.emit("Computing landing statistics...")

            stats = SimulationWorker._compute_stats(scatter, prob_pct)

            self.progress.emit(98)
            self.sig_status_text.emit("Packaging results...")

            self.sig_mc_done.emit(SimulationWorker._package_mc(scatter, stats, prob_pct))

            result = SimulationWorker._package_result(nom_pkg, scatter, stats, prob_pct, cancelled=False)
            result["nominal_surf_spd"] = float(p.get("surf_spd", 0.0))
            result["nominal_surf_dir"] = float(p.get("surf_dir", 0.0))
            result["score"] = opt_res['score']
            result["elev"] = best_e
            result["azi"] = best_a

            self.progress.emit(100)
            self.sig_finished.emit(result)

        except Exception as _exc:
            import traceback
            if self._stop_event.is_set() or str(_exc) == 'cancelled':
                self.sig_finished.emit({"cancelled": True})
            else:
                self.error.emit(traceback.format_exc())
