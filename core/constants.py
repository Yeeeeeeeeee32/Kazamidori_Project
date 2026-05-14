"""
core/constants.py
Universal physical and statistical constants shared across the simulation
core (``simulation``, ``monte_carlo``, ``optimization``).

Centralising these values prevents silent drift between modules — for
example, ``9.80665`` vs ``9.81`` for standard gravity, or duplicated copies
of the chi-squared 2-DOF 90 % quantile (``4.605``).  Every consumer should
import from here; do not redefine these values locally.

All constants are SI unless explicitly noted.
"""

from __future__ import annotations


# ── Mechanics ────────────────────────────────────────────────────────────────

# Standard gravity (CODATA / CGPM definition, exact by convention).
# Used for propellant-mass back-calculation (``m_prop = J / (Isp × g0)``)
# and any reference acceleration in diagnostic drag/coast estimates.
G0: float = 9.80665  # m/s²


# ── Atmosphere ───────────────────────────────────────────────────────────────

# Standard sea-level air density (ISA, 15 °C, 101 325 Pa).
# Used only for reference / diagnostic drag estimates that need a single
# rho value.  The RocketPy integrator itself uses the altitude-dependent
# ISA density profile during flight simulation.
RHO_0: float = 1.225  # kg/m³


# ── Wind profile altitudes ───────────────────────────────────────────────────

# Surface anemometer altitude (m AGL) — 自作風速計 installation height.
# Anchors the bottom of every wind profile and feeds the surface ramp.
OBS_ALT: float = 3.0  # m AGL

# Altitude (m AGL) above which GPV / upper-wind data has full weight in
# the blended profile.  Below this, the profile linearly transitions from
# the surface observation to GPV.
BLEND_ALT: float = 100.0  # m AGL


# ── Statistics ───────────────────────────────────────────────────────────────

# Chi-squared upper quantile for 2 degrees of freedom at 90 % containment:
#     P(χ²(2) ≤ CHI2_90) = 0.90
# Equivalent analytic form: -2 × ln(1 − 0.90).
#
# Used as the *squared* scale factor for 2-D bivariate-normal error ellipse
# semi-axes; consumers take ``sqrt(CHI2_90)`` as the linear axis scale.
CHI2_90: float = 4.605
