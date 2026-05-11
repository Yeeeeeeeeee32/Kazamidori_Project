"""
utils/geometry_math.py
Moment of Inertia (MoI) calculation engine for rocket component geometry.

All functions are pure (no I/O, no class state, no UI).

Coordinate convention
---------------------
The rocket longitudinal axis is the **X-axis** (nose tip → tail).

    I_xx  =  roll  MoI — about the X / longitudinal axis  (kg·m²)
    I_yy  =  pitch MoI — about the Y / lateral axis        (kg·m²)
    I_zz  =  yaw   MoI — about the Z / lateral axis        (kg·m²)

For every axisymmetric component (cylinders, cones) I_yy == I_zz.
For fin sets I_yy == I_zz by symmetry when N ≥ 2 fins are equally spaced.

All positions are measured from the **nose tip**, increasing toward the tail.
All inputs and outputs are strictly SI: metres, kilograms, kg·m².

Unit contract
-------------
The .rkt (RockSim) file stores all linear dimensions in **millimetres**.
The caller (XML parser) must convert mm → m before passing values here.
This module never performs unit conversion.

Parallel Axis Theorem (PAT)
---------------------------
For each component i with local CG at x_i and local MoI I_cm,i about its
own centroid:

    I_system_pitch = Σ (I_cm,i.Iyy  +  m_i · (x_i − x_sys)²)
    I_system_roll  = Σ  I_cm,i.Ixx
        (no radial PAT term: all standard shapes are on the rocket axis)
    I_system_yaw   = I_system_pitch   (by symmetry)

Exception: fin sets have pre-loaded radial offset in their Ixx — see
moi_trapezoidal_fin_set() for details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple


# ── Axis-MoI triplet ──────────────────────────────────────────────────────────

class MoITriplet(NamedTuple):
    """Three principal moments of inertia (kg·m²).

    Ixx — roll  (about longitudinal axis)
    Iyy — pitch (about lateral Y axis)
    Izz — yaw   (about lateral Z axis, equals Iyy for symmetric parts)
    """
    Ixx: float
    Iyy: float
    Izz: float


# ── Component descriptor ──────────────────────────────────────────────────────

@dataclass
class Component:
    """Single structural element for MoI summation.

    Attributes:
        name:      Label used in debug output.
        mass:      Component mass [kg].
        local_cg:  Centroid position from nose tip [m].
        local_moi: MoI about the component's own centroid [kg·m²].
    """
    name:      str
    mass:      float
    local_cg:  float        # metres from nose tip
    local_moi: MoITriplet


# ── System MoI result ─────────────────────────────────────────────────────────

class SystemMoI(NamedTuple):
    """Full-rocket moment of inertia about the system CG (kg·m²).

    Ixx — roll  (about longitudinal axis through system CG)
    Iyy — pitch (about lateral Y axis through system CG)
    Izz — yaw   (about lateral Z axis through system CG)
    """
    Ixx: float
    Iyy: float
    Izz: float


# ── Shape MoI helpers ─────────────────────────────────────────────────────────

def moi_hollow_cylinder(
    mass:         float,
    outer_radius: float,
    inner_radius: float,
    length:       float,
) -> MoITriplet:
    """Local MoI for a hollow cylinder (body tube, motor mount, coupler, ring).

    Derivation
    ----------
    Integrate dm = ρ · 2πr · dr · L over r ∈ [r_i, r_o]:

        I_xx  =  m/2 · (r_o² + r_i²)

    Integrate (r²/4 + x²) dm over r and x ∈ [−L/2, L/2]:

        I_yy  =  m/12 · [3(r_o² + r_i²) + L²]

    Args:
        mass:         Component mass [kg].
        outer_radius: Outer radius r_o [m].
        inner_radius: Inner radius r_i [m].  Pass 0.0 for solid cylinder.
        length:       Axial length L [m].

    Returns:
        MoITriplet(Ixx, Iyy, Iyy) — Iyy == Izz for all hollow cylinders.
    """
    ro2 = outer_radius ** 2
    ri2 = inner_radius ** 2
    ixx = 0.5  * mass * (ro2 + ri2)
    iyy = (mass / 12.0) * (3.0 * (ro2 + ri2) + length ** 2)
    return MoITriplet(ixx, iyy, iyy)


def moi_solid_cylinder(
    mass:   float,
    radius: float,
    length: float,
) -> MoITriplet:
    """Local MoI for a solid cylinder (motor propellant, ballast, dense payload).

    Derivation
    ----------
    Special case of moi_hollow_cylinder with r_i = 0:

        I_xx  =  m/2 · r²
        I_yy  =  m/12 · (3r² + L²)

    Args:
        mass:   Component mass [kg].
        radius: Cylinder radius r [m].
        length: Axial length L [m].

    Returns:
        MoITriplet(Ixx, Iyy, Iyy).
    """
    r2  = radius ** 2
    ixx = 0.5  * mass * r2
    iyy = (mass / 12.0) * (3.0 * r2 + length ** 2)
    return MoITriplet(ixx, iyy, iyy)


def moi_hollow_cone(
    mass:        float,
    base_radius: float,
    length:      float,
) -> MoITriplet:
    """Local MoI for a thin-walled conical shell (nose cone, boat-tail shell).

    Models the part as a **thin conical shell** (wall thickness << radius).
    The caller should set local_cg = x_apex + 2·length/3 (nose-tip frame)
    because the shell centroid is at 2/3 of the axial length from the apex.

    Derivation
    ----------
    Strip at height z from apex: radius r(z) = R·z/h, slant width ds = dz/cosα.
    Surface density ρ_s = m / (π·R·s),  s = √(R² + h²).

    Roll  (about cone axis):
        I_xx = ∫ r(z)² dm  =  m·R² / 2

    Lateral about apex (intermediate step):
        I_yy_apex = ∫ [r(z)²/2 + z²] dm  =  m·(R²/4 + h²/2)

    Centroid from apex:
        z_cg = 2h/3

    Lateral about centroid (PAT: subtract m·z_cg²):
        I_yy_cg = m·(R²/4 + h²/2) − m·(2h/3)²
                = m·(R²/4 + h²/18)

    Args:
        mass:        Component mass [kg].
        base_radius: Cone base radius R [m].
        length:      Axial length h from apex to base [m].

    Returns:
        MoITriplet(Ixx, Iyy, Iyy) — all values about the component centroid.
    """
    R2  = base_radius ** 2
    h2  = length ** 2
    ixx = 0.5  * mass * R2
    iyy = mass * (R2 / 4.0 + h2 / 18.0)
    return MoITriplet(ixx, iyy, iyy)


def moi_point_mass(mass: float) -> MoITriplet:  # noqa: ARG001
    """Local MoI for an idealised point mass (parachute, shock cord, nose weight).

    A point mass has zero local MoI; its only system contribution is the
    parallel-axis term  m·(x_cg − x_sys)²  applied by calculate_total_moi.

    Args:
        mass: Component mass [kg] — accepted for API uniformity, not used.

    Returns:
        MoITriplet(0.0, 0.0, 0.0).
    """
    return MoITriplet(0.0, 0.0, 0.0)


def moi_trapezoidal_fin_set(
    mass_total:  float,
    fin_count:   int,
    body_radius: float,
    root_chord:  float,
    tip_chord:   float,
    semi_span:   float,
) -> MoITriplet:
    """Local MoI for a set of N equally-spaced trapezoidal fins (thin plate).

    Each fin is modelled as a **thin flat plate** (thickness ≈ 0).
    The fin-set centroid is on the rocket axis (by N-fold symmetry); its
    axial CG position must be set by the caller.

    Derivation
    ----------
    Let:
        m_f  = mass_total / N          (mass per fin)
        c_a  = (root + tip) / 2        (average chord)
        y_c  = span · (root + 2·tip) / [3·(root + tip)]   (centroid span position,
                                                            from root edge — centroid
                                                            of a trapezoid along height)
        r_c  = body_radius + y_c       (radial distance of fin centroid from rocket axis)

    **Roll  (I_xx) — about the rocket longitudinal axis:**
    For each fin lying in a meridional half-plane at azimuthal angle φ_n,
    its radial-direction extent spans from body_radius to body_radius + semi_span.
    Using PAT from the fin's own centroid (at r_c) inward to the rocket axis:

        I_xx_one = I_fin_span_axis + m_f · r_c²
                 = m_f · span² / 12  +  m_f · r_c²

    where  m_f · span²/12  is the thin-plate moment about an axis through
    the fin's own centroid, parallel to the rocket axis (chord direction).

        I_xx_fins = N · I_xx_one  (fins add coherently for roll)

    **Lateral  (I_yy = I_zz) — about a lateral axis through the fin-set CG:**
    For N ≥ 2 fins equally spaced, the azimuthal symmetry identity gives:

        Σ sin²(φ_n) = N/2

    Each fin's contribution to I_yy has two parts:
      (a)  m_f · r_c² · sin²(φ_n)   — radial mass displaced from Y-axis
      (b)  m_f · c_a² / 12           — thin-plate chord moment (approx. const)

    Summing over N fins:

        I_yy_fins = (N/2) · m_f · r_c²  +  N · m_f · c_a² / 12

    Note: the axial PAT term  N · m_f · (x_cg − x_sys)²  is NOT included
    here; calculate_total_moi() applies it automatically.

    Args:
        mass_total:  Total mass of all fins combined [kg].
        fin_count:   Number of fins N (≥ 2 required for symmetry to hold).
        body_radius: Rocket body radius at fin root [m].
        root_chord:  Fin root chord length [m].
        tip_chord:   Fin tip chord length [m].
        semi_span:   Fin semi-span (root to tip) [m].

    Returns:
        MoITriplet(Ixx, Iyy, Izz) about the fin-set's own CG [kg·m²].
        Ixx includes the radial-offset contribution (unlike other helpers)
        because the fins are never on the rocket axis.
    """
    if fin_count < 1:
        return MoITriplet(0.0, 0.0, 0.0)

    m_f  = mass_total / fin_count
    c_a  = (root_chord + tip_chord) * 0.5
    # Centroid of a trapezoid along its height:  y_c = s·(b₁ + 2b₂)/(3(b₁+b₂))
    denom = root_chord + tip_chord
    y_c   = (semi_span * (root_chord + 2.0 * tip_chord) / (3.0 * denom)
             if denom > 0.0 else semi_span * 0.5)
    r_c   = body_radius + y_c

    # Roll: each fin contributes m_f·(r_c² + span²/12); all N add coherently.
    ixx_one  = m_f * (r_c ** 2 + semi_span ** 2 / 12.0)
    ixx      = fin_count * ixx_one

    # Lateral: azimuthal symmetry gives factor N/2 on the radial term.
    iyy = (fin_count / 2.0) * m_f * r_c ** 2 + fin_count * m_f * c_a ** 2 / 12.0

    return MoITriplet(ixx, iyy, iyy)


# ── System CG ─────────────────────────────────────────────────────────────────

def calculate_system_cg(components: list[Component]) -> float:
    """Return the system centre of mass position from the nose tip [m].

    Args:
        components: All rocket structural elements.

    Returns:
        x_cg [m].  Returns 0.0 for an empty list or zero total mass.
    """
    total_mass = sum(c.mass for c in components)
    if total_mass <= 0.0:
        return 0.0
    return sum(c.mass * c.local_cg for c in components) / total_mass


# ── Parallel Axis Theorem summation ───────────────────────────────────────────

def calculate_total_moi(
    components: list[Component],
    system_cg:  float,
) -> SystemMoI:
    """Sum component MoIs about the system CG using the Parallel Axis Theorem.

    For each component i at local_cg x_i, with mass m_i and local MoI I_cm_i:

        I_roll  = Σ  I_cm_i.Ixx
            (no radial PAT: standard shapes are on the rocket axis;
             fin-set Ixx already carries its radial term — see moi_trapezoidal_fin_set)

        I_pitch = Σ (I_cm_i.Iyy  +  m_i · (x_i − x_sys)²)
        I_yaw   = Σ (I_cm_i.Izz  +  m_i · (x_i − x_sys)²)

    Args:
        components: List of :class:`Component` instances.
        system_cg:  System CG from nose tip [m]; typically from
                    :func:`calculate_system_cg`.

    Returns:
        :class:`SystemMoI` (Ixx, Iyy, Izz) in kg·m².
    """
    total_ixx = 0.0
    total_iyy = 0.0
    total_izz = 0.0

    for comp in components:
        d2 = (comp.local_cg - system_cg) ** 2   # squared axial offset

        total_ixx += comp.local_moi.Ixx
        total_iyy += comp.local_moi.Iyy + comp.mass * d2
        total_izz += comp.local_moi.Izz + comp.mass * d2

    return SystemMoI(total_ixx, total_iyy, total_izz)
