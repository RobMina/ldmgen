"""Detector acceptance and an electron-scattering estimate, in BdNMC's conventions, for records whose DM are produced
upstream (`dm_sample` "produced": G4DarkBreM/DMG4, or anything else with chi vertices and momenta).

- `Cuboid.from_card(text)`: BdNMC's `detector cuboid` (centre x/y/z-position, width (local x), height (local y),
  length (local z), Euler angles det-phi, det-theta, det-psi in its XYX convention), in the beam frame (m).
- `chord_lengths(cuboid, vertex, p)`: the path length of each straight chi line inside the box (0 when it misses or
  points away); `to_detector_sample(record, cuboid)` keeps the crossing chi as a `dm_sample` "detector" record with
  `L_det` in /dm_extra.
- `expected_signal(record, card_text)`: sum over chi of weight x P_scatter, P_scatter = sigma(E) n_e L with BdNMC's
  NCE_electron cross section (bdnmc_jax.electron: complex scalar DM, the card's recoil window) and the card's
  electron density. This is BdNMC's own `scatter_probability`, so a BdNMC "detector" record gives back BdNMC's
  predicted signal; for fermion-DM generators it treats their chi as BdNMC's scalar DM (state so when quoting).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _xyx_rotate(t1, t2, t3, v):
    """BdNMC detector.cpp XYX_rotate (as bdnmc_jax.detector.xyx_rotate and bdnmc_dune.geometry.xyx_rotate)."""
    x, y, z = v
    c1, s1, c2, s2, c3, s3 = math.cos(t1), math.sin(t1), math.cos(t2), math.sin(t2), math.cos(t3), math.sin(t3)
    return np.array([
        x * c2 + z * c3 * s2 + y * s2 * s3,
        x * s1 * s2 + z * (-c2 * c3 * s1 - c1 * s3) + y * (c1 * c3 - c2 * s1 * s3),
        -x * c1 * s2 + y * (c3 * s1 + c1 * c2 * s3) + z * (c1 * c2 * c3 - s1 * s3),
    ])


def _num(s):
    return float(str(s).lower().replace("d", "e"))


@dataclass(frozen=True)
class Cuboid:
    center: tuple          # m, beam frame
    half: tuple            # half sizes along the local axes (m)
    axes: tuple            # rows: local x, y, z unit vectors in the beam frame
    electron_density_cm3: float = float("nan")   # sum over materials of electron_number x number_density

    @classmethod
    def from_card(cls, text):
        k, mats, cur = {}, [], None
        for line in text.splitlines():
            f = line.split("#", 1)[0].split()
            if len(f) < 2:
                continue
            if f[0] == "material":
                cur = {}
                mats.append(cur)
            elif cur is not None and f[0] in ("number_density", "electron_number"):
                cur[f[0]] = _num(f[1])
                continue
            k.setdefault(f[0], f[1])
        if k.get("detector") != "cuboid":
            raise ValueError(f"detector {k.get('detector')!r}: only cuboid is implemented")
        c = tuple(_num(k[a]) for a in ("x-position", "y-position", "z-position"))
        half = (_num(k["width"]) / 2, _num(k["height"]) / 2, _num(k["length"]) / 2)
        ang = [_num(k.get(a, 0)) for a in ("det-phi", "det-theta", "det-psi")]
        axes = tuple(tuple(_xyx_rotate(*ang, e)) for e in np.eye(3))
        ne = sum(m.get("electron_number", 0) * m.get("number_density", 0) for m in mats)
        return cls(c, half, axes, ne if mats else float("nan"))


def chord_lengths(cub, vertex, p):
    """Length (m) of each ray vertex + s p_hat, s > 0, inside the cuboid (slab method, local frame)."""
    A = np.asarray(cub.axes, float)
    o = (np.asarray(vertex, float)[:, :3] - np.asarray(cub.center)) @ A.T
    d = np.asarray(p, float)[:, -3:]
    d = d / np.linalg.norm(d, axis=1)[:, None]
    d = d @ A.T
    h = np.asarray(cub.half)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-h - o) / d, (h - o) / d
    lo = np.where(d != 0, np.minimum(t1, t2), np.where(np.abs(o) <= h, -np.inf, np.inf))
    hi = np.where(d != 0, np.maximum(t1, t2), np.where(np.abs(o) <= h, np.inf, -np.inf))
    t_in = np.maximum(lo.max(axis=1), 0.0)
    t_out = hi.min(axis=1)
    return np.clip(t_out - t_in, 0.0, None)


def to_detector_sample(rec, cub, detector=""):
    """The chi of `rec` whose lines cross `cub`, as a dm_sample "detector" record (L_det in /dm_extra)."""
    L = chord_lengths(cub, rec.dm["vertex"], rec.dm["p4"])
    sub = rec.select_dm(L > 0, f"crossing the cuboid at {cub.center}")
    sub.dm_extra["L_det"] = L[L > 0]
    sub.run.dm_sample = "detector"
    if detector:
        sub.run.detector = detector
    return sub


def expected_signal(rec, card_text, e_min=None, e_max=None):
    """Expected NCE_electron signal events per POT for the chi of a "detector" record (needs L_det), with BdNMC's
    cross section and the card's electron density and recoil window (min/max_scatter_energy, total energies)."""
    import jax.numpy as jnp
    from bdnmc_jax import electron as el
    cub = Cuboid.from_card(card_text)
    kv = {}
    for line in card_text.splitlines():
        f = line.split("#", 1)[0].split()
        if len(f) >= 2:
            kv.setdefault(f[0], f[1])
    m = rec.run.model
    sc = el.ElectronScatter(m.m_chi, m.m_V, m.epsilon, m.alpha_D,
                            _num(kv.get("min_scatter_energy", 0.035)) if e_min is None else e_min,
                            _num(kv.get("max_scatter_energy", 1e3)) if e_max is None else e_max)
    L = rec.dm_extra["L_det"]
    E = jnp.asarray(rec.dm["p4"][:, 0])
    prob = np.asarray(el.scatter_probability(sc, E, jnp.asarray(L), cub.electron_density_cm3))
    return float(np.sum(rec.dm["weight"] * prob) / rec.run.pot), prob
