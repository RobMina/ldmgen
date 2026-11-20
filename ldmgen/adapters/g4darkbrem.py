"""G4DarkBreM and DMG4 adapters: dark bremsstrahlung (and DMG4's positron annihilation) inside Geant4, from this
workspace's test application `dbcompare` (generators/g4darkbrem/, README "Output format", format `dbcompare/1`).

Status: implemented against the documented format (2026-11-16); to be validated on the first matrix runs
(`runs/<user>/g4darkbrem-matrix-<engine>-m<mass>/out/<run>/`).

Input: one run directory with `summary.json` and `emissions.csv.gz`, one row per A' emission:
- the radiating lepton (`parent_pdg` 11 / -11 / 13 / -13, `parent_track`, `parent_primary`) and its four-momentum at
  the emission (the bias wrapper leaves it unchanged: the momentum before the emission);
- the vertex `x_mm, y_mm, z_mm` (origin at the centre of the slab's front face, beam along +z) and `t_ns`;
- the A' (`ap_pdg, ap_mass, ap_E..pz`) and its isotropic decay to chi (`chi1_*`) and chi-bar (`chi2_*`), `chi_mass`;
- `weight` = 1/B (the run's cross-section bias), so sum(weight) / n_primaries = A' per incident lepton.

Output: `dm_sample` "produced", two /dm rows per emission (chi, then chi-bar; same `event`), no /scatter rows.
- vertices: mm -> m, ns -> s, shifted by `slab_origin_m` (where the slab front-face centre sits in the beam frame;
  default 0: the record's frame is then the slab's);
- weights: per POT when `leptons_per_pot` is given (leptons of this beam's energy entering the slab per POT), i.e.
  weight * leptons_per_pot / n_primaries; otherwise per incident lepton (weight / n_primaries), with
  `run.summary["normalisation"]` = "per_incident_lepton" and `run.pot` = 1, which means one incident lepton;
- model: vector mediator with kinetic mixing, fermion chi (both engines); alpha_D is DMG4's (it enters only the
  annihilation width);
- extra per-chi columns in `/dm_extra`: `emission`, `parent_track`, `parent_primary`, `ap_pdg`;
- `weights_per_pot`: for runs driven by a weighted primary list (the NuMI absorber-face muons, stage 2 of option A),
  where every A' already carries w_row / B per POT; `slab_origin_m` then shifts the GDML world (g4numi MCZERO) to
  the beam frame (NuMI target centre: (0, 0, 0.817) m).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from ..record import PARENT_KINDS, EventRecord, Model, RunMeta, source_entry

ENGINES = {"g4darkbrem": "G4DarkBreM", "dmg4": "DMG4"}


def read_emissions(path):
    """The columns of emissions.csv(.gz) by name: numeric columns as float arrays, others (e.g. `material`, added to
    dbcompare's output 2026-11-16) as string arrays."""
    import csv
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    names, data = rows[0], rows[1:]
    if any(len(r) != len(names) for r in data):
        raise ValueError(f"{path}: rows with a different number of columns than the header ({len(names)})")
    cols = {}
    for i, k in enumerate(names):
        v = [r[i] for r in data]
        try:
            cols[k] = np.array(v, dtype=float)
        except ValueError:
            cols[k] = np.array(v, dtype=str)
    return cols


def _p4(c, pre):
    return np.column_stack([c[f"{pre}_E"], c[f"{pre}_px"], c[f"{pre}_py"], c[f"{pre}_pz"]])


def _get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def from_dbcompare(run_dir, engine, *, leptons_per_pot=None, slab_origin_m=(0.0, 0.0, 0.0), detector="",
                   beam_origin="", image_sha256="", version="", weights_per_pot=False):
    run_dir = Path(run_dir)
    summ = json.loads((run_dir / "summary.json").read_text())
    csv = run_dir / "emissions.csv.gz"
    if not csv.exists():
        csv = run_dir / "emissions.csv"
    c = read_emissions(csv)
    cfg = summ.get("config", {})
    eng = summ.get("engine", {}) if isinstance(summ.get("engine"), dict) else {}
    name = (cfg.get("engine") or engine).lower()
    if name != engine:
        raise ValueError(f"{run_dir}: engine {name!r}, expected {engine!r}")
    n_prim = int(summ["n_primaries"])
    n = len(c["weight"])
    if weights_per_pot:  # list-driven runs: each primary already carried its weight per POT, so sum(w) is per POT
        w = c["weight"].astype(float)
        norm = "per POT (primary weights from the input list)"
    elif leptons_per_pot is not None:
        w = c["weight"] * float(leptons_per_pot) / n_prim
        norm = f"per POT ({leptons_per_pot:g} incident leptons per POT)"
    else:
        w = c["weight"] / n_prim
        norm = "per_incident_lepton"
    pdg = c["parent_pdg"].astype(int)
    annihilation = cfg.get("process") == "annihilation"
    kind = np.where(np.abs(pdg) == 13, PARENT_KINDS["muon_brem"],
                    np.where(annihilation & (pdg == -11), PARENT_KINDS["positron_annihilation"],
                             PARENT_KINDS["electron_brem"]))
    vtx = np.column_stack([c["x_mm"], c["y_mm"], c["z_mm"]]) * 1e-3 + np.asarray(slab_origin_m, dtype=float)
    vtx = np.column_stack([vtx, c["t_ns"] * 1e-9])
    two = lambda a: np.repeat(a, 2, axis=0)  # noqa: E731  (chi, chi-bar of each emission)
    chi = np.empty((2 * n, 4))
    chi[0::2], chi[1::2] = _p4(c, "chi1"), _p4(c, "chi2")
    dm = {"event": two(c["event"].astype(np.int64)), "p4": chi, "vertex": two(vtx), "parent_pdg": two(pdg),
          "parent_kind": two(kind), "parent_p4": two(_p4(c, "parent")), "mediator_p4": two(_p4(c, "ap")),
          "mediator_mass": two(c["ap_mass"]), "weight": two(w)}
    extra = {k: two(c[k].astype(np.int64)) for k in ("emission", "parent_track", "parent_primary", "ap_pdg")
             if k in c}
    if "material" in c:  # the volume material of the emission (GDML runs): bytes, for HDF5
        extra["material"] = two(np.char.encode(c["material"], "ascii"))
    m_v = float(_get(cfg, "ap_mass_mev", default=np.nan)) / 1e3
    m_chi = float(_get(cfg, "chi_mass_mev", default=np.nan)) / 1e3
    if n and not np.isclose(c["ap_mass"][0], m_v, rtol=1e-6):
        raise ValueError(f"{run_dir}: ap_mass {c['ap_mass'][0]} GeV but config ap_mass_mev {m_v * 1e3}")
    model = Model("vector_portal", float(cfg.get("epsilon", np.nan)), float(cfg.get("alpha_d", np.nan)), m_chi, m_v,
                  "fermion", params={"engine": ENGINES[engine], "process": cfg.get("process", "brem")})
    beam = {"particle": cfg.get("lepton", ""), "energy_GeV": cfg.get("energy_gev"), "spectrum": cfg.get("spectrum", ""),
            "target": cfg.get("material", ""), "thickness_cm": cfg.get("thickness_cm"),
            "origin": beam_origin or "slab front face", "slab_origin_m": list(map(float, slab_origin_m))}
    run = RunMeta(generator=engine, generator_version=version or str(_get(eng, "version", default="")), model=model,
                  pot=1.0, dm_sample="produced", production=f"{cfg.get('lepton', '')} {cfg.get('process', 'brem')}",
                  detector=detector, image_sha256=image_sha256, beam=beam,
                  source=[source_entry(run_dir / "summary.json", "dbcompare/1 summary"),
                          source_entry(csv, "dbcompare/1 emissions")],
                  card=json.dumps(cfg, sort_keys=True),
                  summary={"normalisation": norm, "n_primaries": n_prim, "n_emissions": n,
                           "bias": summ.get("bias"), "ap_per_lepton": float(c["weight"].sum()) / n_prim,
                           "geant4": summ.get("geant4", "")},
                  notes=f"{ENGINES[engine]} via dbcompare; weights {norm}")
    return EventRecord.build(run, dm, None, dm_extra=extra)


def from_g4darkbrem(run_dir, **kw):
    return from_dbcompare(run_dir, "g4darkbrem", **kw)


def from_dmg4(run_dir, **kw):
    return from_dbcompare(run_dir, "dmg4", **kw)
