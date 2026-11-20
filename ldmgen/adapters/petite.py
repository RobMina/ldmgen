"""PETITE adapter: electromagnetic-shower (and pi0 -> gamma V) production of dark photons and DM at NuMI, written
by this workspace's PETITE driver (generators/petite/, README "Output format").

Status: implemented against the documented format (2026-11-15); to be validated on the first merged file.

The driver writes one HDF5 file per model point and geometry hypothesis (`merged/petite_<config>_<hyp>.h5`), already
close to ldmgen/1:
- root attribute `run` (JSON): generator, generator_version, image_sha256, model (ldmgen Model fields), pot (1.0:
  weights are per POT), dm_sample "detector", detector, frame, hypothesis, source, summary, notes;
- `/dm` with the ldmgen/1 columns. Vertices are the V emission point in m, in the NuMI target-centre frame (the beam
  frame of ldmgen for NOvA), t NaN; parent_p4 NaN (PETITE does not keep the radiating particle at emission);
- extra per-chi columns `process` (0 pi0_decay, 1 dark_brem, 2 annihilation, 3 dark_compton), `shower_material`
  (0 none, 1 graphite, 2 aluminum, 3 iron), `pi0_z`, `pi0_E`, `pi0_row`, `L_det`: kept in the record's `/dm_extra`;
- no /scatter rows (detector scattering is a separate step); weights = chi crossing the detector per POT at the
  file's epsilon, including BR(V -> chi chi);
- seed modes: "local"/"ray" hypotheses shower the photons of g4numi pi0s; hypothesis "emseed" (2026-11-16) showers
  g4numi's own e+-/gamma seeds (the opt-in `emseed` tree). Same layout; there the pi0_z / pi0_E / pi0_row columns hold
  the seed's origin z, energy and row, and the adapter names them seed_z / seed_E / seed_row.
"""
from __future__ import annotations

import json

import numpy as np

from ..record import DETECTOR_BEAM_ORIGIN, DM_COLUMNS, EventRecord, Model, RunMeta, source_entry

PROCESSES = {0: "pi0_decay", 1: "dark_brem", 2: "annihilation", 3: "dark_compton"}
MATERIALS = {0: "none", 1: "graphite", 2: "aluminum", 3: "iron"}
EMSEED_RENAME = {"pi0_z": "seed_z", "pi0_E": "seed_E", "pi0_row": "seed_row"}
RUN_FIELDS = set(RunMeta.__dataclass_fields__)


def from_petite(path, *, detector=None, image_sha256=None, version=None):
    import h5py
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["run"])
        g = f["dm"]
        cols = {k: g[k][()] for k in g}
    if meta.get("generator", "petite") != "petite":
        raise ValueError(f"{path}: generator {meta.get('generator')!r}, expected petite")
    dm = {k: v for k, v in cols.items() if k in DM_COLUMNS}
    extra = {k: v for k, v in cols.items() if k not in DM_COLUMNS}
    if meta.get("hypothesis") == "emseed":  # seeds are g4numi's e/gamma, not pi0s: the pi0_* columns hold the seed's
        extra = {EMSEED_RENAME.get(k, k): v for k, v in extra.items()}
    missing = {"event", "p4", "vertex", "weight"} - set(dm)
    if missing:
        raise ValueError(f"{path}: /dm lacks {sorted(missing)}")
    m = meta["model"]
    model = Model(**{k: m[k] for k in ("name", "epsilon", "alpha_D", "m_chi", "m_V") if k in m},
                  dm_spin=m.get("dm_spin", "any"), params=dict(m.get("params", {})))
    # "any": PETITE's driver decays V -> chi chi-bar isotropically with BR(V -> chi chi) ~ 1 at these masses, so the
    # sample does not depend on the DM spin (the true decay angle is sin^2 for scalar, 1 + cos^2 for fermion DM)
    det = detector if detector is not None else meta.get("detector", "")
    beam = dict(meta.get("beam", {}))
    beam.setdefault("origin", DETECTOR_BEAM_ORIGIN.get(det, "numi-target-centre"))
    others = {k: v for k, v in meta.items() if k not in RUN_FIELDS}  # frame, hypothesis, ...
    notes = meta.get("notes", "")
    if others:
        notes = (notes + "\n" if notes else "") + "petite run keys: " + json.dumps(others, sort_keys=True)
    src = list(meta.get("source", [])) if isinstance(meta.get("source"), list) else [meta.get("source")]
    run = RunMeta(generator="petite", generator_version=version or meta.get("generator_version", ""), model=model,
                  pot=float(meta.get("pot", 1.0)), dm_sample=meta.get("dm_sample", "detector"),
                  production=",".join(PROCESSES[p] for p in sorted(set(np.unique(extra.get("process", []))))),
                  detector=det, image_sha256=image_sha256 or meta.get("image_sha256", ""), beam=beam,
                  source=[s for s in src if s] + [source_entry(path, "petite-hdf5")],
                  card=json.dumps(meta.get("config", {})) if "config" in meta else "",
                  summary=dict(meta.get("summary", {})), notes=notes)
    return EventRecord.build(run, dm, None, dm_extra=extra)
