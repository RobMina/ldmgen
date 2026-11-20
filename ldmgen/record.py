"""The generator-neutral event record and its on-disk form (HDF5, schema `ldmgen/1`).

One file = one run of one generator at one model point, optionally for one detector. Three parts:

- `RunMeta` (root attribute `run`, JSON): generator, version, image sha256, the model and all its parameters, the POT
  the weights are normalised to, what the DM table holds (`dm_sample`), the production channel, the detector, the
  source files and the generator's own summary numbers.
- `/dm`: one row per DM particle (chi). Four-momentum and production vertex in the beam frame, the parent.
- `/scatter`: one row per DM interaction in the detector (empty for production-only samples). Interaction vertex in
  the beam frame (and in the detector frame when the adapter knew it), the outgoing electron or nucleon, the recoil
  DM, and the weight.

Frames and units (the same for every generator; adapters convert):
- beam frame: BdNMC's frame for the experiment: +z along the primary beam, +y up (as far as the beam is
  horizontal), x = y cross z, origin at the point the experiment's detector placement is measured from
  (`BEAM_ORIGINS`; recorded per run in RunMeta.beam["origin"]): for NOvA the centre of the NuMI target,
  for DUNE the LBNF target point used by bdnmc-dune. Adapters for other generators convert to it;
- four-momenta [E, px, py, pz] in GeV; vertices [x, y, z, t] in m and s (t = 0 at the primary interaction;
  NaN when the generator does not give a time);
- detector frame (`/scatter/vertex_det`, optional): [x, y, z, t] in cm and ns, the frame the detector simulation
  expects; the transform is stored with it (attribute `frame`: origin_m, rotation, as detector = R (beam - origin)).

Weights: a row's weight is the number of such rows expected for `RunMeta.pot` protons on target, so
sum(weight) / pot is a yield per POT. For `/scatter` that is the expected signal events; for `/dm` it depends on
`dm_sample`: "produced" (every DM made), "detector" (DM whose line crosses the detector) or "scattered" (only the
DM that scattered; each row carries its scatter's weight).

Missing values are NaN (floats) or -1 / 0 (integers, as documented per column).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "ldmgen/1"

# parent_kind codes (/dm/parent_kind); the mapping is also stored on the dataset as attribute `codes`.
PARENT_KINDS = {"unknown": 0, "meson_decay": 1, "proton_brem": 2, "electron_brem": 3, "positron_annihilation": 4,
                "muon_brem": 5, "compton": 6, "drell_yan": 7, "other": 9}
DM_SAMPLES = ("produced", "detector", "scattered")
# RunMeta.beam["origin"]: the beam frame's origin, per beamline
BEAM_ORIGINS = {"numi-target-centre": "centre of the NuMI target (NOvA)",
                "lbnf-target": "the LBNF target point used by bdnmc-dune for ND-LAr"}
DETECTOR_BEAM_ORIGIN = {"nova-nd": "numi-target-centre", "dune-ndlar": "lbnf-target"}

# Columns: name -> (dtype, trailing shape, fill value, description). The README's schema table is made from these.
DM_COLUMNS = {
    "event": ("i8", (), -1, "generator event number (rows of one event share it)"),
    "p4": ("f8", (4,), np.nan, "chi four-momentum [E, px, py, pz], GeV, beam frame"),
    "vertex": ("f8", (4,), np.nan, "production vertex [x, y, z, t], m and s, beam frame"),
    "parent_pdg": ("i4", (), 0, "PDG code of the parent: 111, 221, ... for meson decays; the radiating particle "
                                "(2212, 11, -11, 13) for brem and annihilation; 0 unknown"),
    "parent_kind": ("i1", (), 0, "production mechanism, PARENT_KINDS code"),
    "parent_p4": ("f8", (4,), np.nan, "parent four-momentum (the meson; the radiating particle for brem), GeV"),
    "mediator_p4": ("f8", (4,), np.nan, "dark-photon (mediator) four-momentum, GeV"),
    "mediator_mass": ("f8", (), np.nan, "mediator mass, GeV (differs from m_V for off-shell production)"),
    "weight": ("f8", (), np.nan, "number of such DM per RunMeta.pot POT (meaning set by RunMeta.dm_sample)"),
}
SCATTER_COLUMNS = {
    "event": ("i8", (), -1, "event number (rows of one event share it); the scatter stage gives each scatter its own"),
    "dm_index": ("i8", (), -1, "row of /dm of the incoming chi (-1: not recorded)"),
    "vertex": ("f8", (4,), np.nan, "interaction vertex [x, y, z, t], m and s, beam frame"),
    "target_pdg": ("i4", (), 11, "struck particle: 11 electron, 2212 proton, 2112 neutron, 100ZZZAAA0 nucleus"),
    "out_pdg": ("i4", (), 11, "outgoing visible particle (the recoil electron or nucleon)"),
    "out_p4": ("f8", (4,), np.nan, "outgoing visible particle four-momentum, GeV, beam frame"),
    "out_mass": ("f8", (), np.nan, "its mass, GeV"),
    "chi_out_p4": ("f8", (4,), np.nan, "recoil chi four-momentum, GeV, beam frame"),
    "weight": ("f8", (), np.nan, "expected signal events per RunMeta.pot POT"),
}
OPTIONAL_SCATTER = {"vertex_det": ("f8", (4,), np.nan, "interaction vertex [x, y, z, t], cm and ns, detector frame "
                                                       "(attribute `frame`)")}


@dataclass
class Model:
    """The physics model point. Masses in GeV. `name` e.g. "vector_portal" (kinetically mixed dark photon decaying
    to chi chi); `dm_spin` "fermion" (Dirac), "majorana", "scalar", or "any" for samples that do not depend on it
    (production with an isotropic V -> chi chi decay and BR = 1; matches either spin in `same_point`); `params` holds anything else the generator
    took (e.g. a coupling g_D instead of alpha_D, or the UFO name)."""
    name: str
    epsilon: float
    alpha_D: float
    m_chi: float
    m_V: float
    dm_spin: str = "fermion"
    params: dict = field(default_factory=dict)

    def same_point(self, other, rtol=1e-6):
        """True when the masses, couplings, spin and model name agree (rtol on the numbers)."""
        num = all(np.isclose(getattr(self, k), getattr(other, k), rtol=rtol, atol=0)
                  for k in ("epsilon", "alpha_D", "m_chi", "m_V"))
        spin = self.dm_spin == other.dm_spin or "any" in (self.dm_spin, other.dm_spin)
        return num and spin and self.name == other.name


@dataclass
class RunMeta:
    generator: str                      # "bdnmc", "bdnmc-jax", "maddump", "petite", "g4darkbrem", "dmg4"
    generator_version: str              # version or commit
    model: Model
    pot: float                          # weights are per this many POT
    dm_sample: str                      # one of DM_SAMPLES
    production: str = ""                # channel as the generator names it ("pi0_decay", "Brem_V", ...)
    detector: str = ""                  # "nova-nd", "dune-ndlar", "" (none: a production-only sample)
    image_sha256: str = ""              # the container the generator ran in
    beam: dict = field(default_factory=dict)        # e.g. {"particle": "p", "energy_GeV": 120, "target": "C"}
    source: list = field(default_factory=list)      # [{"path", "format", "sha256"}] the files converted
    card: str = ""                      # the generator's run card / configuration text, verbatim
    summary: dict = field(default_factory=dict)     # generator's own numbers, per `pot` (e.g. signal_events)
    notes: str = ""
    production_offset_m: float = 0.0    # z added to the generator's vertices to reach the beam frame (adapter)
    created: str = ""
    ldmgen_version: str = ""

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        d["model"] = Model(**d["model"])
        return cls(**d)


def _table(columns, n, given):
    """A dict of arrays for `columns`, length n, from `given` (missing columns filled)."""
    out = {}
    for name, (dtype, shape, fill, _) in columns.items():
        if name in given and given[name] is not None:
            a = np.asarray(given[name], dtype=dtype)
            if a.shape != (n,) + shape:
                a = np.broadcast_to(a, (n,) + shape).copy()
        else:
            a = np.full((n,) + shape, fill, dtype=dtype)
        out[name] = a
    unknown = set(given) - set(columns) - set(OPTIONAL_SCATTER)
    if unknown:
        raise KeyError(f"unknown columns {sorted(unknown)}")
    return out


@dataclass
class EventRecord:
    run: RunMeta
    dm: dict            # column -> array (DM_COLUMNS)
    scatter: dict       # column -> array (SCATTER_COLUMNS, + optional vertex_det)
    frame: dict = field(default_factory=dict)   # the detector frame of scatter["vertex_det"], if present
    dm_extra: dict = field(default_factory=dict)  # generator-specific per-DM columns (group /dm_extra), optional

    @classmethod
    def build(cls, run, dm, scatter=None, frame=None, dm_extra=None):
        """Fill missing columns with their fill values and check shapes."""
        n_dm = len(next(iter(dm.values()))) if dm else 0
        scatter = scatter or {}
        n_sc = len(next(iter(scatter.values()))) if scatter else 0
        sc = _table(SCATTER_COLUMNS, n_sc, scatter)
        if scatter.get("vertex_det") is not None:
            sc["vertex_det"] = np.asarray(scatter["vertex_det"], dtype="f8").reshape(n_sc, 4)
        extra = {k: np.asarray(v) for k, v in (dm_extra or {}).items()}
        bad = [k for k, v in extra.items() if len(v) != n_dm]
        if bad:
            raise ValueError(f"dm_extra columns {bad} do not have one row per DM")
        rec = cls(run, _table(DM_COLUMNS, n_dm, dm), sc, dict(frame or {}), extra)
        rec.validate()
        return rec

    @property
    def n_dm(self):
        return len(self.dm["event"])

    @property
    def n_scatter(self):
        return len(self.scatter["event"])

    def validate(self):
        r = self.run
        if r.dm_sample not in DM_SAMPLES:
            raise ValueError(f"dm_sample {r.dm_sample!r}: one of {DM_SAMPLES}")
        if not (r.pot > 0):
            raise ValueError(f"pot {r.pot}: the weights' POT normalisation must be > 0")
        di = self.scatter["dm_index"]
        if len(di) and (di.max() >= self.n_dm or di.min() < -1):
            raise ValueError("scatter dm_index outside /dm")
        if "vertex_det" in self.scatter and not self.frame:
            raise ValueError("vertex_det without its frame")
        if np.any(~np.isfinite(self.scatter["weight"])) or np.any(~np.isfinite(self.dm["weight"])):
            raise ValueError("non-finite weights")

    def select_dm(self, mask, note=""):
        """A new record with the /dm rows where `mask` is true, their /scatter rows (dm_index remapped) and dm_extra."""
        import copy
        mask = np.asarray(mask, bool)
        new_index = np.cumsum(mask) - 1
        dm = {k: v[mask] for k, v in self.dm.items()}
        extra = {k: v[mask] for k, v in self.dm_extra.items()}
        di = self.scatter["dm_index"]
        keep = (di < 0) | mask[np.clip(di, 0, None)] if len(di) else np.zeros(0, bool)
        sc = {k: v[keep] for k, v in self.scatter.items()}
        if len(sc["dm_index"]):
            sc["dm_index"] = np.where(sc["dm_index"] >= 0, new_index[np.clip(sc["dm_index"], 0, None)], -1)
        run = copy.deepcopy(self.run)
        run.notes = (run.notes + "\n" if run.notes else "") + f"selection: {note}"
        return EventRecord(run, dm, sc, dict(self.frame), extra)

    def yield_per_pot(self):
        """(signal scatters per POT, DM per POT in `dm_sample`)."""
        return float(self.scatter["weight"].sum()) / self.run.pot, float(self.dm["weight"].sum()) / self.run.pot

    # ---- HDF5 -------------------------------------------------------------------------------------------------
    def save(self, path, compression="gzip"):
        """Write the record to `path` (HDF5, schema ldmgen/1). Returns the path."""
        import h5py
        from . import __version__
        self.validate()
        if not self.run.created:
            self.run.created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        self.run.ldmgen_version = __version__
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with h5py.File(tmp, "w") as f:
            f.attrs["schema"] = SCHEMA
            f.attrs["run"] = self.run.to_json()
            f.attrs["generator"] = self.run.generator
            if self.dm_extra:
                g = f.create_group("dm_extra")
                for name, arr in self.dm_extra.items():
                    g.create_dataset(name, data=arr, **({"compression": compression} if compression and arr.size
                                                         else {}))
            for gname, cols, table in (("dm", DM_COLUMNS, self.dm), ("scatter", SCATTER_COLUMNS, self.scatter)):
                g = f.create_group(gname)
                g.attrs["n"] = len(table["event"])
                for name, arr in table.items():
                    kw = {"compression": compression} if compression and arr.size else {}
                    ds = g.create_dataset(name, data=arr, **kw)
                    ds.attrs["description"] = (cols.get(name) or OPTIONAL_SCATTER[name])[3]
                    if name == "parent_kind":
                        ds.attrs["codes"] = json.dumps(PARENT_KINDS)
                    if name == "vertex_det":
                        ds.attrs["frame"] = json.dumps(self.frame)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path, columns=None):
        """Read a record. `columns` (iterable of names) limits what is read from both tables."""
        import h5py
        with h5py.File(path, "r") as f:
            schema = f.attrs.get("schema")
            if schema != SCHEMA:
                raise ValueError(f"{path}: schema {schema!r}, this ldmgen reads {SCHEMA!r}")
            run = RunMeta.from_json(f.attrs["run"])
            tables, frame = [], {}
            for gname in ("dm", "scatter"):
                g = f[gname]
                tables.append({k: g[k][()] for k in g if columns is None or k in columns or k == "event"})
                if "vertex_det" in g:
                    frame = json.loads(g["vertex_det"].attrs["frame"])
            extra = {k: f["dm_extra"][k][()] for k in f["dm_extra"]} if "dm_extra" in f and columns is None else {}
        rec = cls(run, *tables, frame, extra)
        if columns is None:
            rec.validate()
        return rec


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def source_entry(path, fmt):
    p = Path(path)
    return {"path": str(p.resolve()), "format": fmt, "sha256": sha256(p)}
