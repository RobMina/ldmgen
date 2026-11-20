"""HepMC3 ASCII (Asciiv3) writer with the thin LdmHepMC attribute profile.

Public interchange layer (agreed 2026-11-20): generators and analysis keep `ldmgen/1` HDF5;
HepMC3 + LdmHepMC is the shareable particle-level graph. Experiment entry points (HEPEVT /
TextFileGen / edep-sim / Mu2e GenParticle) remain sinks — see `ldmgen.hepevt` and `ldmgen.export`.

This writer is pure Python (no pyhepmc / HepMC3 C++). It emits HepMC3 Asciiv3 text that a
HepMC3-aware reader can parse. LdmHepMC attributes mirror NuHepMC-style run/event metadata.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import __version__

LDMHEPMC_VERSION = "0.1"
CHI_PDG = 51  # complex-scalar / UFO convention used by MadDump VectorPortal_*DM


def _attr(level: str, key: str, value) -> str:
    """One Asciiv3 attribute line. `level` is 'R' (run) or an event index string for 'A'."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        text = repr(float(value))
    else:
        text = str(value).replace('"', "'")
    if level == "R":
        return f'A 0 {key} "{text}"'
    return f'A {level} {key} "{text}"'


def _run_attrs(rec) -> list[str]:
    run = rec.run
    m = run.model
    lines = [
        _attr("R", "LdmHepMC.Version", LDMHEPMC_VERSION),
        _attr("R", "LdmHepMC.Schema", "ldmgen/1"),
        _attr("R", "LdmHepMC.Generator.Name", run.generator),
        _attr("R", "LdmHepMC.Generator.Version", run.generator_version or ""),
        _attr("R", "LdmHepMC.Generator.ImageSha256", run.image_sha256 or ""),
        _attr("R", "LdmHepMC.Model.Name", m.name),
        _attr("R", "LdmHepMC.Model.Epsilon", m.epsilon),
        _attr("R", "LdmHepMC.Model.AlphaD", m.alpha_D),
        _attr("R", "LdmHepMC.Model.MChi", m.m_chi),
        _attr("R", "LdmHepMC.Model.MV", m.m_V),
        _attr("R", "LdmHepMC.Model.DmSpin", m.dm_spin),
        _attr("R", "LdmHepMC.POT", run.pot),
        _attr("R", "LdmHepMC.DMSample", run.dm_sample),
        _attr("R", "LdmHepMC.Detector", run.detector or ""),
        _attr("R", "LdmHepMC.Production", run.production or ""),
        _attr("R", "LdmHepMC.Beam.Origin", (run.beam or {}).get("origin", "")),
        _attr("R", "LdmHepMC.Tools.LdmgenVersion", run.ldmgen_version or __version__),
    ]
    return lines


def _p_line(pid: int, pdg: int, status: int, p4, mass: float, parents=(0, 0)) -> str:
    e, px, py, pz = (float(x) for x in p4)
    m1, m2 = parents
    return f"P {pid} {m1} {pdg} {px} {py} {pz} {e} {mass} {status} 0"


def _v_line(vid: int, vertex, status: int = 1) -> str:
    x, y, z, t = (float(v) if np.isfinite(v) else 0.0 for v in vertex)
    # HepMC3 Asciiv3: V id status x y z t
    return f"V -{vid} {status} {x} {y} {z} {t}"


def write(rec, path, *, max_events: int = -1, chi_pdg: int = CHI_PDG) -> dict:
    """Write a HepMC3 Asciiv3 file with LdmHepMC run attributes.

    Prefer scatter rows when present (one HepMC event per scatter: incoming chi,
    recoil electron/nucleon, outgoing chi). Otherwise emit one event per `/dm` row
    (production-only samples).

    Four-momenta are GeV; vertices are metres and seconds in the beam frame (same as
    `ldmgen/1`). Units are declared as GEV MM for HepMC3 compatibility; vertex
    coordinates are converted to millimetres and times to millimetres/c (c = 299.792458 mm/ns).
    """
    path = Path(path)
    c_mm_per_s = 299792458000.0  # mm / s

    def vtx_mm_ns(v):
        x, y, z, t = (float(a) if np.isfinite(a) else 0.0 for a in v)
        return (x * 1000.0, y * 1000.0, z * 1000.0, t * c_mm_per_s)

    lines = [
        "HepMC::Version 3.02.05",
        "HepMC::Asciiv3-START_EVENT_LISTING",
        *_run_attrs(rec),
    ]

    n_events = 0
    if rec.n_scatter > 0:
        sc = rec.scatter
        events = np.unique(sc["event"])
        if max_events is not None and max_events >= 0:
            events = events[:max_events]
        for ev in events:
            m = sc["event"] == ev
            idxs = np.flatnonzero(m)
            lines.append(f"E {n_events} 0 {len(idxs)}")
            lines.append("U GEV MM")
            lines.append(_attr(str(n_events), "LdmHepMC.Weight", float(sc["weight"][idxs].sum())))
            lines.append(_attr(str(n_events), "LdmHepMC.RecordEvent", int(ev)))
            pid = 1
            for j, i in enumerate(idxs):
                vid = j + 1
                lines.append(_v_line(vid, vtx_mm_ns(sc["vertex"][i])))
                # incoming chi (beam-like status 4), recoil visible (1), outgoing chi (1)
                dm_i = int(sc["dm_index"][i]) if "dm_index" in sc else -1
                if dm_i >= 0 and dm_i < rec.n_dm:
                    chi_in = rec.dm["p4"][dm_i]
                    m_chi = float(rec.run.model.m_chi)
                else:
                    # reconstruct from recoil + out chi when dm_index missing
                    chi_in = sc["chi_out_p4"][i] + sc["out_p4"][i]
                    m_chi = float(rec.run.model.m_chi)
                lines.append(_p_line(pid, chi_pdg, 4, chi_in, m_chi, parents=(0, 0)))
                pid += 1
                out_pdg = int(sc["out_pdg"][i])
                out_mass = float(sc["out_mass"][i]) if "out_mass" in sc and np.isfinite(sc["out_mass"][i]) else 0.0
                lines.append(_p_line(pid, out_pdg, 1, sc["out_p4"][i], out_mass, parents=(-vid, -vid)))
                pid += 1
                lines.append(_p_line(pid, chi_pdg, 1, sc["chi_out_p4"][i], m_chi, parents=(-vid, -vid)))
                pid += 1
            n_events += 1
    else:
        dm = rec.dm
        n = rec.n_dm
        order = np.arange(n) if max_events is None or max_events < 0 else np.arange(min(n, max_events))
        for i in order:
            lines.append(f"E {n_events} 0 1")
            lines.append("U GEV MM")
            lines.append(_attr(str(n_events), "LdmHepMC.Weight", float(dm["weight"][i])))
            lines.append(_attr(str(n_events), "LdmHepMC.RecordEvent", int(dm["event"][i])))
            lines.append(_v_line(1, vtx_mm_ns(dm["vertex"][i])))
            lines.append(_p_line(1, chi_pdg, 1, dm["p4"][i], float(rec.run.model.m_chi), parents=(-1, -1)))
            n_events += 1

    lines.append("HepMC::Asciiv3-END_EVENT_LISTING")
    path.write_text("\n".join(lines) + "\n")
    return {
        "format": "hepmc3-asciiv3",
        "ldmhepmc_version": LDMHEPMC_VERSION,
        "path": str(path),
        "events": n_events,
        "source": "scatter" if rec.n_scatter > 0 else "dm",
        "writer": "ldmgen.hepmc3.write",
    }
